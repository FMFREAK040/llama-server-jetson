# llama.cpp server with CUDA for NVIDIA Jetson Orin (Orin Nano / Orin Nano Super)
# Target: L4T r36.4 (JetPack 6.2), CUDA 12.6, GPU compute capability 8.7, Cortex-A78AE CPU
# Build on arm64:  docker buildx build --platform linux/arm64 -t <image> --push .

ARG L4T_IMAGE_TAG=r36.4.0

# =========================== build stage ===========================
FROM nvcr.io/nvidia/l4t-jetpack:${L4T_IMAGE_TAG} AS build

# L4T apt repo release (must match the L4T version of the node's nvgpu module)
ARG L4T_RELEASE=r36.4
# llama.cpp git branch or tag to build
ARG LLAMA_CPP_REF=master

RUN apt-get update && apt-get install -y --no-install-recommends \
      git cmake build-essential libssl-dev curl gnupg ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch ${LLAMA_CPP_REF} https://github.com/ggml-org/llama.cpp /src
WORKDIR /src

# Let the linker resolve libcuda.so.1 against the CUDA stub at build time
# (the real libcuda comes from the L4T packages below)
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1

# GGML_NATIVE=OFF + explicit ARM arch: don't optimize for the build machine's CPU,
# target the Orin's Cortex-A78AE instead (avoids SIGILL / exit code 132)
RUN cmake -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=87 \
      -DGGML_NATIVE=OFF \
      -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16 \
      -DLLAMA_OPENSSL=ON \
      -DLLAMA_BUILD_TESTS=OFF \
      -DLLAMA_BUILD_EXAMPLES=OFF \
      -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,/usr/local/cuda/lib64/stubs" \
      -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath-link,/usr/local/cuda/lib64/stubs" \
 && cmake --build build -j"$(nproc)" --target llama-server \
 && mkdir -p /out && cp -a build/bin/. /out/

# Jetson driver userspace (libcuda, libnvrm*, ...), unpacked without running
# the packages' maintainer scripts (they expect a real Jetson host)
RUN curl -fsSL https://repo.download.nvidia.com/jetson/jetson-ota-public.asc \
      | gpg --dearmor -o /usr/share/keyrings/jetson.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/jetson.gpg] https://repo.download.nvidia.com/jetson/t234 ${L4T_RELEASE} main" \
      > /etc/apt/sources.list.d/l4t.list \
 && apt-get update \
 && mkdir -p /l4t/root && cd /l4t \
 && apt-get download nvidia-l4t-core nvidia-l4t-cuda \
 && for d in *.deb; do dpkg-deb -x "$d" /l4t/root; done

# ========================== runtime stage ==========================
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
      libssl3 ca-certificates libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Jetson driver userspace
COPY --from=build /l4t/root/usr/lib/aarch64-linux-gnu/ /usr/lib/aarch64-linux-gnu/
# CUDA runtime + cuBLAS
COPY --from=build /usr/local/cuda/lib64/libcudart.so* /usr/local/cuda/lib64/
COPY --from=build /usr/local/cuda/lib64/libcublas*.so* /usr/local/cuda/lib64/
# llama-server and its ggml/llama shared libraries
COPY --from=build /out/ /app/

RUN echo "/usr/lib/aarch64-linux-gnu/nvidia" > /etc/ld.so.conf.d/nvidia.conf \
 && echo "/usr/local/cuda/lib64"            > /etc/ld.so.conf.d/cuda.conf \
 && echo "/app"                             > /etc/ld.so.conf.d/llama.conf \
 && ldconfig

ENV LLAMA_ARG_HOST=0.0.0.0 \
    LLAMA_ARG_PORT=8080 \
    LLAMA_CACHE=/models

EXPOSE 8080
ENTRYPOINT ["/app/llama-server"]
