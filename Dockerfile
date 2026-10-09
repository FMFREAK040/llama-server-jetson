# ---- build stage ----
FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0 AS build
RUN apt-get update && apt-get install -y git cmake build-essential libcurl4-openssl-dev curl gnupg
RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp /src
WORKDIR /src
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
RUN cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 \
      -DLLAMA_CURL=ON -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
      -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,/usr/local/cuda/lib64/stubs" \
      -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath-link,/usr/local/cuda/lib64/stubs" \
 && cmake --build build -j$(nproc) --target llama-server \
 && mkdir -p /out && cp -a build/bin/. /out/

# L4T driver userspace (libcuda etc.), unpacked without maintainer scripts
RUN curl -fsSL https://repo.download.nvidia.com/jetson/jetson-ota-public.asc \
      | gpg --dearmor -o /usr/share/keyrings/jetson.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/jetson.gpg] https://repo.download.nvidia.com/jetson/t234 r36.4 main" \
      > /etc/apt/sources.list.d/l4t.list \
 && apt-get update && mkdir /l4t && cd /l4t \
 && apt-get download nvidia-l4t-core nvidia-l4t-cuda \
 && for d in *.deb; do dpkg-deb -x "$d" /l4t/root; done

# ---- runtime stage ----
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y libcurl4 libgomp1 && rm -rf /var/lib/apt/lists/*
COPY --from=build /l4t/root/usr/lib/aarch64-linux-gnu/ /usr/lib/aarch64-linux-gnu/
COPY --from=build /usr/local/cuda/lib64/libcudart.so* /usr/local/cuda/lib64/libcublas*.so* /usr/local/cuda/lib64/
COPY --from=build /out/ /app/
RUN echo "/usr/lib/aarch64-linux-gnu/nvidia" > /etc/ld.so.conf.d/nvidia.conf \
 && echo "/usr/local/cuda/lib64" > /etc/ld.so.conf.d/cuda.conf \
 && echo "/app" > /etc/ld.so.conf.d/llama.conf && ldconfig
ENTRYPOINT ["/app/llama-server"]
