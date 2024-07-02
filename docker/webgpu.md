# Docker + WebGPU preview

Docker is the leading solution for packaging and deploying portable applications. However, for AI and LLM
workloads, Docker containers are often not portable due to the lack of GPU abstraction -- you will need
a different container image for each GPU / driver combination. In some cases, the GPU is simply not
accessible from inside containers. For example, the "impossible triangle of LLM app, Docker, and Mac GPU"
refers to the lack of Mac GPU access from containers.

Docker is supporting the WebGPU API for container apps. It will allow any underlying GPU or accelerator 
hardware to be accessed through WebGPU. That means container apps just need to write to the WebGPU API
and they will automatically become portable across all GPUs supported by Docker.
However, asking developers to rewrite existing LLM apps, which use the CUDA or Metal or other GPU APIs, 
to WebGPU is a challenge.

LlamaEdge provides an ecosystem of portable AI / LLM apps and components 
that can run on multiple inference backends including the WebGPU.
It supports any programming language that can be compiled into Wasm, such as Rust.
Furthermore, LlamaEdge apps are lightweight and binary portable across different CPUs and OSes, making it an ideal 
runtime to embed into container images.

> Based on the [WasmEdge runtime](https://github.com/WasmEdge/WasmEdge), LlamaEdge features a pluggable architecture that can easily switch between different inference backends without changing the compiled Wasm binary files.

In this article, we will showcase an [OpenAI-compatible speech-to-text API server](https://platform.openai.com/docs/guides/speech-to-text) implemented in LlamaEdge and running inside Docker taking advantage of GPUs through the WebGPU API. 

## Install Docker Desktop Preview with WebGPU

Download the preview Docker Desktop software:

* Mac with Apple Silicon (M series): https://desktop-stage.docker.com/mac/main/arm64/155220/Docker.dmg
* Linux with x86 CPU and any GPU:
  * https://desktop-stage.docker.com/linux/main/amd64/155220/docker-desktop-amd64.deb
  * https://desktop-stage.docker.com/linux/main/amd64/155220/docker-desktop-x86_64.rpm

> On Linux follow steps 1 and 3 from https://docs.docker.com/desktop/install/ubuntu/#install-docker-desktop to install the downloaded package.

Go to Settings,

* In "General", turn on `containerd` support
* In "Features in development", turn on "Enable Wasm"

## Run the API server as a container

Pull the pre-made container image from Docker hub and run it.

```
docker run \
  --runtime=io.containerd.wasmedge.v1 \
  --platform=wasi/wasm \
  --device="docker.com/gpu=webgpu" \
  --env WASMEDGE_WASINN_PRELOAD=default:Burn:GPU:/tiny_en.mpk:/tiny_en.cfg:/tokenizer.json:en \
  -p 8080:8080 \
  secondstate/burn-whisper-server:latest
```

The [Docker image](https://hub.docker.com/r/secondstate/burn-whisper-server/tags) is only 90MB and it contains the entire model, runtime, and the API server.
It is also important to note that the image is for the `wasi/wasm` architecture. It can run on any OS and CPU
platform Docker supports.

## Use the API server

The API server is [OpenAI-compatible](https://platform.openai.com/docs/guides/speech-to-text).
You can use HTTP POST to submit a `.wav` file to transcribe. 
You can use [this file](https://huggingface.co/second-state/whisper-burn/resolve/main/audio16k.wav) as an example.

```
curl -LO https://huggingface.co/second-state/whisper-burn/resolve/main/audio16k.wav
```

You can now make an API request to the server.

```
curl http://localhost:8080/v1/audio/transcriptions \
  -H "Content-Type: multipart/form-data" \
  -F file="@audio16k.wav"
```

The result is as follows.

```
{
    "text": " Hello, I am the whisper machine learning model. If you see this as text then I am working properly."
}
```

### Create your own audio file

The current demo requires `.wav` file in a specific format. 
It should use `lpcm` and the sample rate should be `16000.0`.

The [yt-dlp](https://github.com/yt-dlp/yt-dlp) program can download YouTube audio track in the above format.

```
yt-dlp -f bestaudio --extract-audio --audio-format wav --postprocessor-args "-ss 25 -t 10 -ar 16000 -ac 1" -o "output.wav" "https://www.youtube.com/watch?v=UF8uR6Z6KLc"
```

## Build and publish the API server

The source code for the API server is [here](https://github.com/LlamaEdge/whisper-api-server/). 
It uses WasmEdge's [burn](https://github.com/second-state/wasmedge-burn-plugin) plugin to run
inference operations via WebGPU. But its source code has no dependency on `burn`. Instead, it uses the standard
and portable WASI-NN inferface to interact with the underlying inference runtime.
You can simply compile the Rust project to wasm.

```
cargo build --release --target wasm32-wasi
cp target/wasm32-wasi/release/whisper-api-server.wasm  .
```

Download the whispter AI model files for speech-to-text inference.

```
curl -LO https://huggingface.co/second-state/whisper-burn/resolve/main/tiny_en.tar.gz
tar -xvzf tiny_en.tar.gz
```

Use the following `Dockerfile` to build the image.

```
FROM scratch

# Copy the prepared files from the current directory to the image
COPY tiny_en.cfg /tiny_en.cfg
COPY tiny_en.mpk /tiny_en.mpk
COPY tokenizer.json /tokenizer.json
COPY whisper-api-server.wasm /app.wasm

# Set the entrypoint
ENTRYPOINT [ "/app.wasm" ]
```

Build and publish the Docker image.

```
docker build . --platform wasi/wasm -t secondstate/burn-whisper-server:latest
docker push secondstate/burn-whisper-server:latest
```


