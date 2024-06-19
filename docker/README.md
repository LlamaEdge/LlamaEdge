# Getting started with Docker

You can run all the commands in this document without any change on any machine with the latest Docker and at least 8GB of RAM available to the container.
By default, the container uses the CPU to peform computations, which could be slow for large LLMs. For GPUs,

* Mac: Everything here works on [Docker Desktop for Mac](https://docs.docker.com/desktop/install/mac-install/). However, the Apple GPU cores will not be available inside Docker containers until [WebGPU is supported by Docker](webgpu.md) later in 2024.
* Windows and Linux with Nvidia GPU: You will need to install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installation) for Docker. In the instructions below, replace the `latest` tag with `cuda12` or `cuda11`, and add the `--device nvidia.com/gpu=all` flag, to use take advantage of the GPU. If you need to build the images yourself, replace `Dockerfile` with `Dockerfile.cuda12` or `Dockerfile.cuda11`.

## Quick start

Run the following Docker command to start an OpenAI-compatible LLM API server on your own device.

```
docker run --rm -p 8080:8080 --name api-server secondstate/qwen-2-0.5b-allminilm-2:latest
```

Go to http://localhost:8080 from your browser to chat with the model!

This container starts two models Qwen-2-0.5B is a very small but highly capable LLM chat model, and all-miniLM is 
a widely used embedding model. 
That allows the API server to support both `/chat` and `/embeddings` endpoints, which are crucial for most
LLM agent apps and frameworks based on OpenAI.

Alternatively, you can use the command below to start a server on an Nvidia CUDA 12 machine.

```
docker run --rm -p 8080:8080 --device nvidia.com/gpu=all --name api-server secondstate/qwen-2-0.5b-allminilm-2:cuda12
```

You can make an OpenAI style API request as follows.

```
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "Where is Paris?"}]}'
```

Or, make an embedding request to turn a collection of text paragraphs into vectors. It is required for many RAG apps.

```
curl -X POST http://localhost:8080/v1/embeddings \
    -H 'accept:application/json' \
    -H 'Content-Type: application/json' \
    -d '{"model":"all-MiniLM-L6-v2-ggml-model-f16.gguf", "input":["Paris is the capital of France.","Paris occupies a central position in the rich agricultural region of 890 square miles (2,300 square km).","The population of Paris is 2,145,906"]}'
```

Stop and remove the container once you are done.

```
docker stop api-server
```

## Specify context window sizes

The memory consumption of the container is dependent on the context size you give to the model. You can specify the context size by appending two arguments at the end of the command. The following command starts the container with a context window of 1024 tokens for the chat LLM and a context window of 256 tokens for the embedding model. 

```
docker run --rm -p 8080:8080 --name api-server secondstate/qwen-2-0.5b-allminilm-2:latest ctx-size 1024 256
```

Each model comes with a maximum context size it can support. Your custom context size should not exceed that. Please refer to model documentation for this information. 

> If you set the embedding context size (i.e., the last argument in the above command) to 0, the container would load the chat LLM only.

## Build your own image

You can build nad publish a Docker image to use any models you like. First, download the model files (must be in GGUF format) you want from Huggingface. 
Of course, you could also your private finetuned model files here. 

```
curl -LO https://huggingface.co/second-state/Qwen2-0.5B-Instruct-GGUF/resolve/main/Qwen2-0.5B-Instruct-Q5_K_M.gguf
curl -LO https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF/resolve/main/all-MiniLM-L6-v2-ggml-model-f16.gguf
```

Build a multi-platform image by passing the model files as `--build-arg`. The `PROMPT_TEMPLATE` is the specific text format the chat model is trained on to follow conversations. It differs for each model, and you will need to special attention. For all models published by the second-state organization, you can find the prompt-template in the model card. 

```
docker buildx build . --platform linux/arm64,linux/amd64 \
  --tag secondstate/qwen-2-0.5b-allminilm-2:latest -f Dockerfile \
  --build-arg CHAT_MODEL_FILE=Qwen2-0.5B-Instruct-Q5_K_M.gguf \
  --build-arg EMBEDDING_MODEL_FILE=all-MiniLM-L6-v2-ggml-model-f16.gguf \
  --build-arg PROMPT_TEMPLATE=chatml
```

Once it is built, you can publish it to Docker Hub.

```
docker login
docker push secondstate/qwen-2-0.5b-allminilm-2:latest
```

## What's next

Use the container as a drop-in replacement for the OpenAI API for your favorite agent app or framework! [See some examples here](https://llamaedge.com/docs/category/drop-in-replacement-for-openai). 

## A production-ready example

The quick start example uses very small models. They are great to get started but most production use cases require 
at least a 7B class LLM.
Let's build a new image for the Llama-3-8B chat LLM and the Nomic-1.5 embedding model. Both models are highly rated.
Download the two models in GGUF format.

```
curl -LO https://huggingface.co/second-state/Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q5_K_M.gguf
curl -LO https://huggingface.co/second-state/Nomic-embed-text-v1.5-Embedding-GGUF/resolve/main/nomic-embed-text-v1.5-f16.gguf
```

Build the image using the downloaded model files. The prompt template is `llama-3-chat`.

```
docker buildx build . --platform linux/arm64,linux/amd64 \
  --tag secondstate/llama-3-8b-nomic-1.5:latest -f Dockerfile \
  --build-arg CHAT_MODEL_FILE=Meta-Llama-3-8B-Instruct-Q5_K_M.gguf \
  --build-arg EMBEDDING_MODEL_FILE=nomic-embed-text-v1.5-f16.gguf \
  --build-arg PROMPT_TEMPLATE=llama-3-chat
```

Run an API server container using the models.
You will need at least 8GB of RAM for the container.

```
docker run --rm -p 8080:8080 --name api-server secondstate/llama-3-8b-nomic-1.5:latest
```

Run both models at their maximum context sizes. You will need at least 32GB of RAM for the container in order
to support these content sizes.

```
docker run --rm -p 8080:8080 --name api-server secondstate/llama-3-8b-nomic-1.5:latest ctx-size 8192 8192
```

## Why LlamaEdge?

LlamaEdge supports multiple types of models and their corresponding OpenAI-compatible APIs. 
We have seen LLM chat and embedding APIs here. 
It also supports multimodal vision models like [Llava](https://www.secondstate.io/articles/llava-v1.6-vicuna-7b/), 
[voice-to-text](https://github.com/WasmEdge/WasmEdge/issues/3170) models like Whisper, and text-to-image models like Stable Diffusion.

LlamaEdge supports multiple underlying GPU or CPU drivers and execution frameworks.

* GGML for generic Mac (Apple Silicon), Nvidia, and all CPUs
* [MLX](https://github.com/WasmEdge/WasmEdge/issues/3266) for advanced Apple Silicon
* TensorRT for advanced Nvidia
* [Intel Neural Speed](https://github.com/second-state/WasmEdge-WASINN-examples/pull/135) for advanced Intel CPUs
* Experimental WebGPU support in Docker

Finally, LlamaEdge enables developers to create applications on top of its component APIs. 
For example, [GaiaNet](https://docs.gaianet.ai/) built personalized and decentralized RAG services. 
The [Moxin](https://github.com/moxin-org/moxin) project is a rich Chatbot UI. Both projects are based on the LlamaEdge Rust SDK. 
The [OpenInterpreter](https://github.com/OpenInterpreter/01) project embeds LlamaEdge as an LLM provider in their Python app.
You can use model-specific Docker images as base images for your own apps.

