# Getting started with Docker

> For Nvidia devices: replace the `latest` tag with `cuda12` or `cuda11`. If you need to build the images yourself, replace `Dockerfile` with `Dockerfile.cuda11` or `Dockerfile.cuda12`.

## Start a chatbot server

First, download an LLM chat model file. You can find many of them [here](https://huggingface.co/second-state).
Here we use Qwen-2's 0.5B model as an example. It is a very small but capable model. You can go up to LLMs
of any size as long as you allocate enough memory for your Docker container.

```
curl -LO https://huggingface.co/second-state/Qwen2-0.5B-Instruct-GGUF/resolve/main/Qwen2-0.5B-Instruct-Q5_K_M.gguf
```

Start the server in a container as follows. 
The arguments following `secondstate/llamaedge` are 
the chat LLM file name (the GGUF file name), 
the prompt template for this model, 
and the context size you would like to support (limited by the max context size of this model).

```
docker run --rm -p 8080:8080 -v $(pwd):/models:z --name llamaedge secondstate/llamaedge:latest Qwen2-0.5B-Instruct-Q5_K_M.gguf chatml 1024
```

Go to http://localhost:8080 from your browser to chat with the model!

## Start an OpenAI compaitible server

A fully featured OpenAI compaitible server requires not only a chat LLM but also an embedding model.
That allows the server to support both `/chat` and `/embedding` endpoints, which are crucial for most
LLM agent apps and frameworks based on OpenAI.

```
curl -LO https://huggingface.co/second-state/Qwen2-0.5B-Instruct-GGUF/resolve/main/Qwen2-0.5B-Instruct-Q5_K_M.gguf
curl -LO https://huggingface.co/second-state/Nomic-embed-text-v1.5-Embedding-GGUF/resolve/main/nomic-embed-text-v1.5-f16.gguf
```

Start the server in a container as follows. 
The arguments following `secondstate/llamaedge` are 
the chat LLM file name (the GGUF file name), 
the prompt template for this model, 
the context size you would like to support (limited by the max context size of this model),
the embedding model file name (also in GGUF),
and the context size for the embedding model.

```
docker run --rm -p 8080:8080 -v $(pwd):/models:z --name llamaedge secondstate/llamaedge:latest Qwen2-0.5B-Instruct-Q5_K_M.gguf chatml 1024 nomic-embed-text-v1.5-f16.gguf 512
```

You can still access the server as a chatbot at http://localhost:8080. You can also make an OpenAI style API
request as follows.

```
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "Where is Paris?"}]}'
```

## Stop the server

```
docker container stop llamaedge
```

## Build your own Docker image locally

```
docker build . --tag secondstate/llamaedge:latest -f Dockerfile
```

Cross-platform build.

```
docker buildx build . --platform linux/arm64,linux/amd64 --tag secondstate/llamaedge:latest -f Dockerfile
```

Publish to Docker hub.

```
docker login
docker push secondstate/llamaedge:latest
```
