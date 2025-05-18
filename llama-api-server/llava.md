# Vision Models

LlamaEdge provides support for a collection of open-source vision models, such as [Gemma-3-it](https://huggingface.co/collections/second-state/gemma-3-it-gguf-models-67d18b5fb8e881054276af2b) and [Qwen2.5-VL](https://huggingface.co/collections/second-state/qwen25-vl-gguf-models-6829541a1443e849614a9fa0). The following sections demonstrate how to run vision models using [LlamaEdge 0.18.5](https://github.com/LlamaEdge/LlamaEdge/releases/tag/0.18.5) or later versions, using the Qwen2.5-VL model as an example.

## Setup

### Install WasmEdge Runtime

- CPU Only

  ```bash
  # Version of WasmEdge Runtime
  export wasmedge_version="0.14.1"

  # Version of ggml plugin
  export ggml_plugin="b5361"

  # For CPU
  curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install_v2.sh | bash -s -- -v $wasmedge_version --ggmlbn=$ggml_plugin
  ```

- GPU

  ```bash
  # Version of WasmEdge Runtime
  export wasmedge_version="0.14.1"

  # Version of ggml plugin
  export ggml_plugin="b5361"

  # CUDA version: 11 or 12
  export ggmlcuda=12

  curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install_v2.sh | bash -s -- -v $wasmedge_version --ggmlbn=$ggml_plugin --ggmlcuda=$ggmlcuda
  ```

### Download llama-api-server

```bash
# Version of llama-api-server to use
export api_server_version="0.18.5"

# Download
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/download/$api_server_version/llama-api-server.wasm
```

### Download vision model

For this example, we'll use [second-state/Qwen2.5-VL-7B-Instruct-GGUF](https://huggingface.co/second-state/Qwen2.5-VL-7B-Instruct-GGUF/blob/main/Qwen2.5-VL-7B-Instruct-Q5_K_M.gguf):

```bash
curl -LO https://huggingface.co/second-state/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/Qwen2.5-VL-7B-Instruct-Q5_K_M.gguf
```

## Running the Server

### Start llama-api-server

Execute the following command to start llama-api-server:

```bash
wasmedge --dir .:. \
  --nn-preload default:GGML:AUTO:Qwen2.5-VL-7B-Instruct-Q5_K_M.gguf \
  llama-api-server.wasm \
  --model-name Qwen2.5-VL-7B-Instruct \
  --prompt-template qwen2-vision \
  --llava-mmproj Qwen2.5-VL-7B-Instruct-vision.gguf \
  --ctx-size 32000
```

Upon successful execution, you should see output similar to the following:

```bash
[2025-05-18 11:23:09.970] [info] llama_api_server in llama-api-server/src/main.rs:202: LOG LEVEL: info
[2025-05-18 11:23:09.973] [info] llama_api_server in llama-api-server/src/main.rs:205: SERVER VERSION: 0.18.5
[2025-05-18 11:23:09.976] [info] llama_api_server in llama-api-server/src/main.rs:544: model_name: Qwen2.5-VL-7B-Instruct

...

[2025-05-18 11:23:10.531] [info] llama_api_server in llama-api-server/src/main.rs:917: plugin_ggml_version: b5361 (commit cf0a43bb)
[2025-05-18 11:23:10.533] [info] llama_api_server in llama-api-server/src/main.rs:952: Listening on 0.0.0.0:8080
```

### Sending Image Requests

The following command demonstrates how to send a CURL request to llama-api-server. The request includes a base64-encoded string of an image in the `image_url` field. For demonstration purposes, only a portion of the base64 string is shown here. In practice, you should use the complete base64 string. The full base64 string used in the following request can be found in [image_b64.txt](../assets/image_b64.txt).

```bash
curl --location 'http://localhost:8080/v1/chat/completions' \
--header 'Content-Type: application/json' \
--data '{
    "messages": [
        {
            "content": [
                {
                    "type": "text",
                    "text": "Describe the picture"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "/9j/4AAQSkZJRgABAQAASABIAAD ... knr+Vb+lWR8oTTNwfujOc/hSuhuSsf//Z"
                    }
                }
            ],
            "role": "user"
        }
    ],
    "model": "Qwen2.5-VL-7B-Instruct"
}'
```

If the request is processed successfully, you will receive a response similar to the following:

```bash
{
    "id": "chatcmpl-4367085d-6451-4896-bbd8-a5090604394d",
    "object": "chat.completion",
    "created": 1747369554,
    "model": "Qwen2-VL-2B-Instruct",
    "choices": [
        {
            "index": 0,
            "message": {
                "content": "mixed berries in a paper bowl",
                "role": "assistant"
            },
            "finish_reason": "stop",
            "logprobs": null
        }
    ],
    "usage": {
        "prompt_tokens": 27,
        "completion_tokens": 8,
        "total_tokens": 35
    }
}
```
