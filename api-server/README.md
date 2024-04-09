# LlamaEdge API Server

LlamaEdge API server offers OpenAI-compatible REST APIs. It can accelerate developers to build LLM-driven applications, AI infrastructure, and etc. In addition, LlamaEdge is also friendly to AI frameworks, such as LangChain and LlamaIndex.

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [LlamaEdge API Server](#llamaedge-api-server)
  - [Dependencies](#dependencies)
  - [Get LlamaEdge API server](#get-llamaedge-api-server)
  - [Get model](#get-model)
  - [Run LlamaEdge API server](#run-llamaedge-api-server)
  - [Endpoints](#endpoints)
    - [`/v1/models` endpoint for model list](#v1models-endpoint-for-model-list)
    - [`/v1/chat/completions` endpoint for chat completions](#v1chatcompletions-endpoint-for-chat-completions)
    - [`/v1/files` endpoint for uploading text and markdown files](#v1files-endpoint-for-uploading-text-and-markdown-files)
    - [`/v1/chunks` endpoint for segmenting files to chunks](#v1chunks-endpoint-for-segmenting-files-to-chunks)
    - [`/v1/embeddings` endpoint for computing embeddings](#v1embeddings-endpoint-for-computing-embeddings)
  - [Add a web UI](#add-a-web-ui)
  - [CLI options for the API server](#cli-options-for-the-api-server)

<!-- /code_chunk_output -->

## Dependencies

Install the latest WasmEdge with plugins:

<details> <summary> For macOS (apple silicon) </summary>

```console
# install WasmEdge with wasi-nn-ggml plugin
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | bash -s -- --plugin wasi_nn-ggml

# Assuming you use zsh (the default shell on macOS), run the following command to activate the environment
source $HOME/.zshenv
```

</details>

<details> <summary> For Ubuntu (>= 20.04) </summary>

```console
# install libopenblas-dev
apt update && apt install -y libopenblas-dev

# install WasmEdge with wasi-nn-ggml plugin
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | bash -s -- --plugin wasi_nn-ggml

# Assuming you use bash (the default shell on Ubuntu), run the following command to activate the environment
source $HOME/.bashrc
```

</details>

<details> <summary> For General Linux </summary>

```console
# install WasmEdge with wasi-nn-ggml plugin
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | bash -s -- --plugin wasi_nn-ggml

# Assuming you use bash (the default shell on Ubuntu), run the following command to activate the environment
source $HOME/.bashrc
```

</details>

## Get LlamaEdge API server

- Download LlamaEdge API server with the support for `HTTP` scheme only:

  ```console
  curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-api-server.wasm
  ```

- Download LlamaEdge API server with the support for `HTTP` and `WebSocket` schemes:

  ```console
  curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-api-server-full.wasm
  ```

## Get model

[huggingface.co/second-state](https://huggingface.co/second-state?search_models=GGUF) maintains a group of GGUF models for different usages. In addition, you can also pick a GGUF model from the [https://huggingface.co/models?sort=trending&search=gguf](https://huggingface.co/models?sort=trending&search=gguf).

## Run LlamaEdge API server

Run the API server with the following command:

```bash
wasmedge --dir .:. \
    --nn-preload default:GGML:AUTO:llama-2-7b-chat.Q5_K_M.gguf \
    llama-api-server.wasm \
    --prompt-template llama-2-chat \
    --model-name llama-2-7b-chat
```

The command above starts the API server on the default socket address. Besides, there are also some other options specified in the command:

- The `--dir .:.` option specifies the current directory as the root directory of the WASI file system.

- The `--nn-preload default:GGML:AUTO:llama-2-7b-chat.Q5_K_M.gguf` option specifies the Llama model to be used by the API server. The pattern of the argument is `<name>:<encoding>:<target>:<model path>`. Here, the model used is `llama-2-7b-chat.Q5_K_M.gguf`; and we give it an alias `default` as its name in the runtime environment. You can change the model name here if you're not using llama2-7b-chat
- The `--prompt-template llama-2-chat` is the prompt template for the model.
- The `--model-name llama-2-7b-chat` specifies the model name. It is used in the chat request.

## Endpoints

### `/v1/models` endpoint for model list

`llama-api-server` provides a POST API `/v1/models` to list currently available models. You can use `curl` to test it on a new terminal:

```bash
curl -X POST http://localhost:8080/v1/models -H 'accept:application/json'
```

If the command is successful, you should see the similar output as below in your terminal:

```bash
{
    "object":"list",
    "data":[
        {
            "id":"llama-2-7b-chat",
            "created":1697084821,
            "object":"model",
            "owned_by":"Not specified"
        }
    ]
}
```

### `/v1/chat/completions` endpoint for chat completions

Ask a question using OpenAI's JSON message format.

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
    -H 'accept:application/json' \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "Who is Robert Oppenheimer?"}], "model":"llama-2-7b-chat"}'
```

Here is the response.

```bash
{
    "id":"",
    "object":"chat.completion",
    "created":1697092593,
    "model":"llama-2-7b-chat",
    "choices":[
        {
            "index":0,
            "message":{
                "role":"assistant",
                "content":"Robert Oppenheimer was an American theoretical physicist and director of the Manhattan Project, which developed the atomic bomb during World War II. He is widely regarded as one of the most important physicists of the 20th century and is known for his contributions to the development of quantum mechanics and the theory of the atomic nucleus. Oppenheimer was also a prominent figure in the post-war nuclear weapons debate, advocating for international control and regulation of nuclear weapons."
            },
            "finish_reason":"stop"
        }
    ],
    "usage":{
        "prompt_tokens":9,
        "completion_tokens":12,
        "total_tokens":21
    }
}
```

### `/v1/files` endpoint for uploading text and markdown files

The following command upload a text file [paris.txt](https://huggingface.co/datasets/gaianet/paris/raw/main/paris.txt) to the API server via the `/v1/files` endpoint:

```bash
curl -X POST http://127.0.0.1:8080/v1/files -F "file=@paris.txt"
```

If the command is successful, you should see the similar output as below in your terminal:

```bash
{
    "id": "file_4bc24593-2a57-4646-af16-028855e7802e",
    "bytes": 2161,
    "created_at": 1711611801,
    "filename": "paris.txt",
    "object": "file",
    "purpose": "assistants"
}
```

The `id` and `filename` fields are important for the next step, for example, to segment the uploaded file to chunks for computing embeddings.

If you'd like to build a RAG chatbot, it's strongly recommended to visit [LlamaEdge-RAG API Server](https://github.com/LlamaEdge/rag-api-server).

### `/v1/chunks` endpoint for segmenting files to chunks

To segment the uploaded file to chunks for computing embeddings, use the `/v1/chunks` API. The following command sends the uploaded file ID and filename to the API server and gets the chunks:

```bash
curl -X POST http://localhost:8080/v1/chunks \
    -H 'accept:application/json' \
    -H 'Content-Type: application/json' \
    -d '{"id":"file_4bc24593-2a57-4646-af16-028855e7802e", "filename":"paris.txt"}'
```

The following is an example return with the generated chunks:

```json
{
    "id": "file_4bc24593-2a57-4646-af16-028855e7802e",
    "filename": "paris.txt",
    "chunks": [
        "Paris, city and capital of France, ..., for Paris has retained its importance as a centre for education and intellectual pursuits.",
        "Paris’s site at a crossroads ..., drawing to itself much of the talent and vitality of the provinces."
    ]
}
```

If you'd like to build a RAG chatbot, it's strongly recommended to visit [LlamaEdge-RAG API Server](https://github.com/LlamaEdge/rag-api-server).

### `/v1/embeddings` endpoint for computing embeddings

To compute embeddings for user query or file chunks, use the `/v1/embeddings` API. The following command sends a query to the API server and gets the embeddings as return:

```bash
curl -X POST http://localhost:8080/v1/embeddings \
    -H 'accept:application/json' \
    -H 'Content-Type: application/json' \
    -d '{"model": "e5-mistral-7b-instruct-Q5_K_M", "input":["Paris, city and capital of France, ..., for Paris has retained its importance as a centre for education and intellectual pursuits.", "Paris’s site at a crossroads ..., drawing to itself much of the talent and vitality of the provinces."]}'
```

The embeddings returned are like below:

```json
{
    "object": "list",
    "data": [
        {
            "index": 0,
            "object": "embedding",
            "embedding": [
                0.1428378969,
                -0.0447309874,
                0.007660218049,
                ...
                -0.0128974719,
                -0.03543198109,
                0.03974733502,
                0.00946635101,
                -0.01531364303
            ]
        },
        {
            "index": 1,
            "object": "embedding",
            "embedding": [
                0.0697753951,
                -0.0001159032545,
                0.02073983476,
                ...
                0.03565846011,
                -0.04550019652,
                0.02691745944,
                0.02498772368,
                -0.003226313973
            ]
        }
    ],
    "model": "e5-mistral-7b-instruct-Q5_K_M",
    "usage": {
        "prompt_tokens": 491,
        "completion_tokens": 0,
        "total_tokens": 491
    }
}
```

If you'd like to build a RAG chatbot, it's strongly recommended to visit [LlamaEdge-RAG API Server](https://github.com/LlamaEdge/rag-api-server).

<!-- - Completions

    To obtain the completion for a single prompt, use the `/v1/completions` API. The following command sends a prompt to the API server and gets the completion:

    ```bash
    curl -X POST http://50.112.58.64:8080/v1/completions \
        -H 'accept:application/json' \
        -H 'Content-Type: application/json' \
        -d '{"prompt":["Long long ago, "], "model":"tinyllama"}'
    ```

    The response looks like below:

    ```json
    {
        "id": "b68bfc92-8b23-4435-bbe1-492e11c973a3",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "logprobs": null,
                "text": "in a galaxy far, far away, a group of Jedi Knights lived and trained to defend the peace and security of the galaxy. They were the last hope for peace and justice in a world where the dark side of the force was rife with corruption and injustice. The Knights were a select few, and their training and abilities were the envy of the galaxy. They were the chosen ones. They were the ones who would bring peace and justice to the galaxy. ..."
            }
        ],
        "created": 1702046592,
        "model": "tinyllama",
        "object": "text_completion",
        "usage": {
            "prompt_tokens": 3,
            "completion_tokens": 804,
            "total_tokens": 807
        }
    }
    ``` -->

## Add a web UI

We provide a front-end Web UI for you to easily interact with the API. You can download and extract it by running:

```bash
curl -LO https://github.com/LlamaEdge/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
```

After that, you can use the same command line to create the API server

```bash
wasmedge --dir .:. --nn-preload default:GGML:AUTO:llama-2-7b-chat.Q5_K_M.gguf llama-api-server.wasm -p llama-2-chat
```

Then, you will be asked to open `http://127.0.0.1:8080` from your browser.

## CLI options for the API server

The `-h` or `--help` option can list the available options of the `llama-api-server` wasm app:

```console
$ wasmedge llama-api-server.wasm -h

Usage: llama-api-server.wasm [OPTIONS] --prompt-template <PROMPT_TEMPLATE>

Options:
  -m, --model-name <MODEL_NAME>
          Model name [default: default]
  -a, --model-alias <MODEL_ALIAS>
          Model alias [default: default]
  -c, --ctx-size <CTX_SIZE>
          Context size [default: 512]
  -p, --prompt-template <PROMPT_TEMPLATE>
          Sets the prompt template [possible values: llama-2-chat, mistral-instruct, mistrallite, openchat, codellama-instruct, codellama-super-instruct, human-assistant, vicuna-1.0-chat, vicuna-1.1-chat, vicuna-llava, chatml, baichuan-2, wizard-coder, zephyr, stablelm-zephyr, intel-neural, deepseek-chat, deepseek-coder, solar-instruct, phi-2-chat, phi-2-instruct, gemma-instruct]
  -r, --reverse-prompt <REVERSE_PROMPT>
          Halt generation at PROMPT, return control
  -n, --n-predict <N_PREDICT>
          Number of tokens to predict [default: 1024]
  -g, --n-gpu-layers <N_GPU_LAYERS>
          Number of layers to run on the GPU [default: 100]
  -b, --batch-size <BATCH_SIZE>
          Batch size for prompt processing [default: 512]
      --temp <TEMP>
          Temperature for sampling [default: 1.0]
      --top-p <TOP_P>
          An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. 1.0 = disabled [default: 1.0]
      --repeat-penalty <REPEAT_PENALTY>
          Penalize repeat sequence of tokens [default: 1.1]
      --presence-penalty <PRESENCE_PENALTY>
          Repeat alpha presence penalty. 0.0 = disabled [default: 0.0]
      --frequency-penalty <FREQUENCY_PENALTY>
          Repeat alpha frequency penalty. 0.0 = disabled [default: 0.0]
      --llava-mmproj <LLAVA_MMPROJ>
          Path to the multimodal projector file
      --log-prompts
          Print prompt strings to stdout
      --log-stat
          Print statistics to stdout
      --log-all
          Print all log information to stdout
      --socket-addr <SOCKET_ADDR>
          Socket address of LlamaEdge API Server instance [default: 0.0.0.0:8080]
      --web-ui <WEB_UI>
          Root path for the Web UI files [default: chatbot-ui]
  -h, --help
          Print help
  -V, --version
          Print version
```

Please guarantee that the port is not occupied by other processes. If the port specified is available on your machine and the command is successful, you should see the following output in the terminal:

```console
[INFO] LlamaEdge HTTP listening on 8080
```

If the Web UI is ready, you can navigate to `http://127.0.0.1:8080` to open the chatbot, it will interact with the API of your server.
