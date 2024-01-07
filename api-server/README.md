# Ceate an OpenAI compatible API server for your LLM

An OpenAI-compatible web API allows the model to work with a large ecosystem of LLM tools and agent frameworks such as flows.network, LangChain and LlamaIndex.

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [Ceate an OpenAI compatible API server for your LLM](#ceate-an-openai-compatible-api-server-for-your-llm)
  - [Dependencies](#dependencies)
  - [Get the llama-api-server.wasm app](#get-the-llama-api-serverwasm-app)
  - [Get model](#get-model)
  - [Run the API server via curl](#run-the-api-server-via-curl)
    - [Test the API server via terminal](#test-the-api-server-via-terminal)
  - [Add a web UI](#add-a-web-ui)
  - [CLI options for the API server](#cli-options-for-the-api-server)
  - [Optional: Build the `llama-chat` wasm app yourself](#optional-build-the-llama-chat-wasm-app-yourself)

<!-- /code_chunk_output -->

## Dependencies

Install the latest WasmEdge with plugins:

<details> <summary> For macOS (apple silicon) </summary>

```console
# install WasmEdge-0.13.4 with wasi-nn-ggml plugin
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | bash -s -- --plugin wasi_nn-ggml

# Assuming you use zsh (the default shell on macOS), run the following command to activate the environment
source $HOME/.zshenv
```

</details>

<details> <summary> For Ubuntu (>= 20.04) </summary>

```console
# install libopenblas-dev
apt update && apt install -y libopenblas-dev

# install WasmEdge-0.13.4 with wasi-nn-ggml plugin
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | bash -s -- --plugin wasi_nn-ggml

# Assuming you use bash (the default shell on Ubuntu), run the following command to activate the environment
source $HOME/.bashrc
```

</details>

<details> <summary> For General Linux </summary>

```console
# install WasmEdge-0.13.4 with wasi-nn-ggml plugin
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | bash -s -- --plugin wasi_nn-ggml

# Assuming you use bash (the default shell on Ubuntu), run the following command to activate the environment
source $HOME/.bashrc
```

</details>

## Get the llama-api-server.wasm app

Download the api server app

```console
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-api-server.wasm
```

## Get model

Cilck [here](../models.md) to see the model download link and commadns to run the API server and test the API server.

## Run the API server via curl

Run the API server with the following command:

```bash
wasmedge --dir .:. --nn-preload default:GGML:AUTO:llama-2-7b-chat.Q5_K_M.gguf llama-api-server.wasm -p llama-2-chat
```

The command above starts the API server on the default socket address. Besides, there are also some other options specified in the command:

- The `--dir .:.` option specifies the current directory as the root directory of the WASI file system.

- The `--nn-preload default:GGML:AUTO:llama-2-7b-chat.Q5_K_M.gguf` option specifies the Llama model to be used by the API server. The pattern of the argument is `<name>:<encoding>:<target>:<model path>`. Here, the model used is `llama-2-7b-chat.Q5_K_M.gguf`; and we give it an alias `default` as its name in the runtime environment. You can change the model name here if you're not using llama2-7b-chat
- The `-p llama-2-chat` is the prompt template for the model.

### Test the API server via terminal

- List models

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
                "id":"llama-2-chat",
                "created":1697084821,
                "object":"model",
                "owned_by":"Not specified"
            }
        ]
    }
    ```

- Chat completions

    Ask a question using OpenAI's JSON message format.

    ```bash
    curl -X POST http://localhost:8080/v1/chat/completions -H 'accept:application/json' -H 'Content-Type: application/json' -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "Who is Robert Oppenheimer?"}], "model":"llama-2-chat"}'
    ```

    Here is the response.

    ```bash
    {
        "id":"",
        "object":"chat.completion",
        "created":1697092593,
        "model":"llama-2-chat",
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

- Completions

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
    ```

## Add a web UI

We provide a front-end Web UI for you to easily interact with the API. You can download and extract it by running:

```bash
curl -LO https://github.com/second-state/chatbot-ui/releases/download/v0.1.0/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
```

After that, you can use the same command line to create the API server

```bash
wasmedge --dir .:. --nn-preload default:GGML:AUTO:llama-2-7b-chat.Q5_K_M.gguf llama-api-server.wasm -p llama-2-chat
```

Then, you will be asked to open http://127.0.0.1:8080 from your browser.

## CLI options for the API server

The `-h` or `--help` option can list the available options of the `llama-api-server` wasm app:

  ```console
  ~/llama-utils/api-server$ wasmedge llama-api-server.wasm -h

  Usage: llama-api-server.wasm [OPTIONS]

  Options:
    -s, --socket-addr <IP:PORT>
            Sets the socket address [default: 0.0.0.0:8080]
    -m, --model-name <MODEL-NAME>
            Sets the model name [default: default]
    -a, --model-alias <MODEL-ALIAS>
            Sets the alias name of the model in WasmEdge runtime [default: default]
    -c, --ctx-size <CTX_SIZE>
            Sets the prompt context size [default: 512]
    -n, --n-predict <N_PRDICT>
            Number of tokens to predict [default: 1024]
    -g, --n-gpu-layers <N_GPU_LAYERS>
            Number of layers to run on the GPU [default: 100]
    -b, --batch-size <BATCH_SIZE>
            Batch size for prompt processing [default: 512]
        --temp <TEMP>
            Temperature for sampling [default: 0.8]
        --repeat-penalty <REPEAT_PENALTY>
            Penalize repeat sequence of tokens [default: 1.1]
    -r, --reverse-prompt <REVERSE_PROMPT>
            Halt generation at PROMPT, return control.
    -p, --prompt-template <TEMPLATE>
            Sets the prompt template. [default: llama-2-chat] [possible values: llama-2-chat, codellama-instruct, mistral-instruct-v0.1, mistral-instruct, mistrallite, openchat, belle-llama-2-chat, vicuna-chat, vicuna-1.1-chat, chatml, baichuan-2, wizard-coder, zephyr, intel-neural, deepseek-chat, deepseek-coder, solar-instruct]
        --stream
          Enable streaming mode
        --log-prompts
            Print prompt strings to stdout
        --log-stat
            Print statistics to stdout
        --log-all
            Print all log information to stdout
        --web-ui <WEB_UI>
            Root path for the Web UI files [default: chatbot-ui]
    -h, --help
            Print help
    -V, --version
          Print version
  ```

  Please guarantee that the port is not occupied by other processes. If the port specified is available on your machine and the command is successful, you should see the following output in the terminal:

  ```console
  Listening on http://0.0.0.0:8080
  ```

  If the Web UI is ready, you can navigate to `http://127.0.0.1:8080` to open the chatbot, it will interact with the API of your server.

## Optional: Build the `llama-chat` wasm app yourself

Run the following command:

```console
cargo build --target wasm32-wasi --release
```

The `llama-api-server.wasm` will be generated in the `target/wasm32-wasi/release` folder.
