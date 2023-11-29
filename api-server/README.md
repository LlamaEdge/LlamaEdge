#  Ceate an OpenAI compatible API server for your LLM

An OpenAI-compatible web API allows the model to work with a large ecosystem of LLM tools and agent frameworks such as flows.network, LangChain and LlamaIndex.


**ToC**

* [Dependencies](#Dependencies)
* [Get the llama-api-server.wasm app](#get-the-llama-api-serverwasm-app)
* [Get the model](#get-the-model)
* [Run the API server via curl](#run-the-api-server-via-curl)
  * [Test the API server via terminal](#test-the-api-server-via-terminal)
* [Add a Web UI](#add-a-web-ui)
* [CLI options](#cli-options-for-the-api-server)


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

```
curl -LO https://github.com/second-state/llama-utils/raw/main/api-server/llama-api-server.wasm
```

## Get the model 

Cilck [here](../llama-utils/chat/README.md#get-the-model) to learn the mode we support.


## Run the API server via curl

Run the API server with the following command:

```bash
wasmedge --dir .:. --nn-preload default:GGML:AUTO:llama-2-7b-chat.Q5_K_M.gguf llama-api-server.wasm -p llama-2-chat
```

The command above starts the API server on the default socket address. Besides, there are also some other options specified in the command:

* The `--dir .:.` option specifies the current directory as the root directory of the WASI file system.

* The `--nn-preload default:GGML:AUTO:llama-2-7b-chat.Q5_K_M.gguf` option specifies the Llama model to be used by the API server. The pattern of the argument is `<name>:<encoding>:<target>:<model path>`. Here, the model used is `llama-2-7b-chat.Q5_K_M.gguf`; and we give it an alias `default` as its name in the runtime environment. You can change the model name here if you're not using llama2-7b-chat
* The `-p llama-2-chat` is the prompt template for the model.

### Test the API server via terminal

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
            Sets the prompt context size [default: 4096]
    -n, --n-predict <N_PRDICT>
            Number of tokens to predict [default: 1024]
    -g, --n-gpu-layers <N_GPU_LAYERS>
            Number of layers to run on the GPU [default: 100]
    -b, --batch-size <BATCH_SIZE>
            Batch size for prompt processing [default: 4096]
    -r, --reverse-prompt <REVERSE_PROMPT>
            Halt generation at PROMPT, return control.
    -p, --prompt-template <TEMPLATE>
            Sets the prompt template. [default: llama-2-chat] [possible values: llama-2-chat, codellama-instruct, mistral-instruct-v0.1, mistrallite, openchat, belle-llama-2-chat, vicuna-chat, chatml, baichuan-2, wizard-coder, zephyr, intel-neural]
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
  ```




  Please guarantee that the port is not occupied by other processes. If the port specified is available on your machine and the command is successful, you should see the following output in the terminal:

  ```console
  Listening on http://0.0.0.0:8080
  ```

  If the Web UI is ready, you can navigate to `http://127.0.0.1:8080` to open the chatbot, it will interact with the API of your server. 








