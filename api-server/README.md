# LLAMA API SERVER

> Note: Before reading the following content, please make sure that you are working in an environment of Ubuntu 20.04/22.04 and have installed the following necessary dependencies:
>
> * Rust-stable (>= 1.69.0)
> * Add `wasm32-wasi` target to Rust toolchain by running `rustup target add wasm32-wasi` in the terminal
> * WasmEdge 0.13.4 ([Installation](https://wasmedge.org/docs/start/install#generic-linux-and-macos))

## Build and run

Now let's build and run the API server.

* Build the `llama-api-server` wasm app:

    ```bash
    git clone https://github.com/second-state/llama-utils.git

    cd api-server

    // build the wasm app
    cargo build -p llama-api-server --target wasm32-wasi --release
    ```

    If the commands are successful, you should find the wasm app in `target/wasm32-wasi/release/llama-api-server.wasm`.

* Download the Llama model of gguf format

  When we run the API server in the next step, we will use the `llama-2-7b-chat.Q5_K_M.gguf` model. You can download it by running the following command:

  ```bash
  curl -LO https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q5_K_M.gguf
  ```

* Run the API server:

  ```bash
  wasmedge --dir .:. --env SOCKET_ADDRESS=0.0.0.0:8080 --nn-preload default:GGML:CPU:llama-2-7b-chat.Q5_K_M.gguf target/wasm32-wasi/release/llama-api-server.wasm default llama-2-chat
  ```

  * The `--env SOCKET_ADDRESS=<ip-address>:<port>` option specifies the socket address of the API server. The default socket address `0.0.0.0:8080` is used if this option is not specified.

  * The `--nn-preload default:GGML:CPU:llama-2-7b-chat.Q5_K_M.gguf` option specifies the Llama model to be used by the API server. The pattern of the argument is `<name>:<encoding>:<target>:<model path>`. Here, the model used is `llama-2-7b-chat.Q5_K_M.gguf`; and we give it an alias `default` as its name in the runtime environment.
  * The `default` value in the second-to-last argument of the command specifies the alias of the model to be used in the runtime. This name should be the same as the name specified in the `--nn-preload` option.
  * The `llama-2-chat` value in the last argument of the command specifies the name of the prompt template to be used. Currently,the available prompt templates are `llama-2-chat`, `codellama-instruct`, and `mistral-instruct-v0.1`.

  Please guarantee that the port is not occupied by other processes. If the port specified is available on your machine and the command is successful, you should see the following output in the terminal:

  ```bash
  Listening on http://0.0.0.0:8080
  ```

## Test the API server

`llama-api-server` provides a POST API `/v1/models` to list currently available models. You can use `curl` to test it:

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
curl -X POST http://localhost:8000/v1/chat/completions -H 'accept:application/json' -H 'Content-Type: application/json' -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "Who is Robert Oppenheimer?"}], "model":"llama-2-chat"}'
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

## Multi-turn Conversations

(Todo: add steps to run `chatbot-ui`)
