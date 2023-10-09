# LLM API SERVER

> Note: Before reading the following content, please make sure that you are working in an environment of Ubuntu 20.04/22.04 and have installed the following necessary dependencies:
>
> * Rust-stable (>= 1.69.0)
> * Add `wasm32-wasi` target to Rust toolchain by running `rustup target add wasm32-wasi` in the terminal
> * WasmEdge 0.13.4 ([Installation](https://wasmedge.org/docs/start/install#generic-linux-and-macos))
> * WasmEdge TLS plugin ([Installation](https://wasmedge.org/docs/start/install#tls-plug-in))

## Build and run

Now let's build and run the API server.

* Build the `llm-api-server` wasm app:

    ```bash
    git clone https://github.com/second-state/llama-utils.git

    cd api-server

    // build the wasm app
    cargo build -p llm-api-server --target wasm32-wasi --release
    ```

    If the commands are successful, you should find the wasm app in `target/wasm32-wasi/release/llm-api-server.wasm`.

* Download the Llama model of gguf format

  When we run the API server in the next step, we will use the `llama-2-7b-chat.Q5_K_M.gguf` model. You can download it by running the following command:

  ```bash
  curl -LO https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q5_K_M.gguf
  ```

* Run the API server:

  ```bash
  wasmedge --dir .:. --env SOCKET_ADDRESS=0.0.0.0:8080 --nn-preload default:GGML:CPU:llama-2-7b-chat.Q5_K_M.gguf target/wasm32-wasi/release/llm-api-server.wasm default
  ```

  * The `--env SOCKET_ADDRESS=0.0.0.0:8080` option specifies the socket address of the API server. The default socket address `0.0.0.0:8080` is used if this option is not specified.

  * The `--nn-preload default:GGML:CPU:llama-2-7b-chat.Q5_K_M.gguf` option specifies the Llama model to be used by the API server. The pattern of the argument is `<name>:<encoding>:<target>:<model path>`. Here, the model used is `llama-2-7b-chat.Q5_K_M.gguf`; and we give it an alias `default` as its name in the runtime environment.

  Please guarantee that the port is not occupied by other processes. If the port specified is available on your machine and the command is successful, you should see the following output in the terminal:

  ```bash
  Listening on http://0.0.0.0:8080
  ```

## Test the API server

`llm-api-server` provides a POST API `/echo` for testing. You can use `curl` to test it:

```bash
curl -X POST http://localhost:8080/echo
```

If the command is successful, you should see the following output in the terminal:

```bash
echo test
```

## Multi-turn Conversations

(Todo: add steps to run `chatbot-ui`)
