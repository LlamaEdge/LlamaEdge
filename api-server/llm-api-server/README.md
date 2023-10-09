# LLM WEB API SERVER

> Note: Before reading the following content, please make sure that you are working in an environment of Ubuntu 20.04/22.04 and have installed the following necessary dependencies:
>
> * Rust-stable (>= 1.69.0)
> * Add `wasm32-wasi` target to Rust toolchain by running `rustup target add wasm32-wasi` in the terminal
> * WasmEdge 0.13.4 ([Installation](https://wasmedge.org/docs/start/install#generic-linux-and-macos))
> * WasmEdge TLS plugin ([Installation](https://wasmedge.org/docs/start/install#tls-plug-in))

## How to build and run?

Before building the wasm app, you may set the socket address of the web API server in `config.yml`. The default socket address is:

```yaml
socket_address:
  ip_address: "0.0.0.0"
  port: "8080"
```

Please guarantee that the port is not occupied by other processes. Now let's build and run the web API server:

* First, build the `llm-web-api-server` wasm app:

    ```bash
    git clone https://github.com/apepkuss/llm-web-api-server.git

    cd llm-web-api-server

    // build the wasm app
    cargo build --target wasm32-wasi --release
    ```

    If the commands are successful, you should find the wasm app in `target/wasm32-wasi/release/llm-web-api-server.wasm`.

* Second, to maximize the performance of the wasm app, use `WasmEdge AOT Compiler` to compile the wasm app to native code:

    ```bash
    wasmedge compile target/wasm32-wasi/release/llm-web-api-server.wasm llm-web-api-server.so
    ```

    If the command is successful, you should find `llm-web-api-server.so` in the root directory.

* Finally, run the wasm app, namely starting the web API server:

    ```bash
    wasmedge run --dir .:. llm-web-api-server.so
    ```

    if the `8080` port is available on your machine and the command is successful, you should see the following output in the terminal:

    ```bash
    Listening on http://0.0.0.0:8080
    ```

    Note that the command above is only used for testing. In production, you need to specify which LLM model will be used. For example, if you want to use the `llm-model` in the `llm-models` directory, you can run the following command:

## Test the web API server

`llm-web-api-server` provides a POST API `/echo` for testing. You can use `curl` to test it:

```bash
curl -X POST http://localhost:8080/echo
```

If the command is successful, you should see the following output in the terminal:

```bash
echo test
```

## Multi-turn Conversations

* Download the Llama model of gguf format

  ```bash
  curl -LO https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q5_K_M.gguf
  ```

* Start the web API server

  ```bash
  wasmedge --dir .:. --nn-preload default:GGML:CPU:llama-2-7b-chat.Q5_K_M.gguf llm-web-api-server.so default
  ```

* Download `wasmedge-web-api-client` and run the client

  ```bash
  git clone https://github.com/second-state/wasmedge-web-api-client.git
  ```

  Note that you need to update the `URL_CHAT_COMPLETIONS` static variable in the `main.rs` file to the address of the web API server.

  ```bash
  cd wasmedge-web-api-client

  // build and run the client
  cargo run
  ```
  
  You will see the following output in the terminal if the command runs successfully:
  
  ```bash
  Enter some text (or press Ctrl + Q to exit):
  [Question]:
  ```

  Now you can enter your question and wait for the answe. For example:

  ```bash
  Enter some text (or press Ctrl + Q to exit):
  [Question]:
  what is the capital of France?
  [answer] The capital of France is Paris.
  [Question]:
  what about Norway?
  [answer] The capital of Norway is Oslo.
  [Question]:
  I have two apples, each costing 5 dollars. What is the total   cost of these apples?
  [answer] The total cost of the two apples is 10 dollars.
  [Question]:
  What if I have 3 apples?
  [answer] If you have 3 apples, each costing 5 dollars, the   total cost of the apples is 15 dollars.
  ```
