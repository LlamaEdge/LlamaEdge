# Chat on the CLI

[See it in action!](https://x.com/juntao/status/1705588244602114303)

## Dependencies

Install the latest wasmedge with plugins:

```bash
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | bash -s -- --plugins wasi_nn-ggml
```

## Get the compiled wasm binary program

Download the wasm file:

```bash
curl -LO https://github.com/second-state/llama-utils/raw/main/chat/llama-chat.wasm
```

## Get Model

Download llama model:

```bash
curl -LO https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q5_K_M.gguf
```

## Execute

Execute the WASM with the `wasmedge` using the named model feature to preload large model:

```bash
LLAMA_LOG=1 LLAMA_N_CTX=1024 LLAMA_N_PREDICT=512 \
wasmedge --dir .:. \
  --nn-preload default:GGML:CPU:llama-2-7b-chat.Q5_K_M.gguf \
  llama-chat.wasm default
```

After executing the command, you may need to wait a moment for the input prompt to appear.
You can enter your question once you see the `Question:` prompt:

```console
Question:
What's the capital of the United States?
Answer:
The capital of the United States is Washington, D.C. (District of Columbia).
Question:
What about France?
Answer:
The capital of France is Paris.
Question:
I have two apples, each costing 5 dollars. What is the total cost of these apples?
Answer:
The total cost of the two apples is $10.
Question:
What if I have 3 apples?
Answer:
The total cost of 3 apples would be 15 dollars. Each apple costs 5 dollars, so 3 apples would cost 3 x 5 = 15 dollars.
```

## Optional: Build the wasm file yourself

Compile the application to WebAssembly:

```bash
cargo build --target wasm32-wasi --release
```

The output wasm file will be at `target/wasm32-wasi/release/`.

