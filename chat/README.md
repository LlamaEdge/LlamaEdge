# Chat on the CLI

[See it in action!](https://x.com/juntao/status/1705588244602114303)

## Dependencies

Install the latest WasmEdge with plugins:

<details> <summary> For macOS (apple silicon) </summary>

```bash
# install WasmEdge-0.13.4 with wasi-nn-ggml plugin
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | bash -s -- --plugin wasi_nn-ggml

# Assuming you use zsh (the default shell on macOS), run the following command to activate the environment
source $HOME/.zshenv
```

</details>

<details> <summary> For Ubuntu (>= 20.04) </summary>

```bash
# install libopenblas-dev
apt update && apt install -y libopenblas-dev

# install WasmEdge-0.13.4 with wasi-nn-ggml plugin
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | bash -s -- --plugin wasi_nn-ggml

# Assuming you use bash (the default shell on Ubuntu), run the following command to activate the environment
source $HOME/.bashrc
```

</details>

<details> <summary> For General Linux </summary>

```bash
# install WasmEdge-0.13.4 with wasi-nn-ggml plugin
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | bash -s -- --plugin wasi_nn-ggml

# Assuming you use bash (the default shell on Ubuntu), run the following command to activate the environment
source $HOME/.bashrc
```

</details>

## Get `chat` wasm app

Download the `chat.wasm`:

```bash

```


## Get Model

<details> <summary> Choose the model you'd like to run: </summary>

- [x] Llama2-Chat

  ```bash
  # llama-2-7b
  curl -LO https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q5_K_M.gguf

  # llama-2-13b
  curl -LO https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q5_K_M.gguf
  ```

- [x] CodeLlama-Instruct

  ```bash
  # codellama-13b-instruct
  curl -LO curl -LO https://huggingface.co/TheBloke/CodeLlama-13B-Instruct-GGUF/resolve/main/codellama-13b-instruct.Q4_0.gguf
  ```

- [x] BELLE-Llama2-Chat

  ```bash
  # BELLE-Llama2-13B-Chat-0.4M
  curl -LO https://huggingface.co/second-state/BELLE-Llama2-13B-Chat-GGUF/resolve/main/BELLE-Llama2-13B-Chat-0.4M-ggml-model-q4_0.gguf
  ```

- [x] Mistral-Instruct-v0.1

  ```bash
  # mistral-7b-instruct-v0.1
  curl -LO https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q5_K_M.gguf
  ```

- [x] Wizard-Vicuna

  ```bash
  # wizard-vicuna-13b
  curl -LO https://huggingface.co/second-state/wizard-vicuna-13B-GGUF/resolve/main/wizard-vicuna-13b-ggml-model-q8_0.gguf
  ```

- [x] rpguild-chatml

  ```bash
  # rpguild-chatml-13b
  curl -LO https://huggingface.co/second-state/rpguild-chatml-13B-GGUF/resolve/main/rpguild-chatml-13b.Q5_K_M.gguf
  ```

- [ ] Baichuan2-Chat (Coming soon)

- [ ] Baichuan-Chat (Coming soon)

- [ ] CodeShell-Chat (Coming soon)

</details>

## Execute

Execute the WASM with the `wasmedge` using the named model feature to preload large model. Here we use the `Llama-2-7B-Chat` model as an example:

```bash
# download model
curl -LO https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q5_K_M.gguf

# run chat.wasm with the model
wasmedge --dir .:. \
  --nn-preload default:GGML:CPU:llama-2-7b-chat.Q5_K_M.gguf \
  chat.wasm --model-alias default --prompt-template llama-2-chat
```

After executing the command, you may need to wait a moment for the input prompt to appear.
You can enter your question once you see the `[USER]:` prompt:

```console
[USER]:
What's the capital of France?
[ASSISTANT]:
The capital of France is Paris.
[USER]:
what about Norway?
[ASSISTANT]:
The capital of Norway is Oslo.
[USER]:
I have two apples, each costing 5 dollars. What is the total cost of these apples?
[ASSISTANT]:
The total cost of the two apples is 10 dollars.
[USER]:
What if I have 3 apples?
[ASSISTANT]:
If you have 3 apples, each costing 5 dollars, the total cost of the apples is 15 dollars.
```

## Parameters

Currently, we support the following parameters:

- `LLAMA_LOG`: Set it to a non-empty value to enable logging.
- `LLAMA_N_CTX`: Set the context size, the same as the `--ctx-size` parameter in llama.cpp (default: 512).
- `LLAMA_N_PREDICT`: Set the number of tokens to predict, the same as the `--n-predict` parameter in llama.cpp (default: 512).

These parameters can be set by adding the following environment variables before the `wasmedge` command:

```bash
LLAMA_LOG=1 LLAMA_N_CTX=2048 LLAMA_N_PREDICT=512 \
wasmedge --dir .:. \
  --nn-preload default:GGML:CPU:llama-2-7b-chat.Q5_K_M.gguf \
  chat.wasm --model-alias default --ctx-size 2048
```

## Optional: Build `chat.wasm` yourself

Run the following command:

```bash
cargo build --target wasm32-wasi --release
```

The `chat.wasm` will be generated in the `target/wasm32-wasi/release` folder.
