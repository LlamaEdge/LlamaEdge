# Chat on the CLI

[See it in action!](https://x.com/juntao/status/1705588244602114303)

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

## Get `llama-chat` wasm app

Download the `llama-chat.wasm`:

```bash
curl -LO https://github.com/second-state/llama-utils/raw/main/chat/llama-chat.wasm
```

The options for `llama-chat` wasm app are:

```console
~/llama-utils/chat$ wasmedge llama-chat.wasm -h
Usage: llama-chat.wasm [OPTIONS] --prompt-template <TEMPLATE>

Options:
  -m, --model-alias <ALIAS>            Sets the model alias [default: default]
  -c, --ctx-size <CTX_SIZE>            Sets the prompt context size [default: 2048]
  -s, --system-prompt <SYSTEM_PROMPT>  Sets the system prompt message string [default: "[Default system message for the prompt template]"]
  -p, --prompt-template <TEMPLATE>     Sets the prompt template. [default: llama-2-chat] [possible values: llama-2-chat, codellama-instruct, mistral-instruct-v0.1, belle-llama-2-chat, vicuna-chat, chatml]
  -h, --help                           Print help
```

## Get Model

<details> <summary> Choose the model you want to download: </summary>

- [x] Llama2-Chat

  ```console
  # llama-2-7b
  curl -LO https://huggingface.co/second-state/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q5_K_M.gguf

  # llama-2-13b
  curl -LO https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q5_K_M.gguf
  ```

- [x] CodeLlama-Instruct

  ```console
  # codellama-13b-instruct
  curl -LO curl -LO https://huggingface.co/TheBloke/CodeLlama-13B-Instruct-GGUF/resolve/main/codellama-13b-instruct.Q4_0.gguf
  ```

- [x] BELLE-Llama2-Chat

  ```console
  # BELLE-Llama2-13B-Chat-0.4M
  curl -LO https://huggingface.co/second-state/BELLE-Llama2-13B-Chat-GGUF/resolve/main/BELLE-Llama2-13B-Chat-0.4M-ggml-model-q4_0.gguf
  ```

- [x] Mistral-7B-Instruct-v0.1

  ```console
  # mistral-7b-instruct-v0.1
  curl -LO https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q5_K_M.gguf
  ```

- [x] Wizard-Vicuna

  ```console
  # wizard-vicuna-13b
  curl -LO https://huggingface.co/second-state/wizard-vicuna-13B-GGUF/resolve/main/wizard-vicuna-13b-ggml-model-q8_0.gguf
  ```

- [x] CausalLM-14B

  ```console
  # CausalLM-14B
  curl -LO https://huggingface.co/TheBloke/CausalLM-14B-GGUF/resolve/main/causallm_14b.Q5_1.gguf
  ```

- [ ] rpguild-chatml (Coming soon)

  ```console
  # rpguild-chatml-13b
  curl -LO https://huggingface.co/second-state/rpguild-chatml-13B-GGUF/resolve/main/rpguild-chatml-13b.Q5_K_M.gguf
  ```

- [ ] Baichuan2-Chat (Coming soon)

- [ ] Baichuan-Chat (Coming soon)

- [ ] CodeShell-Chat (Coming soon)

</details>

## Execute

Execute the WASM with the `wasmedge` using the named model feature to preload large model. Here we use the `Llama-2-7B-Chat` model as an example:

```console
# download model
curl -LO https://huggingface.co/second-state/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q5_K_M.gguf

# run the `llama-chat` wasm app with the model
wasmedge --dir .:. \
  --nn-preload default:GGML:CPU:llama-2-7b-chat.Q5_K_M.gguf \
  llama-chat.wasm --model-alias default --prompt-template llama-2-chat
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

```console
LLAMA_LOG=1 LLAMA_N_CTX=2048 LLAMA_N_PREDICT=512 \
wasmedge --dir .:. \
  --nn-preload default:GGML:CPU:llama-2-7b-chat.Q5_K_M.gguf \
  llama-chat.wasm --model-alias default --ctx-size 2048
```

## Optional: Build the `llama-chat` wasm app yourself

Run the following command:

```console
cargo build --target wasm32-wasi --release
```

The `llama-chat.wasm` will be generated in the `target/wasm32-wasi/release` folder.
