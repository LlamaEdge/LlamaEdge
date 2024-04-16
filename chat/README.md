# Run the LLM via CLI

[See it in action!](https://x.com/juntao/status/1705588244602114303)

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [Run the LLM via CLI](#run-the-llm-via-cli)
  - [Dependencies](#dependencies)
  - [Get `llama-chat` wasm app](#get-llama-chat-wasm-app)
  - [Get Model](#get-model)
  - [Execute](#execute)
  - [CLI options](#cli-options)
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

## Get `llama-chat` wasm app

Download the `llama-chat.wasm`:

```bash
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-chat.wasm
```

## Get Model

Click [here](../models.md) to see the download link and commands to run the model.

## Execute

Execute the WASM with the `wasmedge` using the named model feature to preload large model. Here we use the `Llama-2-7B-Chat` model as an example:

```console
# download model
curl -LO https://huggingface.co/second-state/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q5_K_M.gguf

# run the `llama-chat` wasm app with the model
wasmedge --dir .:. --nn-preload default:GGML:AUTO:llama-2-7b-chat.Q5_K_M.gguf llama-chat.wasm --prompt-template llama-2-chat
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

## CLI options

The options for `llama-chat` wasm app are:

```console
~/workspace/llama-utils/chat$ wasmedge llama-chat.wasm -h
Usage: llama-chat.wasm [OPTIONS]

Options:
  -a, --model-alias <ALIAS>
          Model alias [default: default]
  -c, --ctx-size <CTX_SIZE>
          Size of the prompt context [default: 512]
  -n, --n-predict <N_PRDICT>
          Number of tokens to predict [default: 1024]
  -g, --n-gpu-layers <N_GPU_LAYERS>
          Number of layers to run on the GPU [default: 100]
  -b, --batch-size <BATCH_SIZE>
          Batch size for prompt processing [default: 512]
      --temp <TEMP>
          Temperature for sampling [default: 0.8]
      --top-p <TOP_P>
          An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. 1.0 = disabled. [default: 0.9]
      --repeat-penalty <REPEAT_PENALTY>
          Penalize repeat sequence of tokens [default: 1.1]
      --presence-penalty <PRESENCE_PENALTY>
          Repeat alpha presence penalty. 0.0 = disabled. [default: 0.0]
      -frequency-penalty <FREQUENCY_PENALTY>
          Repeat alpha frequency penalty. 0.0 = disabled [default: 0.0]
  -r, --reverse-prompt <REVERSE_PROMPT>
          Halt generation at PROMPT, return control.
  -s, --system-prompt <SYSTEM_PROMPT>
          System prompt message string [default: "[Default system message for the prompt template]"]
  -p, --prompt-template <TEMPLATE>
          Prompt template. [default: llama-2-chat] [possible values: llama-2-chat, codellama-instruct, codellama-super-instruct, mistral-instruct, mistrallite, openchat, human-assistant, vicuna-1.0-chat, vicuna-1.1-chat, chatml, baichuan-2, wizard-coder, wizard-2-chat, zephyr, stablelm-zephyr, intel-neural, deepseek-chat, deepseek-coder, solar-instruct, phi-2-instruct, gemma-instruct]
      --log-prompts
          Print prompt strings to stdout
      --log-stat
          Print statistics to stdout
      --log-all
          Print all log information to stdout
  -h, --help
          Print help
  -V, --version
          Print version

Example: the command to run `llama-2-7B` model,
  wasmedge --dir .:. --nn-preload default:GGML:AUTO:llama-2-7b-chat.Q5_K_M.gguf llama-chat.wasm -p llama-2-chat
```

## Optional: Build the `llama-chat` wasm app yourself

Run the following command:

```console
cargo build --target wasm32-wasi --release
```

The `llama-chat.wasm` will be generated in the `target/wasm32-wasi/release` folder.
