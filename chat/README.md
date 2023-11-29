# Run the LLM via CLI

[See it in action!](https://x.com/juntao/status/1705588244602114303)

**ToC**

* [Dependencies](#dependencies)
* [Get the inference app](#get-llama-chat-wasm-app )
* [Get model](#get-model)
* [Execute the model](#execute)
* [Optional: Build your own inference app](#optional-build-the-llama-chat-wasm-app-yourself)

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
~/workspace/llama-utils/chat$ wasmedge llama-chat.wasm -h
Usage: llama-chat.wasm [OPTIONS]

Options:
  -a, --model-alias <ALIAS>
          Model alias [default: default]
  -c, --ctx-size <CTX_SIZE>
          Size of the prompt context [default: 4096]
  -n, --n-predict <N_PRDICT>
          Number of tokens to predict [default: 1024]
  -g, --n-gpu-layers <N_GPU_LAYERS>
          Number of layers to run on the GPU [default: 100]
  -b, --batch-size <BATCH_SIZE>
          Batch size for prompt processing [default: 4096]
  -r, --reverse-prompt <REVERSE_PROMPT>
          Halt generation at PROMPT, return control.
  -s, --system-prompt <SYSTEM_PROMPT>
          System prompt message string [default: "[Default system message for the prompt template]"]
  -p, --prompt-template <TEMPLATE>
          Prompt template. [default: llama-2-chat] [possible values: llama-2-chat, codellama-instruct, mistral-instruct-v0.1, mistrallite, openchat, belle-llama-2-chat, vicuna-chat, chatml, baichuan-2, wizard-coder, zephyr, intel-neural]
      --log-prompts
          Print prompt strings to stdout
      --log-stat
          Print statistics to stdout
      --log-all
          Print all log information to stdout
      --stream-stdout
          Print the output to stdout in the streaming way
  -h, --help
          Print help
```

## Get Model

<details> <summary> Choose the model you want to download and run: </summary>

| Models       | Prompt template       |
|--------------------------------------------------------------------------------------------------------------------------------------|-----------------------|
| Llama-2-7B-Chat ([download here](https://huggingface.co/second-state/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q5_K_M.gguf)) | -p llama-2-chat       |
| Llama-2-13B-Chat ([download here](https://huggingface.co/second-state/Llama-2-13B-Chat-GGUF/resolve/main/llama-2-13b-chat.Q5_K_M.gguf))                                                                                                                    | -p llama-2-chat       |
| CodeLlama-13B-Instruct ([download here](https://huggingface.co/second-state/CodeLlama-13B-Instruct-GGUF/resolve/main/codellama-13b-instruct.Q4_0.gguf))      | -p codellama-instruct |
| BELLE-Llama2-13B-Chat ([download here](https://huggingface.co/second-state/BELLE-Llama2-13B-Chat-0.4M-GGUF/resolve/main/BELLE-Llama2-13B-Chat-0.4M-ggml-model-q4_0.gguf)) |     -p belle-llama-2-chat    |
| Mistral-7B-Instruct-v0.1 ([download here](https://huggingface.co/second-state/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q5_K_M.gguf))   |    -p mistral-instruct-v0.1    |
| MistralLite-7B ([download here](https://huggingface.co/second-state/MistralLite-7B-GGUF/resolve/main/mistrallite.Q5_K_M.gguf))  |                 -p mistrallite  |
|  OpenChat-3.5 ([download here](https://huggingface.co/second-state/OpenChat-3.5-GGUF/resolve/main/openchat_3.5.Q5_K_M.gguf))   |                -p openchat -r '<|end_of_turn|>' |
|  Wizard-Vicuna ([download here](https://huggingface.co/second-state/wizard-vicuna-13B-GGUF/resolve/main/wizard-vicuna-13b-ggml-model-q8_0.gguf))   |  -p vicuna-chat |
|  CausalLM-14B ([download here](https://huggingface.co/second-state/CausalLM-14B-GGUF/resolve/main/causallm_14b.Q5_1.gguf))   |  -p chatml |
|  TinyLlama-1.1B-Chat-v0.3 ([download here](https://huggingface.co/second-state/TinyLlama-1.1B-Chat-v0.3-GGUF/resolve/main/tinyllama-1.1b-chat-v0.3.Q5_K_M.gguf))   |  -p chatml |
|  Baichuan2-13B-Chat ([download here](https://huggingface.co/second-state/Baichuan2-13B-Chat-GGUF/resolve/main/Baichuan2-13B-Chat-ggml-model-q4_0.gguf))   | -p baichuan-2 -r '用户:'|
|  Baichuan2-7B-Chat ([download here](https://huggingface.co/second-state/Baichuan2-7B-Chat-GGUF/resolve/main/Baichuan2-13B-Chat-ggml-model-q4_0.gguf))   | -p baichuan-2 -r '用户:'|
|  OpenHermes-2.5-Mistral-7B ([download here](https://huggingface.co/second-state/OpenHermes-2.5-Mistral-7B-GGUF/resolve/main/openhermes-2.5-mistral-7b.Q5_K_M.gguf))   | -p chatml -r '<|im_end|>'|
|  Dolphin-2.2-Yi-34B ([download here](https://huggingface.co/second-state/Dolphin-2.2-Yi-34B-GGUF/resolve/main/dolphin-2.2-yi-34b-ggml-model-q4_0.gguf))   | -p chatml -r '<|im_end|>' -s 'You are a helpful AI assistant'|
|  Dolphin-2.2-Mistral-7B ([download here](https://huggingface.co/second-state/Dolphin-2.2-Mistral-7B-GGUF/resolve/main/dolphin-2.2-mistral-7b-ggml-model-q4_0.gguf))   | -p chatml -r '<|im_end|>'|
|  Dolphin-2.2.1-Mistral-7B ([download here](https://huggingface.co/second-state/Dolphin-2.2.1-Mistral-7B/resolve/main/dolphin-2.2.1-mistral-7b-ggml-model-q4_0.gguf))   | -p chatml -r '<|im_end|>'|
|  Samantha-1.2-Mistral-7B ([download here](https://huggingface.co/second-state/Samantha-1.2-Mistral-7B/resolve/main/samantha-1.2-mistral-7b-ggml-model-q4_0.gguf))   | -p chatml -r '<|im_end|>'|
|  Dolphin-2.1-Mistral-7B ([download here](https://huggingface.co/second-state/Dolphin-2.1-Mistral-7B-GGUF/resolve/main/dolphin-2.1-mistral-7b-ggml-model-q4_0.gguf))   | -p chatml -r '<|im_end|>'|
|  Dolphin-2.0-Mistral-7B ([download here](https://huggingface.co/second-state/Dolphin-2.0-Mistral-7B-GGUF/resolve/main/dolphin-2.1-mistral-7b-ggml-model-q4_0.gguf))   | -p chatml -r '<|im_end|>'|
|  WizardLM-1.0-Uncensored-CodeLlama-34B ([download here](https://huggingface.co/second-state/WizardLM-1.0-Uncensored-CodeLlama-34b/resolve/main/WizardLM-1.0-Uncensored-CodeLlama-34b-ggml-model-q4_0.gguf))   | -p vicuna-chat -s 'You are a helpful AI assistant.' |
|  Samantha-1.11-CodeLlama-34B ([download here](https://huggingface.co/second-state/Samantha-1.11-CodeLlama-34B-GGUF/resolve/main/Samantha-1.11-CodeLlama-34b-ggml-model-q4_0.gguf))   | -p vicuna-chat -s 'You are a helpful AI assistant.' |
|  Samantha-1.11-7B ([download here](https://huggingface.co/second-state/Samantha-1.11-7B-GGUF/resolve/main/Samantha-1.11-7b-ggml-model-q4_0.gguf))   | -p vicuna-chat -s 'You are Samantha, a sentient AI companion.' |
|  WizardCoder-Python-7B-V1.0 ([download here](https://huggingface.co/second-state/WizardCoder-Python-7B-V1.0/resolve/main/WizardCoder-Python-7B-V1.0-ggml-model-q4_0.gguf))   | -p wizard-coder -s 'Below is an instruction that describes a task. Write a response that appropriately completes the request.'  |
|  Zephyr-7B-Alpha ([download here](https://huggingface.co/second-state/Zephyr-7B-Alpha-GGUF/resolve/main/zephyr-7b-alpha.Q5_K_M.gguf))   | -p zephyr -s 'You are a friendly chatbot who always responds in the style of a pirate.' -r '</s>'  |
|  WizardLM-7B-V1.0-Uncensored ([download here](https://huggingface.co/second-state/WizardLM-7B-V1.0-Uncensored-GGUF/resolve/main/wizardlm-7b-v1.0-uncensored.Q5_K_M.gguf))   | -p vicuna-chat -s 'You are a helpful AI assistant.'  |
|  WizardLM-13B-V1.0-Uncensored ([download here](https://huggingface.co/second-state/WizardLM-13B-V1.0-Uncensored-GGUF/resolve/main/wizardlm-7b-v1.0-uncensored.Q5_K_M.gguf))   | -p vicuna-chat -s 'You are a helpful AI assistant.'  |
|  Orca-2-13B ([download here](https://huggingface.co/second-state/Orca-2-13B-GGUF/resolve/main/Orca-2-13b-ggml-model-q4_0.gguf))   | -p chatml -s 'You are Orca, an AI language model created by Microsoft. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior.'  |
|  Neural-Chat-7B-v3-1 ([download here](https://huggingface.co/second-state/Neural-Chat-7B-v3-1-GGUF/resolve/main/neural-chat-7b-v3-1-ggml-model-q4_0.gguf))   |  -p intel-neural  |
|  Yi-34B-Chat ([download here](https://huggingface.co/second-state/Yi-34B-Chat-GGUF/resolve/main/Yi-34B-Chat-ggml-model-q4_0.gguf))   |  -p chatml -r '<|im_end|>'  |
|  Starling-LM-7B-alpha ([download here](https://huggingface.co/second-state/Starling-LM-7B-alpha-GGUF/resolve/main/starling-lm-7b-alpha.Q5_K_M.gguf))   |  -p openchat -r '<|end_of_turn|>'  |

</details>

## Execute

Execute the WASM with the `wasmedge` using the named model feature to preload large model. Here we use the `Llama-2-7B-Chat` model as an example:

```console
# download model
curl -LO https://huggingface.co/second-state/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q5_K_M.gguf

# run the `llama-chat` wasm app with the model
wasmedge --dir .:. --nn-preload default:GGML:AUTO:llama-2-7b-chat.Q5_K_M.gguf \
  llama-chat.wasm --prompt-template llama-2-chat
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

## Optional: Build the `llama-chat` wasm app yourself

Run the following command:

```console
cargo build --target wasm32-wasi --release
```

The `llama-chat.wasm` will be generated in the `target/wasm32-wasi/release` folder.
