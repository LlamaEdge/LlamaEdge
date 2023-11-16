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
          Prompt template. [default: llama-2-chat] [possible values: llama-2-chat, codellama-instruct, mistral-instruct-v0.1, mistrallite, openchat, belle-llama-2-chat, vicuna-chat, chatml, baichuan-2]
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

- [x] Llama-2-7B-Chat

  ```console
  # llama-2-7b
  curl -LO https://huggingface.co/second-state/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q5_K_M.gguf
  ```

  ```console
  # command to run the model
  wasmedge --dir .:. --nn-preload default:GGML:AUTO:llama-2-7b-chat.Q5_K_M.gguf llama-chat.wasm -p llama-2-chat
  ```

  Please check the sha256sum of the downloaded model file to make sure it is correct:

  ```text
  # llama-2-7b-chat.Q5_K_M.gguf
  e0b99920cf47b94c78d2fb06a1eceb9ed795176dfa3f7feac64629f1b52b997f
  ```

- [x] Llama-2-13B-Chat

  ```console
  # llama-2-13b
  curl -LO https://huggingface.co/second-state/Llama-2-13B-Chat-GGUF/resolve/main/llama-2-13b-chat.Q5_K_M.gguf
  ```

  ```console
  # command to run the model
  wasmedge --dir .:. --nn-preload default:GGML:AUTO:llama-2-13b-chat.Q5_K_M.gguf llama-chat.wasm -p llama-2-chat
  ```

  Please check the sha256sum of the downloaded model file to make sure it is correct:

  ```text
  # llama-2-13b-chat.Q5_K_M.gguf
  ef36e090240040f97325758c1ad8e23f3801466a8eece3a9eac2d22d942f548a
  ```

- [x] CodeLlama-13B-Instruct

  ```console
  # codellama-13b-instruct
  curl -LO curl -LO https://huggingface.co/second-state/CodeLlama-13B-Instruct-GGUF/resolve/main/codellama-13b-instruct.Q4_0.gguf
  ```

  ```console
  # command to run the model
  wasmedge --dir .:. --nn-preload default:GGML:AUTO:codellama-13b-instruct.Q4_0.gguf llama-chat.wasm -p codellama-instruct
  ```

  Please check the sha256sum of the downloaded model file to make sure it is correct:

  ```text
  # codellama-13b-instruct.Q4_0.gguf
  693021fa3a170a348b0a6104ab7d3a8c523331826a944dc0371fecd922df89dd
  ```

- [x] BELLE-Llama2-13B-Chat

  ```console
  # BELLE-Llama2-13B-Chat-0.4M
  curl -LO https://huggingface.co/second-state/BELLE-Llama2-13B-Chat-0.4M-GGUF/resolve/main/BELLE-Llama2-13B-Chat-0.4M-ggml-model-q4_0.gguf
  ```

  ```console
  # command to run the model
  wasmedge --dir .:. --nn-preload default:GGML:AUTO:BELLE-Llama2-13B-Chat-0.4M-ggml-model-q4_0.gguf llama-chat.wasm -p belle-llama-2-chat
  ```

  Please check the sha256sum of the downloaded model file to make sure it is correct:

  ```text
  # BELLE-Llama2-13B-Chat-0.4M-ggml-model-q4_0.gguf
  56879e1fd6ee6a138286730e121f2dba1be51b8f7e261514a594dea89ef32fe7
  ```

- [x] Mistral-7B-Instruct-v0.1

  ```console
  # mistral-7b-instruct-v0.1
  curl -LO https://huggingface.co/second-state/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q5_K_M.gguf
  ```

  ```console
  # command to run the model
  wasmedge --dir .:. --nn-preload default:GGML:AUTO:mistral-7b-instruct-v0.1.Q5_K_M.gguf llama-chat.wasm -p mistral-instruct-v0.1
  ```

  Please check the sha256sum of the downloaded model file to make sure it is correct:

  ```text
  # mistral-7b-instruct-v0.1.Q5_K_M.gguf
  c4b062ec7f0f160e848a0e34c4e291b9e39b3fc60df5b201c038e7064dbbdcdc

  # mistral-7b-instruct-v0.1.Q4_K_M.gguf
  14466f9d658bf4a79f96c3f3f22759707c291cac4e62fea625e80c7d32169991
  ```

- [x] MistralLite-7B

  ```console
  # mistral-lite-7b
  curl -LO https://huggingface.co/second-state/MistralLite-7B-GGUF/resolve/main/mistrallite.Q5_K_M.gguf
  ```

  ```console
  # command to run the model
  wasmedge --dir .:. --nn-preload default:GGML:AUTO:mistrallite.Q5_K_M.gguf llama-chat.wasm -p mistrallite
  ```

  Please check the sha256sum of the downloaded model file to make sure it is correct:

  ```text
  # mistrallite.Q5_K_M.gguf
  d06d149c24eea0446ea7aad596aca396fe7f3302441e9375d5bbd3fd9ba8ebea
  ```

- [x] OpenChat-3.5

  ```console
  # openchat-3.5
  curl -LO https://huggingface.co/second-state/OpenChat-3.5-GGUF/resolve/main/openchat_3.5.Q5_K_M.gguf
  ```

  ```console
  # command to run the model
  wasmedge --dir .:. --nn-preload default:GGML:AUTO:openchat_3.5.Q5_K_M.gguf llama-chat.wasm -p openchat -r '<|end_of_turn|>'
  ```

  Please check the sha256sum of the downloaded model file to make sure it is correct:

  ```text
  # openchat_3.5.Q5_K_M.gguf
  3abf26b0f2ff11394351a23f8d538a1404a2afb69465a6bbaba8836fef51899d
  ```

- [x] Wizard-Vicuna

  ```console
  # wizard-vicuna-13b
  curl -LO https://huggingface.co/second-state/wizard-vicuna-13B-GGUF/resolve/main/wizard-vicuna-13b-ggml-model-q8_0.gguf
  ```

  ```console
  # command to run the model
  wasmedge --dir .:. --nn-preload default:GGML:AUTO:wizard-vicuna-13b-ggml-model-q8_0.gguf llama-chat.wasm -p vicuna-chat
  ```

  Please check the sha256sum of the downloaded model file to make sure it is correct:

  ```text
  # wizard-vicuna-13b-ggml-model-q8_0.gguf
  681b6571e624fd211ae81308b573f24f0016f6352252ae98241b44983bb7e756
  ```

- [x] CausalLM-14B

  ```console
  # CausalLM-14B
  curl -LO https://huggingface.co/second-state/CausalLM-14B-GGUF/resolve/main/causallm_14b.Q5_1.gguf
  ```

  ```console
  # command to run the model
  wasmedge --dir .:. --nn-preload default:GGML:AUTO:causallm_14b.Q5_1.gguf llama-chat.wasm -p chatml
  ```

  Please check the sha256sum of the downloaded model file to make sure it is correct:

  ```text
  # causallm_14b.Q5_1.gguf
  8ddb4c04e6f0c06971e9b6723688206bf9a5b8ffc85611cc7843c0e8c8a66c4e
  ```

- [x] TinyLlama-1.1B-Chat-v0.3

  ```console
  # TinyLlama-1.1B-Chat-v0.3
  curl -LO https://huggingface.co/second-state/TinyLlama-1.1B-Chat-v0.3-GGUF/resolve/main/tinyllama-1.1b-chat-v0.3.Q5_K_M.gguf
  ```

  ```console
  # command to run the model
  wasmedge --dir .:. --nn-preload default:GGML:AUTO:tinyllama-1.1b-chat-v0.3.Q5_K_M.gguf llama-chat.wasm -p chatml
  ```

  Please check the sha256sum of the downloaded model file to make sure it is correct:

  ```text
  # tinyllama-1.1b-chat-v0.3.Q5_K_M.gguf
  7c255febbf29c97b5d6f57cdf62db2f2bc95c0e541dc72c0ca29786ca0fa5eed
  ```

- [x] Baichuan2-13B-Chat

  ```console
  # Baichuan2-13B-Chat
  curl -LO https://huggingface.co/second-state/Baichuan2-13B-Chat-GGUF/resolve/main/Baichuan2-13B-Chat-ggml-model-q4_0.gguf
  ```

  ```console
  # command to run the model
  wasmedge --dir .:. --nn-preload default:GGML:AUTO:Baichuan2-13B-Chat-ggml-model-q4_0.gguf llama-chat.wasm -p baichuan-2 -r '用户:'
  ```

  Please check the sha256sum of the downloaded model file to make sure it is correct:

  ```text
  # Baichuan2-13B-Chat-ggml-model-q4_0.gguf
  789685b86c86af68a1886949015661d3da0a9c959dffaae773afa4fe8cfdb840
  ```

- [x] Baichuan2-7B-Chat

  ```console
  # Baichuan2-7B-Chat
  curl -LO https://huggingface.co/second-state/Baichuan2-7B-Chat-GGUF/resolve/main/Baichuan2-7B-Chat-ggml-model-q4_0.gguf
  ```

  ```console
  # command to run the model
  wasmedge --dir .:. --nn-preload default:GGML:AUTO:Baichuan2-7B-Chat-ggml-model-q4_0.gguf llama-chat.wasm -p baichuan-2 -r '用户:'
  ```

  Please check the sha256sum of the downloaded model file to make sure it is correct:

  ```text
  # Baichuan2-7B-Chat-ggml-model-q4_0.gguf
  82deec2b1ed20fa996b45898abfcff699a92e8a6dc8e53e4fd487328ec9181a9
  ```

- [x] OpenHermes-2.5-Mistral-7B

  ```console
  # OpenHermes-2.5-Mistral-7B
  curl -LO https://huggingface.co/second-state/OpenHermes-2.5-Mistral-7B-GGUF/resolve/main/openhermes-2.5-mistral-7b.Q5_K_M.gguf
  ```

  ```console
  # command to run the model
  wasmedge --dir .:. --nn-preload default:GGML:AUTO:openhermes-2.5-mistral-7b.Q5_K_M.gguf llama-chat.wasm -p chatml -r '<|im_end|>'
  ```

  Please check the sha256sum of the downloaded model file to make sure it is correct:

  ```text
  # openhermes-2.5-mistral-7b.Q5_K_M.gguf
  61e9e801d9e60f61a4bf1cad3e29d975ab6866f027bcef51d1550f9cc7d2cca6
  ```

- [x] Dolphin-2.2-Yi-34B

  ```console
  # Dolphin-2.2-Yi-34B
  curl -LO https://huggingface.co/second-state/Dolphin-2.2-Yi-34B-GGUF/resolve/main/dolphin-2.2-yi-34b-ggml-model-q4_0.gguf
  ```

  ```console
  # command to run the model
  wasmedge --dir .:. --nn-preload default:GGML:AUTO:dolphin-2.2-yi-34b-ggml-model-q4_0.gguf llama-chat.wasm -p chatml -r '<|im_end|>' -s 'You are a helpful AI assistant'
  ```

  Please check the sha256sum of the downloaded model file to make sure it is correct:

  ```text
  # dolphin-2.2-yi-34b-ggml-model-q4_0.gguf
  641b644fde162fd7f8e8991ca6873d8b0528b7a027f5d56b8ee005f7171ac002
  ```

- [x] Dolphin-2.2-Mistral-7B

  ```console
  # Dolphin-2.2-Mistral-7B
  curl -LO https://huggingface.co/second-state/Dolphin-2.2-Mistral-7B-GGUF/resolve/main/dolphin-2.2-mistral-7b-ggml-model-q4_0.gguf
  ```

  ```console
  # command to run the model
  wasmedge --dir .:. --nn-preload default:GGML:AUTO:dolphin-2.2-mistral-7b-ggml-model-q4_0.gguf llama-chat.wasm -p chatml -r '<|im_end|>'
  ```

  Please check the sha256sum of the downloaded model file to make sure it is correct:

  ```text
  # dolphin-2.2-mistral-7b-ggml-model-q4_0.gguf
  77cf0861b5bc064e222075d0c5b73205d262985fc195aed6d30a7d3bdfefbd6c
  ```

- [x] Dolphin-2.2.1-Mistral-7B

  ```console
  # Dolphin-2.2.1-Mistral-7B
  curl -LO https://huggingface.co/second-state/Dolphin-2.2.1-Mistral-7B/resolve/main/dolphin-2.2.1-mistral-7b-ggml-model-q4_0.gguf
  ```

  ```console
  # command to run the model
  wasmedge --dir .:. --nn-preload default:GGML:AUTO:dolphin-2.2.1-mistral-7b-ggml-model-q4_0.gguf llama-chat.wasm -p chatml -r '<|im_end|>'
  ```

  Please check the sha256sum of the downloaded model file to make sure it is correct:

  ```text
  # dolphin-2.2.1-mistral-7b-ggml-model-q4_0.gguf
  c88edaa19afeb45075d566930571fc1f580329c6d6980f5222f442ee2894234e
  ```

- [x] Samantha-1.2-Mistral-7B

  ```console
  # Dolphin-2.2.1-Mistral-7B
  curl -LO https://huggingface.co/second-state/Samantha-1.2-Mistral-7B/resolve/main/samantha-1.2-mistral-7b-ggml-model-q4_0.gguf
  ```

  ```console
  # command to run the model
  wasmedge --dir .:. --nn-preload default:GGML:AUTO:samantha-1.2-mistral-7b-ggml-model-q4_0.gguf llama-chat.wasm -p chatml -r '<|im_end|>'
  ```

  Please check the sha256sum of the downloaded model file to make sure it is correct:

  ```text
  # samantha-1.2-mistral-7b-ggml-model-q4_0.gguf
  c29d3e84c626b6631864cf111ed2ce847d74a105f3bd66845863bbd8ea06628e
  ```

- [x] Dolphin-2.1-Mistral-7B

  ```console
  # Dolphin-2.1-Mistral-7B
  curl -LO https://huggingface.co/second-state/Dolphin-2.1-Mistral-7B-GGUF/resolve/main/dolphin-2.1-mistral-7b-ggml-model-q4_0.gguf
  ```

  ```console
  # command to run the model
  wasmedge --dir .:. --nn-preload default:GGML:AUTO:dolphin-2.1-mistral-7b-ggml-model-q4_0.gguf llama-chat.wasm -p chatml -r '<|im_end|>'
  ```

  Please check the sha256sum of the downloaded model file to make sure it is correct:

  ```text
  # dolphin-2.1-mistral-7b-ggml-model-q4_0.gguf
  021b2d9eb466e2b2eb522bc6d66906bb94c0dac721d6278e6718a4b6c9ecd731
  ```

- [x] Dolphin-2.0-Mistral-7B

  ```console
  # Dolphin-2.0-Mistral-7B
  curl -LO https://huggingface.co/second-state/Dolphin-2.0-Mistral-7B-GGUF/resolve/main/dolphin-2.0-mistral-7b-ggml-model-q4_0.gguf
  ```

  ```console
  # command to run the model
  wasmedge --dir .:. --nn-preload default:GGML:AUTO:dolphin-2.0-mistral-7b-ggml-model-q4_0.gguf llama-chat.wasm -p chatml -r '<|im_end|>'
  ```

  Please check the sha256sum of the downloaded model file to make sure it is correct:

  ```text
  # dolphin-2.0-mistral-7b-ggml-model-q4_0.gguf
  37adbc161e6e98354ab06f6a79eaf30c4eb8dc60fb1226ef2fe8e84a84c5fdd6
  ```

- [x] WizardLM-1.0-Uncensored-CodeLlama-34B

  ```console
  # WizardLM-1.0-Uncensored-CodeLlama-34b
  curl -LO https://huggingface.co/second-state/WizardLM-1.0-Uncensored-CodeLlama-34b/resolve/main/WizardLM-1.0-Uncensored-CodeLlama-34b-ggml-model-q4_0.gguf
  ```

  ```console
  # command to run the model
  wasmedge --dir .:. --nn-preload default:GGML:AUTO:WizardLM-1.0-Uncensored-CodeLlama-34b-ggml-model-q4_0.gguf llama-chat.wasm -p vicuna-chat -s 'You are a helpful AI assistant.'
  ```

  Please check the sha256sum of the downloaded model file to make sure it is correct:

  ```text
  # WizardLM-1.0-Uncensored-CodeLlama-34b-ggml-model-q4_0.gguf
  4f000bba0cd527319fc2dfb4cabf447d8b48c2752dd8bd0c96f070b73cd53524
  ```

- [x] Samantha-1.11-CodeLlama-34B

  ```console
  # Samantha-1.11-CodeLlama-34B
  curl -LO https://huggingface.co/second-state/Samantha-1.11-CodeLlama-34B-GGUF/resolve/main/Samantha-1.11-CodeLlama-34b-ggml-model-q4_0.gguf
  ```

  ```console
  # command to run the model
  wasmedge --dir .:. --nn-preload default:GGML:AUTO:Samantha-1.11-CodeLlama-34b-ggml-model-q4_0.gguf llama-chat.wasm -p vicuna-chat -s 'You are a helpful AI assistant.' --stream-stdout
  ```

  Please check the sha256sum of the downloaded model file to make sure it is correct:

  ```text
  # Samantha-1.11-CodeLlama-34b-ggml-model-q4_0.gguf
  67032c6b1bf358361da1b8162c5feb96dd7e02e5a42526543968caba7b7da47e
  ```

- [ ] Samantha-Mistral-Instruct-7B

- [ ] Samantha-Mistral-7B

- [ ] Dolphin-2.1-70B

- [ ] Dolphin-2.2-70B

- [ ] rpguild-chatml (Coming soon)

- [ ] CodeShell-Chat (Coming soon)

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
