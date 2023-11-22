# llama-utils

This is a project that shows you how to run LLM inference and build OpenAI-compatible API services for the Llama2 series of LLMswith Rust and WasmEdge.

## How to use?

* The folder `api-server` includes the source code and instructions to create OpenAI-compatible API service for your llama2 model or the LLama2 model itself.
* The folder `chat` includes the source code and instructions to run llama2 models that can have continuous conversations.
* The folder `simple` includes the source code and instructions to run llama2 models that can answer one question.

## Why use Rust + Wasm

The Rust+Wasm stack provides a strong alternative to Python in AI inference.

* Lightweight. The total runtime size is 30MB as opposed to 4GB for Python and 350MB for Ollama.
* Fast. Full native speed on GPUs.
* Portable. Single cross-platform binary on different CPUs, GPUs, and OSes.
* Secure. Sandboxed and isolated execution on untrusted devices.
* Container-ready. Supported in Docker, containerd, Podman, and Kubernetes.

For more information, please check out [Fast and Portable Llama2 Inference on the Heterogeneous Edge](https://www.secondstate.io/articles/fast-llm-inference/).

## Supported Models

The llama-utils project, in theory, supports all Language Learning Models (LLMs) based on the llama2 framework in GGUF format. Below is a list of models that have been successfully verified to work on both Mac and Jetson Orin platforms. We are committed to continuously expanding this list by verifying additional models. If you have successfully operated other LLMs, don't hesitate to contribute by creating a Pull Request (PR) to help extend this list.

- [x] [Llama-2-7B-Chat-GGUF](https://huggingface.co/second-state/Llama-2-7B-Chat-GGUF) | [Guide](https://www.secondstate.io/articles/llama-2-models/)

- [x] [Llama-2-13B-chat-GGUF](https://huggingface.co/second-state/Llama-2-13B-Chat-GGUF) | [Guide](https://www.secondstate.io/articles/llama-2-models/)

- [x] [CodeLlama-13B-GGUF](https://huggingface.co/second-state/CodeLlama-13B-Instruct-GGUF) | [Guide](https://www.secondstate.io/articles/code-llama/)
- [ ] [Getting Started with Orca-2-13B](https://huggingface.co/second-state/Orca-2-13B-GGUF/tree/main) [Guide](https://www.secondstate.io/articles/orca-2-13b/)

- [x] [Mistral-7B-Instruct-v0.1-GGUF](https://huggingface.co/second-state/Mistral-7B-Instruct-v0.1-GGUF) | [Guide](https://www.secondstate.io/articles/mistral-7b-instruct-v0.1/)

- [x] [MistralLite-7B-GGUF](https://huggingface.co/second-state/MistralLite-7B-GGUF) | [Guide](https://www.secondstate.io/articles/mistrallite/)
- [x] [WizardCoder-Python-7B](https://huggingface.co/second-state/WizardCoder-Python-7B-V1.0) | [Guide](https://www.secondstate.io/articles/wizardcoder-python-7b/)

- [x] [OpenChat-3.5-GGUF](https://huggingface.co/second-state/OpenChat-3.5-GGUF) | [Guide](https://www.secondstate.io/articles/openchat-3.5/)

- [x] [BELLE-Llama2-13B-Chat-0.4M-GGUF](https://huggingface.co/second-state/BELLE-Llama2-13B-Chat-0.4M-GGUF)
- [x] [Yi-34B](https://huggingface.co/second-state/Yi-34B-GGUF/tree/main) | [Guide](https://www.secondstate.io/articles/wizard-vicuna-13b/)

- [x] [wizard-vicuna-13B-GGUF](https://huggingface.co/second-state/wizard-vicuna-13B-GGUF) | [Guide](https://www.secondstate.io/articles/wizard-vicuna-13b/)
- [x] [Zephyr-7B](https://huggingface.co/TheBloke/zephyr-7B-alpha-GGUF/tree/main) | [Guide](https://www.secondstate.io/articles/zephyr-7b/)

- [x] [CausalLM-14B](https://huggingface.co/second-state/CausalLM-14B-GGUF) | [Guide](https://www.secondstate.io/articles/causallm-14b/)

- [x] [TinyLlama-1.1B-Chat-v0.3](https://huggingface.co/second-state/TinyLlama-1.1B-Chat-v0.3-GGUF) | [Guide](https://www.secondstate.io/articles/tinyllama-1.1b-chat/)

- [x] [Baichuan2-7B-Chat-GGUF](https://huggingface.co/second-state/Baichuan2-7B-Chat-GGUF)

- [x] [Baichuan2-13B-Chat-GGUF](https://huggingface.co/second-state/Baichuan-13B-Chat-GGUF) | [Guide](https://www.secondstate.io/articles/baichuan2-13b-chat/)

- [x] [OpenHermes-2.5-Mistral-7B-GGUF](https://huggingface.co/second-state/OpenHermes-2.5-Mistral-7B-GGUF)

- [x] [Dolphin-2.2-Yi-34B-GGUF](https://huggingface.co/second-state/Dolphin-2.2-Yi-34B-GGUF) | [Guide](https://www.secondstate.io/articles/dolphin-2.2-yi-34b/)

- [ ] [rpguild-chatml-13B-GGUF](https://huggingface.co/second-state/rpguild-chatml-13B-GGUF)

- [ ] [CodeShell-7B-Chat-GGUF](https://huggingface.co/second-state/CodeShell-7B-Chat-GGUF)

## Requirements

### For macOS (apple silicon)

Install WasmEdge 0.13.5+WASI-NN ggml plugin(Metal enabled on apple silicon) via installer

```bash
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | bash -s -- --plugin wasi_nn-ggml
# After install the wasmedge, you have to activate the environment.
# Assuming you use zsh (the default shell on macOS), you will need to run the following command
source $HOME/.zshenv
```

### For Ubuntu (>= 20.04)

#### CUDA enabled

The installer from WasmEdge 0.13.5 will detect cuda automatically.

If CUDA is detected, the installer will always attempt to install a CUDA-enabled version of the plugin.

Install WasmEdge 0.13.5+WASI-NN ggml plugin via installer

```bash
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | bash -s -- --plugin wasi_nn-ggml
# After installing the wasmedge, you have to activate the environment.
# Assuming you use bash (the default shell on Ubuntu), you will need to run the following command
source $HOME/.bashrc
```

This version is verified on the following platforms:
1. Nvidia Jetson AGX Orin 64GB developer kit
2. Intel i7-10700 + Nvidia GTX 1080 8G GPU
2. AWS EC2 `g5.xlarge` + Nvidia A10G 24G GPU + Amazon deep learning base Ubuntu 20.04

#### CPU only

If the CPU is the only available hardware on your machine, the installer will install the OpenBLAS version of the plugin instead.

You may need to install `libopenblas-dev` by `apt update && apt install -y libopenblas-dev`.

Install WasmEdge 0.13.5+WASI-NN ggml plugin via installer

```bash
apt update && apt install -y libopenblas-dev # You may need sudo if the user is not root.
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | bash -s -- --plugin wasi_nn-ggml
# After installing the wasmedge, you have to activate the environment.
# Assuming you use bash (the default shell on Ubuntu), you will need to run the following command
source $HOME/.bashrc
```

### For General Linux

Install WasmEdge 0.13.5+WASI-NN ggml plugin via installer

```bash
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | bash -s -- --plugin wasi_nn-ggml
# After install the wasmedge, you have to activate the environment.
# Assuming you use bash (the default shell on Ubuntu), you will need to run the following command
source $HOME/.bashrc
```

## Troubleshooting

- After running `apt update && apt install -y libopenblas-dev`, you may encounter the following error:

  ```bash
  ...
  E: Could not open lock file /var/lib/dpkg/lock-frontend - open (13: Permission denied)
  E: Unable to acquire the dpkg frontend lock (/var/lib/dpkg/lock-frontend), are you root?
  ```

   This indicates that you are not logged in as `root`. Please try installing again using the `sudo` command:

  ```bash
  sudo apt update && sudo apt install -y libopenblas-dev
  ```

- After running the `wasmedge` command, you may receive the following error:

  ```bash
  [2023-10-02 14:30:31.227] [error] loading failed: invalid path, Code: 0x20
  [2023-10-02 14:30:31.227] [error]     load library failed:libblas.so.3: cannot open shared object file: No such file or directory
  [2023-10-02 14:30:31.227] [error] loading failed: invalid path, Code: 0x20
  [2023-10-02 14:30:31.227] [error]     load library failed:libblas.so.3: cannot open shared object file: No such file or directory
  unknown option: nn-preload
  ```

  This suggests that your plugin installation was not successful. To resolve this issue, please attempt to install your desired plugin again.

- After executing the `wasmedge` command, you might encounter the error message: `[WASI-NN] GGML backend: Error: unable to init model.` This error signifies that the model setup was not successful. To resolve this issue, please verify the following:

  1. Check if your model file and the WASM application are located in the same directory. The WasmEdge runtime requires them to be in the same location to locate the model file correctly.
  2. Ensure that the model has been downloaded successfully. You can use the command `shasum -a 256 <gguf-filename>` to verify the model's sha256sum. Compare your result with the correct sha256sum available on [the Hugging Face page](https://huggingface.co/second-state/Dolphin-2.2-Yi-34B-GGUF/blob/main/dolphin-2.2-yi-34b-ggml-model-q4_0.gguf) for the model.
      
     <img width="1286" alt="image" src="https://github.com/second-state/llama-utils/assets/45785633/24286d8e-b438-4d1a-a443-62c1466e9992">

 
  

## Credits

The WASI-NN ggml plugin embedded [`llama.cpp`](git://github.com/ggerganov/llama.cpp.git@b1217) as its backend.
