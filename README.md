# llama-utils

The llama-utils shows you how to run LLM inference and build OpenAI-compatible API services for the Llama2 series of LLMs .


## Quick start

Run the one single command on your terminal to quick start. (macOS and Linux on CPU and GPU are avavible.)

```
bash <(curl -sSf https://raw.githubusercontent.com/second-state/llama-utils/main/run-llm.sh)
```
After that, please follow the prompt to install the WasmEdge Runtime and download your favorite open-source LLM. Then, you will be asked to choose whether you want to chat with the model via the CLI or via a web interface. 

[See in action](https://youtu.be/Hqu-PBqkzDk) | Docs

## How to use?

* The folder `api-server` includes the source code and instructions to create OpenAI-compatible API service for your llama2 model or the LLama2 model itself.
* The folder `chat` includes the source code and instructions to run llama2 models that can have continuous conversations.
* The folder `simple` includes the source code and instructions to run llama2 models that can answer one question.

## Why use llama-utils

The Rust+Wasm stack provides a strong alternative to Python in AI inference.

* Lightweight. The total runtime size is 30MB.
* Fast. Full native speed on GPUs.
* Portable. Single cross-platform binary on different CPUs, GPUs, and OSes.
* Secure. Sandboxed and isolated execution on untrusted devices.
* Container-ready. Supported in Docker, containerd, Podman, and Kubernetes.

For more information, please check out [Fast and Portable Llama2 Inference on the Heterogeneous Edge](https://www.secondstate.io/articles/fast-llm-inference/).

## Models

The llama-utils project, in theory, supports all Language Learning Models (LLMs) based on the llama2 framework in GGUF format. Below is a list of models that have been successfully verified to work on both Mac and Jetson Orin platforms. We are committed to continuously expanding this list by verifying additional models. If you have successfully operated other LLMs, don't hesitate to contribute by creating a Pull Request (PR) to help extend this list.

Click [here](./models.md) to see the supported model list and its download link, commands, and prompt template.

## Platforms

The compiled Wasm file is cross platfrom. You can use the same Wasm file to run the LLM both on CPU and GPU. 

The installer from WasmEdge 0.13.5 will detect cuda automatically. If CUDA is detected, the installer will always attempt to install a CUDA-enabled version of the plugin. The CUDA support is verified on the following platforms:
* Nvidia Jetson AGX Orin 64GB developer kit
* Intel i7-10700 + Nvidia GTX 1080 8G GPU
* AWS EC2 `g5.xlarge` + Nvidia A10G 24G GPU + Amazon deep learning base Ubuntu 20.04

If you're using CPU only machine, the installer will install the OpenBLAS version of the plugin instead. You may need to install `libopenblas-dev` by `apt update && apt install -y libopenblas-dev`.

| Platforms | Status |
|-----------|--------|
| macOS (apple silicon)   |   ✅     |
| Ubuntu (>= 20.04)    |     ✅   |
| General Linux      |   ✅     |


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
