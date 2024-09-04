# LlamaEdge

The LlamaEdge project makes it easy for you to run LLM inference apps and create OpenAI-compatible API services for the Llama2 series of LLMs locally.

⭐ Like our work? Give us a star!

Checkout our [official docs](https://llamaedge.com/docs) and a [Manning ebook](https://www.manning.com/liveprojectseries/open-source-llms-on-your-own-computer) on how to customize open source models.

## Quick start

Enhance your onboarding experience and quickly get started with LlamaEdge using the following scripts.

#1: Quick start without any argument

```
bash <(curl -sSfL 'https://raw.githubusercontent.com/LlamaEdge/LlamaEdge/main/run-llm.sh')
```

It will download and start the [Gemma-2-9b-it](https://huggingface.co/second-state/gemma-2-9b-it-GGUF) model automatically. Open http://127.0.0.1:8080 in your browser and start chatting right away!


#2: Specify a model using `--model model_name`

```
bash <(curl -sSfL 'https://raw.githubusercontent.com/LlamaEdge/LlamaEdge/main/run-llm.sh') --model llama-3-8b-instruct
```

The script will start an API server for the Llama3 8b model with a chatbot UI based on your choice. Open http://127.0.0.1:8080 in your browser and start chatting right away!

To explore all the available models, please use the following command line

```
bash <(curl -sSfL 'https://raw.githubusercontent.com/LlamaEdge/LlamaEdge/main/run-llm.sh') --model help
```
#3:  Interactively choose and confirm all steps in the script using using `--interactive` flag

```
bash <(curl -sSfL 'https://raw.githubusercontent.com/LlamaEdge/LlamaEdge/main/run-llm.sh') --interactive
```
Follow the on-screen instructions to install the WasmEdge Runtime and download your favorite open-source LLM. Then, choose whether you want to chat with the model via the CLI or via a web UI.

[See it in action](https://youtu.be/Hqu-PBqkzDk) | [Docs](https://www.secondstate.io/articles/run-llm-sh/)

## How it works?

The Rust source code for the inference applications are all open source and you can modify and use them freely for your own purposes.

* The folder `llama-simple` contains the source code project to generate text from a prompt using run llama2 models.
* The folder `llama-chat` contains the source code project to "chat" with a llama2 model on the command line.
* The folder `llama-api-server` contains the source code project for a web server. It provides an OpenAI-compatible API service, as well as an optional web UI, for llama2 models.

## The tech stack

The [Rust+Wasm stack](https://medium.com/stackademic/why-did-elon-musk-say-that-rust-is-the-language-of-agi-eb36303ce341) provides a strong alternative to Python in AI inference.

* Lightweight. The total runtime size is 30MB.
* Fast. Full native speed on GPUs.
* Portable. Single cross-platform binary on different CPUs, GPUs, and OSes.
* Secure. Sandboxed and isolated execution on untrusted devices.
* Container-ready. Supported in Docker, containerd, Podman, and Kubernetes.

For more information, please check out [Fast and Portable Llama2 Inference on the Heterogeneous Edge](https://www.secondstate.io/articles/fast-llm-inference/).

## Models

The LlamaEdge project supports all Large Language Models (LLMs) based on the llama2 framework. The model files must be in the GGUF format. We are committed to continuously testing and validating new open-source models that emerge every day.

[Click here](https://huggingface.co/second-state) to see the supported model list with a download link and startup commands for each model. If you have success with other LLMs, don't hesitate to contribute by creating a Pull Request (PR) to help extend this list.

## Platforms

The compiled Wasm file is cross platfrom. You can use the same Wasm file to run the LLM across OSes (e.g., MacOS, Linux, Windows SL), CPUs (e.g., x86, ARM, Apple, RISC-V), and GPUs (e.g., NVIDIA, Apple).

The installer from WasmEdge 0.13.5 will detect NVIDIA CUDA drivers automatically. If CUDA is detected, the installer will always attempt to install a CUDA-enabled version of the plugin. The CUDA support is tested on the following platforms in our automated CI.

* Nvidia Jetson AGX Orin 64GB developer kit
* Intel i7-10700 + Nvidia GTX 1080 8G GPU
* AWS EC2 `g5.xlarge` + Nvidia A10G 24G GPU + Amazon deep learning base Ubuntu 20.04

> If you're using CPU only machine, the installer will install the OpenBLAS version of the plugin instead. You may need to install `libopenblas-dev` by `apt update && apt install -y libopenblas-dev`.

## Troubleshooting

Q: Why I got the following errors after starting the API server?

```
[2024-03-05 16:09:05.800] [error] instantiation failed: module name conflict, Code: 0x60
[2024-03-05 16:09:05.801] [error]     At AST node: module
```

A: TThe module conflict error is a known issue, and these are false-positive errors. They do not impact your program's functionality.

Q: Even though my machine has a large RAM, after asking several questions, I received an error message returns 'Error: Backend Error: WASI-NN'. What should I do?

A: To enable machines with smaller RAM, like 8 GB, to run a 7b model, we've set the context size limit to 512. If your machine has more capacity, you can increase both the context size and batch size up to 4096 using the CLI options available [here](https://github.com/second-state/llama-utils/tree/main/chat#cli-options). Use these commands to adjust the settings:

```
-c, --ctx-size <CTX_SIZE>
-b, --batch-size <BATCH_SIZE>
```

Q: After running `apt update && apt install -y libopenblas-dev`, you may encounter the following error:

  ```bash
  ...
  E: Could not open lock file /var/lib/dpkg/lock-frontend - open (13: Permission denied)
  E: Unable to acquire the dpkg frontend lock (/var/lib/dpkg/lock-frontend), are you root?
  ```

A: This indicates that you are not logged in as `root`. Please try installing again using the `sudo` command:

  ```bash
  sudo apt update && sudo apt install -y libopenblas-dev
  ```

Q: After running the `wasmedge` command, you may receive the following error:

  ```bash
  [2023-10-02 14:30:31.227] [error] loading failed: invalid path, Code: 0x20
  [2023-10-02 14:30:31.227] [error]     load library failed:libblas.so.3: cannot open shared object file: No such file or directory
  [2023-10-02 14:30:31.227] [error] loading failed: invalid path, Code: 0x20
  [2023-10-02 14:30:31.227] [error]     load library failed:libblas.so.3: cannot open shared object file: No such file or directory
  unknown option: nn-preload
  ```

A: This suggests that your plugin installation was not successful. To resolve this issue, please attempt to install your desired plugin again.

Q: After executing the `wasmedge` command, you might encounter the error message: `[WASI-NN] GGML backend: Error: unable to init model.`

A: This error signifies that the model setup was not successful. To resolve this issue, please verify the following:

  1. Check if your model file and the WASM application are located in the same directory. The WasmEdge runtime requires them to be in the same location to locate the model file correctly.
  2. Ensure that the model has been downloaded successfully. You can use the command `shasum -a 256 <gguf-filename>` to verify the model's sha256sum. Compare your result with the correct sha256sum available on [the Hugging Face page](https://huggingface.co/second-state/Dolphin-2.2-Yi-34B-GGUF/blob/main/dolphin-2.2-yi-34b-ggml-model-q4_0.gguf) for the model.

<img width="1286" alt="image" src="https://github.com/second-state/llama-utils/assets/45785633/24286d8e-b438-4d1a-a443-62c1466e9992">

## Credits

The WASI-NN ggml plugin embedded [`llama.cpp`](git://github.com/ggerganov/llama.cpp.git@b1217) as its backend.
