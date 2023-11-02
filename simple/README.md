# Simple text completion

## Dependencies

Install the latest wasmedge with plugins:

```bash
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | bash -s -- --plugins wasi_nn-ggml
```

## Get the compiled wasm binary program

Download the wasm file:

```bash
curl -LO https://github.com/second-state/llama-utils/raw/main/simple/llama-simple.wasm
```

## Get Model

Download llama model:

```bash
curl -LO https://huggingface.co/second-state/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q5_K_M.gguf
```

## Execute

Execute the WASM with the `wasmedge` using the named model feature to preload large model:

```bash
LLAMA_LOG=1 LLAMA_N_CTX=4096 LLAMA_N_PREDICT=128 wasmedge --dir .:. \
  --nn-preload default:GGML:CPU:llama-2-7b.Q5_K_M.gguf llama-simple.wasm default \
  --prompt 'Robert Oppenheimer most important achievement is ' \
  --ctx-size 4096
```

- The CLI options of `llama-simple` wasm app:

  ```console
  ~/llama-utils/simple$ wasmedge llama-simple.wasm -h
  Usage: llama-simple.wasm [OPTIONS] --prompt <PROMPT>

  Options:
    -p, --prompt <PROMPT>
            Sets the prompt string, including system message if required.
    -m, --model-alias <ALIAS>
            Sets the model alias [default: default]
    -c, --ctx-size <CTX_SIZE>
            Sets the prompt context size [default: 2048]
    -n, --n-predict <N_PRDICT>
            Number of tokens to predict [default: 1024]
    -g, --n-gpu-layers <N_GPU_LAYERS>
            Number of layers to run on the GPU [default: 0]
    -b, --batch-size <BATCH_SIZE>
            Batch size for prompt processing [default: 512]
    -r, --reverse-prompt <REVERSE_PROMPT>
            Halt generation at PROMPT, return control.
        --log-enable
            Enable trace logs
    -h, --help
            Print help
  ```

After executing the command, it takes some time to wait for the output.
Once the execution is complete, the following output will be generated:

```console
...................................................................................................
[2023-10-08 23:13:10.272] [info] [WASI-NN] GGML backend: set n_ctx to 4096
llama_new_context_with_model: kv self size  = 2048.00 MB
llama_new_context_with_model: compute buffer total size =  297.47 MB
llama_new_context_with_model: max tensor size =   102.54 MB
[2023-10-08 23:13:10.472] [info] [WASI-NN] GGML backend: llama_system_info: AVX = 0 | AVX2 = 0 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | FMA = 0 | NEON = 1 | ARM_FMA = 1 | F16C = 0 | FP16_VA = 1 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 0 | SSSE3 = 0 | VSX = 0 |
[2023-10-08 23:13:10.472] [info] [WASI-NN] GGML backend: set n_predict to 128
[2023-10-08 23:13:16.014] [info] [WASI-NN] GGML backend: llama_get_kv_cache_token_count 128

llama_print_timings:        load time =  1431.58 ms
llama_print_timings:      sample time =     3.53 ms /   118 runs   (    0.03 ms per token, 33446.71 tokens per second)
llama_print_timings: prompt eval time =  1230.69 ms /    11 tokens (  111.88 ms per token,     8.94 tokens per second)
llama_print_timings:        eval time =  4295.81 ms /   117 runs   (   36.72 ms per token,    27.24 tokens per second)
llama_print_timings:       total time =  5742.71 ms
Robert Oppenheimer most important achievement is
1945 Manhattan Project.
Robert Oppenheimer was born in New York City on April 22, 1904. He was the son of Julius Oppenheimer, a wealthy German-Jewish textile merchant, and Ella Friedman Oppenheimer.
Robert Oppenheimer was a brilliant student. He attended the Ethical Culture School in New York City and graduated from the Ethical Culture Fieldston School in 1921. He then attended Harvard University, where he received his bachelor's degree
```

## Optional: Build the wasm file yourself

Compile the application to WebAssembly:

```bash
cargo build --target wasm32-wasi --release
```

The output wasm file will be at `target/wasm32-wasi/release/`.
