# GGUF Models

**To catch up the latest model, please visit [Second State's Huggingface page](https://huggingface.co/second-state), which includes different kinds of auantized models.** 

You can find the model download link, the command to run the model, the command to create an OpenAI compatible API server for the model, and the sha256sum of the model.

Before you start, you need to [install WasmEdge and its ggml plugin via one single command line](https://github.com/LlamaEdge/LlamaEdge/tree/main/api-server#dependencies).

<details>
<summary> <b>Llama-2-7B-Chat</b> </summary>
<hr/>
<b>Download the model</b>

```bash
curl -LO https://huggingface.co/second-state/Llama-2-7B-Chat-GGUF/resolve/main/Llama-2-7b-chat-hf-Q5_K_M.gguf
```

Please check the sha256sum of the downloaded model file to make sure it is correct.

```text
39fdaca41ef03de1e9b709602557faaf2e8490c830622823cb6f8dc9ac14db04
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-chat.wasm

wasmedge --dir .:. --nn-preload default:GGML:AUTO:Llama-2-7b-chat-hf-Q5_K_M.gguf llama-chat.wasm -p llama-2-chat
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/LlamaEdge/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:Llama-2-7b-chat-hf-Q5_K_M.gguf llama-api-server.wasm -p llama-2-chat
```

Open your browser to http://localhost:8080 to start the chat!

<b>Send an API request to the server</b>

Test the API server from another terminal using the following command

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'accept:application/json' \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "What is the capital of France?"}], "model":"llama-2-7b-chat"}'
```

</details>

<details>
<summary> <b>Llama-2-13B-Chat</b> </summary>
<hr/>
<b>Download the model</b>

```bash
curl -LO https://huggingface.co/second-state/Llama-2-13B-Chat-GGUF/resolve/main/Llama-2-13b-chat-hf-Q5_K_M.gguf
```

Please check the sha256sum of the downloaded model file to make sure it is correct.

```text
67c08278e8ed7ae96e25e8968e77ba8fc4ae8b974a8b47a105880756f8f82f3e
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-chat.wasm
wasmedge --dir .:. --nn-preload default:GGML:AUTO:Llama-2-13b-chat-hf-Q5_K_M.gguf llama-chat.wasm -p llama-2-chat
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/LlamaEdge/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:Llama-2-13b-chat-hf-Q5_K_M.gguf llama-api-server.wasm -p llama-2-chat
```

Open your browser to http://localhost:8080 to start the chat!

<b>Send an API request to the server</b>

Test the API server from another terminal using the following command

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'accept:application/json' \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "What is the capital of France"}], "model":"llama-2-13b-chat"}'
```

</details>

<details>
<summary> <b>CodeLlama-13B-Instruct</b> </summary>
<hr/>
<b>Download the model</b>

```bash
curl -LO https://huggingface.co/second-state/CodeLlama-13B-Instruct-GGUF/resolve/main/CodeLlama-13b-Instruct-hf-Q5_K_M.gguf
```

Please check the sha256sum of the downloaded model file to make sure it is correct:

```bash
b30d01b5a22f2b3dc6cd01084b50114dde5e63cbc240ee7ad20ebbd6c63eab95
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-chat.wasm

wasmedge --dir .:. --nn-preload default:GGML:AUTO:CodeLlama-13b-Instruct-hf-Q5_K_M.gguf llama-chat.wasm -p codellama-instruct
```

<b>This model isn't suitable for API server</b>
</details>

<details>
<summary> <b>Mistral-7B-Instruct-v0.1</b> </summary>
<hr/>
<b>Download the model</b>

```bash
curl -LO https://huggingface.co/second-state/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/Mistral-7B-Instruct-v0.1-Q5_K_M.gguf
```

Please check the sha256sum of the downloaded model file to make sure it is correct:

```bash
287a6520a937fcdb9d1d21b1f9145ba3c8624a4c8ce5411dae5e74991a911a94
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-chat.wasm
wasmedge --dir .:. --nn-preload default:GGML:AUTO:Mistral-7B-Instruct-v0.1-Q5_K_M.gguf llama-chat.wasm -p mistral-instruct
```
<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/LlamaEdge/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:Mistral-7B-Instruct-v0.1-Q5_K_M.gguf llama-api-server.wasm -p mistral-instruct
```

Open your browser to http://localhost:8080 to start the chat!

<b>Send an API request to the server</b>

Test the API server from another terminal using the following command

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'accept:application/json' \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user", "content": "What is the capital of France?"}], "model":"Mistral-7B-Instruct-v0.1"}'
```

</details>

<details>
<summary> <b>Mistral-7B-Instruct-v0.2</b> </summary>
<hr/>
<b>Download the model</b>

```bash
curl -LO https://huggingface.co/second-state/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/Mistral-7B-Instruct-v0.2-Q5_K_M.gguf
```

Please check the sha256sum of the downloaded model file to make sure it is correct:

```bash
4ce9a46d73b47ca1d46aa0f182c12bd18ee2f3bcfffcc397de191ae31c3c3c4e
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-chat.wasm

wasmedge --dir .:. --nn-preload default:GGML:AUTO:Mistral-7B-Instruct-v0.2-Q5_K_M.gguf llama-chat.wasm -p mistral-instruct
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/LlamaEdge/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:Mistral-7B-Instruct-v0.2-Q5_K_M.gguf llama-api-server.wasm -p mistral-instruct
```

Open your browser to http://localhost:8080 to start the chat!

<b>Send an API request to the server</b>

Test the API server from another terminal using the following command

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'accept:application/json' \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user", "content": "What is the capital of France?"}], "model":"Mistral-7B-Instruct-v0.2"}'
```

</details>

<details>
<summary> <b>MistralLite-7B</b> </summary>
<hr/>
<b>Download the model</b>

```bash
curl -LO https://huggingface.co/second-state/MistralLite-7B-GGUF/resolve/main/MistralLite-Q5_K_M.gguf
```

Please check the sha256sum of the downloaded model file to make sure it is correct.

```bash
c8f5d6117cc9ec8dceb2e28e1268770c0c32f39949fceceb105d1e0837e07361
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-chat.wasm

wasmedge --dir .:. --nn-preload default:GGML:AUTO:MistralLite-Q5_K_M.gguf llama-chat.wasm -p mistrallite -r '</s>'
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/LlamaEdge/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:MistralLite-Q5_K_M.gguf llama-api-server.wasm -p mistrallite -r '</s>'
```

Open your browser to http://localhost:8080 to start the chat!

<b>Send an API request to the server</b>

Test the API server from another terminal using the following command

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'accept:application/json' \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "What is the capital of France?"}], "model":"MistralLite-7B"}'
```

</details>

<details>
<summary> <b>OpenChat-3.5-0106</b> </summary>
<hr/>
<b>Download the model</b>

```bash
curl -LO https://huggingface.co/second-state/OpenChat-3.5-0106-GGUF/resolve/main/openchat-3.5-0106-Q5_K_M.gguf
```

Please check the sha256sum of the downloaded model file to make sure it is correct.

```bash
c28f69693336ab63369451da7f1365e5003d79f3ac69566de72100a8299a967a
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-chat.wasm
wasmedge --dir .:. --nn-preload default:GGML:AUTO:openchat-3.5-0106-Q5_K_M.gguf llama-chat.wasm -p openchat -r '<|end_of_turn|>'
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/LlamaEdge/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:openchat-3.5-0106-Q5_K_M.gguf llama-api-server.wasm -p openchat -r '<|end_of_turn|>'
```

Open your browser to http://localhost:8080 to start the chat!

<b>Send an API request to the server</b>

Test the API server from another terminal using the following command

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'accept:application/json' \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "What is the capital of France?"}], "model":"OpenChat-3.5-0106"}'
```

</details>

<details>
<summary> <b>OpenChat-3.5-1210</b> </summary>
<hr/>
<b>Download the model</b>

```bash
curl -LO https://huggingface.co/second-state/openchat-3.5-1210-GGUF/resolve/main/openchat-3.5-1210-Q5_K_M.gguf
```

Please check the sha256sum of the Downloaded model file to make sure it is correct.

```bash
e1c5c50d0185d047f53ceb48a7c02d33f0a7fe0e1467f98c4b575502e9cabbdd
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-chat.wasm

wasmedge --dir .:. --nn-preload default:GGML:AUTO:openchat-3.5-1210-Q5_K_M.gguf llama-chat.wasm -p openchat -r '<|end_of_turn|>'
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/LlamaEdge/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:openchat-3.5-1210-Q5_K_M.gguf llama-api-server.wasm -p openchat -r '<|end_of_turn|>'
```

Open your browser to http://localhost:8080 to start the chat!

<b>Send an API request to the server</b>

Test the API server from another terminal using the following command

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'accept:application/json' \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "What is the capital of France?"}], "model":"OpenChat-3.5-1210"}'
```

</details>

<details>
<summary> <b>OpenChat-3.5</b> </summary>
<hr/>
<b>Download the model</b>

```bash
curl -LO https://huggingface.co/second-state/OpenChat-3.5-GGUF/resolve/main/openchat_3.5-Q5_K_M.gguf
```

Please check the sha256sum of the downloaded model file to make sure it is correct.

```bash
cea9e09a3e1d0fa779224710a543a07d92af46a64090af7a32001b94faf66a92
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-chat.wasm

wasmedge --dir .:. --nn-preload default:GGML:AUTO:openchat_3.5-Q5_K_M.gguf llama-chat.wasm -p openchat -r '<|end_of_turn|>'
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/LlamaEdge/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:openchat_3.5-Q5_K_M.gguf llama-api-server.wasm -p openchat -r '<|end_of_turn|>'
```

Open your browser to http://localhost:8080 to start the chat!

<b>Send an API request to the server</b>

Test the API server from another terminal using the following command

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'accept:application/json' \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "What is the capital of France?"}], "model":"OpenChat-3.5"}'
```

</details>

<details>
<summary> <b>Wizard-Vicuna-13B-Uncensored-GGUF</b> </summary>
<hr/>
<b>Download the model</b>

```bash
curl -LO https://huggingface.co/second-state/Wizard-Vicuna-13B-Uncensored-GGUF/resolve/main/Wizard-Vicuna-13B-Uncensored-Q5_K_M.gguf
```

Please check the sha256sum of the Downloaded model file to make sure it is correct.

```bash
bb6bda4e7383f1be98d7a9ab8c6cfff6daebb937badb11c25ed16e0f908f5b4d
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-chat.wasm

wasmedge --dir .:. --nn-preload default:GGML:AUTO:Wizard-Vicuna-13B-Uncensored-Q5_K_M.gguf llama-chat.wasm -p vicuna-1.0-chat
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/LlamaEdge/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:Wizard-Vicuna-13B-Uncensored-Q5_K_M.gguf llama-api-server.wasm -p vicuna-1.0-chat
```

Open your browser to http://localhost:8080 to start the chat!

<b>Send an API request to the server</b>

Test the API server from another terminal using the following command

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'accept:application/json' \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "What is the capital of France?"}], "model":"wizard-vicuna-13B"}'
```

</details>

<details>
<summary> <b>TinyLlama-1.1B-Chat-v1.0</b> </summary>
<hr/>
<b>Download the model</b>

```bash
curl -LO https://huggingface.co/second-state/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/TinyLlama-1.1B-Chat-v1.0-Q5_K_M.gguf
```

Please check the sha256sum of the downloaded model file to make sure it is correct.

```bash
4602b3e1751346bc22e6454fa2670f743351546401cb353a10c8b5329075e67f
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-chat.wasm

wasmedge --dir .:. --nn-preload default:GGML:AUTO:TinyLlama-1.1B-Chat-v1.0-Q5_K_M.gguf llama-chat.wasm -p chatml
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/LlamaEdge/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:TinyLlama-1.1B-Chat-v1.0-Q5_K_M.gguf llama-api-server.wasm -p chatml
```

Open your browser to http://localhost:8080 to start the chat!

<b>Send an API request to the server</b>

Test the API server from another terminal using the following command

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'accept:application/json' \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "What is the capital of France?"}], "model":"TinyLlama-1.1B-Chat-v1.0"}'
```

</details>

<details>
<summary> <b>Baichuan2-13B-Chat</b> </summary>
<hr/>
<b>Download the model</b>

```bash
curl -LO https://huggingface.co/second-state/Baichuan2-13B-Chat-GGUF/resolve/main/Baichuan2-13B-Chat-Q5_K_M.gguf
```

Please check the sha256sum of the Downloaded model file to make sure it is correct.

```bash
2b7781b78d27dd4d15bf171649b1114c8591bccb8a98b9d9a0cff1386e536b24
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-chat.wasm

wasmedge --dir .:. --nn-preload default:GGML:AUTO:Baichuan2-13B-Chat-Q5_K_M.gguf llama-chat.wasm -p baichuan-2 -r '用户:'
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/LlamaEdge/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:Baichuan2-13B-Chat-Q5_K_M.gguf llama-api-server.wasm -p baichuan-2 -r '用户:'
```

Open your browser to http://localhost:8080 to start the chat!

<b>Send an API request to the server</b>

Test the API server from another terminal using the following command

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'accept:application/json' \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "李白是谁"}], "model":"Baichuan2-13B-Chat"}'
```

</details>

<details>
<summary> <b>OpenHermes-2.5-Mistral-7B</b> </summary>
<hr/>
<b>Download the model</b>

```bash
curl -LO https://huggingface.co/second-state/OpenHermes-2.5-Mistral-7B-GGUF/resolve/main/OpenHermes-2.5-Mistral-7B-Q5_K_M.gguf
```

Please check the sha256sum of the downloaded model file to make sure it is correct.

```bash
76746d87d2c47ce32218fe05e4b20e5fa1849f3a33c743101309e48912581536
```

<b>Chat with the model on the CLI</b>

```bash
wasmedge --dir .:. --nn-preload default:GGML:AUTO:OpenHermes-2.5-Mistral-7B-Q5_K_M.gguf llama-chat.wasm -p chatml -r '<|im_end|>'
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/LlamaEdge/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:OpenHermes-2.5-Mistral-7B-Q5_K_M.gguf llama-api-server.wasm -p chatml -r '<|im_end|>'
```

Open your browser to http://localhost:8080 to start the chat!

<b>Send an API request to the server</b>

Test the API server from another terminal using the following command

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'accept:application/json' \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "What is the capital of France?"}], "model":"OpenHermes-2.5-Mistral-7B"}'
```

</details>

<details>
<summary> <b>Dolphin-2.2-Yi-34B</b> </summary>
<hr/>
<b>Download the model</b>

```bash
curl -LO https://huggingface.co/second-state/Dolphin-2.2-Yi-34B-GGUF/resolve/main/dolphin-2_2-yi-34b-Q5_K_M.gguf
```

Please check the sha256sum of the downloaded model file to make sure it is correct.

```bash
28e80a924fae51644f7d869d9ad3eec72bafd28b5f221015e8a56a328847ac19
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-chat.wasm

wasmedge --dir .:. --nn-preload default:GGML:AUTO:dolphin-2_2-yi-34b-Q5_K_M.gguf llama-chat.wasm -p chatml -r '<|im_end|>' -s 'You are a helpful AI assistant'
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/LlamaEdge/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:dolphin-2_2-yi-34b-Q5_K_M.gguf llama-api-server.wasm -p chatml -r '<|im_end|>' -s 'You are a helpful AI assistant'
```

Open your browser to http://localhost:8080 to start the chat!

<b>Send an API request to the server</b>

Test the API server from another terminal using the following command

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'accept:application/json' \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "What is the capital of France?"}], "model":"Dolphin-2.2-Yi-34B"}'
```

</details>

<details>
<summary> <b>Dolphin-2.6-Mistral-7B</b> </summary>
<hr/>
<b>Download the model</b>

```bash
curl -LO https://huggingface.co/second-state/dolphin-2.6-mistral-7B-GGUF/resolve/main/dolphin-2.6-mistral-7b-Q5_K_M.gguf
```

Please check the sha256sum of the downloaded model file to make sure it is correct.

```bash
ec3c988cda2d831542449fcd0e82a039067a8da2c747b05268eee482b0e12bdf
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-chat.wasm

wasmedge --dir .:. --nn-preload default:GGML:AUTO:dolphin-2.6-mistral-7b-Q5_K_M.gguf llama-chat.wasm -p chatml
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/LlamaEdge/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:dolphin-2.6-mistral-7b-Q5_K_M.gguf llama-api-server.wasm -p chatml
```

Open your browser to http://localhost:8080 to start the chat!

<b>Send an API request to the server</b>

Test the API server from another terminal using the following command

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'accept:application/json' \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "What is the capital of France?"}], "model":"dolphin-2.6-mistral-7b"}'
```

</details>


<details>
<summary> <b>Samantha-1.2-Mistral-7B</b> </summary>
<hr/>
<b>Download the model</b>

```bash
curl -LO https://huggingface.co/second-state/Samantha-1.2-Mistral-7B-GGUF/resolve/main/samantha-1.2-mistral-7b-Q5_K_M.gguf
```

Please check the sha256sum of the downloaded model file to make sure it is correct.

```bash
1051ff70a76561776427c22fe022f8984166bdeca82a1c0c2edcd6fa6d2c5dee
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-chat.wasm

wasmedge --dir .:. --nn-preload default:GGML:AUTO:samantha-1.2-mistral-7b-Q5_K_M.gguf llama-chat.wasm -p chatml -r '<|im_end|>'
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/LlamaEdge/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:samantha-1.2-mistral-7b-Q5_K_M.gguf llama-api-server.wasm -p chatml -r '<|im_end|>'
```

Open your browser to http://localhost:8080 to start the chat!

<b>Send an API request to the server</b>

Test the API server from another terminal using the following command

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'accept:application/json' \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "What is the capital of France?"}], "model":"Samantha-1.2-Mistral-7B"}'
```

</details>

<details>
<summary> <b>Samantha-1.11-CodeLlama-34B</b> </summary>
<hr/>
<b>Download the model</b>

```bash
curl -LO https://huggingface.co/second-state/Samantha-1.11-CodeLlama-34B-GGUF/resolve/main/Samantha-1.11-CodeLlama-34b-Q5_K_M.gguf
```

Please check the sha256sum of the downloaded model file to make sure it is correct.

```bash
325a14a42c657845aed815b7699dc876df2b830c01f78d0c3fada8a67b4c56e0
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-chat.wasm

wasmedge --dir .:. --nn-preload default:GGML:AUTO:Samantha-1.11-CodeLlama-34b-Q5_K_M.gguf llama-chat.wasm -p vicuna-1.0-chat -s 'You are a helpful AI assistant.'
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/LlamaEdge/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:Samantha-1.11-CodeLlama-34b-Q5_K_M.gguf llama-api-server.wasm -p vicuna-1.0-chat -s 'You are a helpful AI assistant.'
```

Open your browser to http://localhost:8080 to start the chat!

<b>Send an API request to the server</b>

Test the API server from another terminal using the following command

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'accept:application/json' \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "What is the capital of France?"}], "model":"Samantha-1.11-CodeLlama-34b"}'
```

</details>

<details>
<summary> <b>WizardCoder-Python-7B-V1.0</b> </summary>
<hr/>
<b>Download the model</b>

```bash
curl -LO https://huggingface.co/second-state/WizardCoder-Python-7B-v1.0-GGUF/resolve/main/WizardCoder-Python-7B-V1.0-Q5_K_M.gguf
```

Please check the sha256sum of the Downloaded model file to make sure it is correct.

```bash
ff61076feb2f3c9d049d12869532d6f5feb855ce501eff1a3d155ac6d29f283a
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-chat.wasm

wasmedge --dir .:. --nn-preload default:GGML:AUTO:WizardCoder-Python-7B-V1.0-Q5_K_M.gguf llama-chat.wasm -p wizard-coder -s 'Below is an instruction that describes a task. Write a response that appropriately completes the request.'
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/LlamaEdge/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:WizardCoder-Python-7B-V1.0-Q5_K_M.gguf llama-api-server.wasm -p wizard-coder
```

Open your browser to http://localhost:8080 to start the chat!

<b>Send an API request to the server</b>

Test the API server from another terminal using the following command

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'accept:application/json' \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "What is the capital of France?"}], "model":"WizardCoder-Python-7B"}'
```

</details>

<details>
<summary> <b>Zephyr-7B-Alpha</b> </summary>
<hr/>
<b>Download the model</b>

```bash
curl -LO https://huggingface.co/second-state/Zephyr-7B-Alpha-GGUF/resolve/main/zephyr-7b-alpha-Q5_K_M.gguf
```

Please check the sha256sum of the downloaded model file to make sure it is correct.

```bash
cd035904c4b16904049c2ba7e45f1b34ad2868af3ecbe51d8c77daa371b96245
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-chat.wasm

wasmedge --dir .:. --nn-preload default:GGML:AUTO:zephyr-7b-alpha-Q5_K_M.gguf llama-chat.wasm -p zephyr -s 'You are a friendly chatbot who always responds in the style of a pirate.' -r '</s>'
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/LlamaEdge/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:zephyr-7b-alpha-Q5_K_M.gguf llama-api-server.wasm -p zephyr -r '</s>'
```

Open your browser to http://localhost:8080 to start the chat!

<b>Send an API request to the server</b>

Test the API server from another terminal using the following command

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'accept:application/json' \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "What is the capital of France?"}], "model":"Zephyr-7B"}'
```

</details>

<details>
<summary> <b>WizardLM-13B-V1.0-Uncensored</b> </summary>
<hr/>
<b>Download the model</b>

```bash
curl -LO https://huggingface.co/second-state/WizardLM-13B-V1.0-Uncensored-GGUF/resolve/main/WizardLM-13B-V1.0-Uncensored-Q5_K_M.gguf
```

Please check the sha256sum of the downloaded model file to make sure it is correct.

```bash
a8329103ecc3a5a736b76e633970f39ded3f0a75a4d29f37f9e46d180ce2234b
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-chat.wasm

wasmedge --dir .:. --nn-preload default:GGML:AUTO:WizardLM-13B-V1.0-Uncensored-Q5_K_M.gguf llama-chat.wasm -p vicuna-1.0-chat -s 'You are a helpful AI assistant.'
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/LlamaEdge/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:WizardLM-13B-V1.0-Uncensored-Q5_K_M.gguf llama-api-server.wasm -p vicuna-1.0-chat
```

Open your browser to http://localhost:8080 to start the chat!

<b>Send an API request to the server</b>

Test the API server from another terminal using the following command

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'accept:application/json' \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "What is the capital of France?"}], "model":"WizardLM-13B-V1.0-Uncensored"}'
```

</details>

<details>
<summary> <b>Orca-2-13B</b> </summary>
<hr/>
<b>Download the model</b>

```bash
curl -LO https://huggingface.co/second-state/Orca-2-13B-GGUF/resolve/main/Orca-2-13b-Q5_K_M.gguf
```

Please check the sha256sum of the downloaded model file to make sure it is correct.

```bash
104d8239756f5bc861d0c5a407035e894f54218bc2e32b7b7ae437bc8dc6079d
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-chat.wasm

wasmedge --dir .:. --nn-preload default:GGML:AUTO:Orca-2-13b-Q5_K_M.gguf llama-chat.wasm -p chatml -s 'You are Orca, an AI language model created by Microsoft. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior.'
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/LlamaEdge/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:Orca-2-13b-Q5_K_M.gguf llama-api-server.wasm -p chatml
```

Open your browser to http://localhost:8080 to start the chat!

<b>Send an API request to the server</b>

Test the API server from another terminal using the following command

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'accept:application/json' \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "What is the capital of France?"}], "model":"Orca-2-13B"}'
```

</details>

<details>
<summary> <b>Neural-Chat-7B-v3-1</b> </summary>
<hr/>
<b>Download the model</b>

```bash
curl -LO https://huggingface.co/second-state/Neural-Chat-7B-v3-1-GGUF/resolve/main/neural-chat-7b-v3-1-Q5_K_M.gguf
```

Please check the sha256sum of the Downloaded model file to make sure it is correct.

```bash
c754cefc47842167b229fc78bff511f96c173c00962e5dbb44ea11d206492370
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-chat.wasm

wasmedge --dir .:. --nn-preload default:GGML:AUTO:neural-chat-7b-v3-1-Q5_K_M.gguf llama-chat.wasm -p intel-neural
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/LlamaEdge/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:neural-chat-7b-v3-1-Q5_K_M.gguf llama-api-server.wasm -p intel-neural
```

Open your browser to http://localhost:8080 to start the chat!

<b>Send an API request to the server</b>

Test the API server from another terminal using the following command

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'accept:application/json' \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "What is the capital of France?"}], "model":"Neural-Chat-7B-v3-1"}'
```

</details>

<details>
<summary> <b>Yi-34B-Chat</b> </summary>
<hr/>
<b>Download the model</b>

```bash
curl -LO https://huggingface.co/second-state/Yi-34B-Chat-GGUF/resolve/main/Yi-34B-Chat-Q5_K_M.gguf
```

Please check the sha256sum of the downloaded model file to make sure it is correct.

```bash
b9693f42372a06ca8b044ab2c4db84e4359de207be1cc11fcf023f09a8238f76
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-chat.wasm

wasmedge --dir .:. --nn-preload default:GGML:AUTO:Yi-34B-Chat-Q5_K_M.gguf llama-chat.wasm -p chatml -r '<|im_end|>'
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/LlamaEdge/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:Yi-34B-Chat-Q5_K_M.gguf llama-api-server.wasm -p chatml -r '<|im_end|>'
```

Open your browser to http://localhost:8080 to start the chat!

<b>Send an API request to the server</b>

Test the API server from another terminal using the following command

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'accept:application/json' \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "What is the capital of France?"}], "model":"Yi-34B-Chat"}'
```

</details>

<details>
<summary> <b>Starling-LM-7B-alpha</b> </summary>
<hr/>
<b>Download the model</b>

```bash
curl -LO https://huggingface.co/second-state/Starling-LM-7B-alpha-GGUF/resolve/main/Starling-LM-7B-alpha-Q5_K_M.gguf
```

Please check the sha256sum of the downloaded model file to make sure it is correct.

```text
8022640fea02e50b294a5ca3b9701f753e3870f61c596b16e16e8fac4f130cea
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-chat.wasm

wasmedge --dir .:. --nn-preload default:GGML:AUTO:Starling-LM-7B-alpha-Q5_K_M.gguf llama-chat.wasm -p openchat -r '<|end_of_turn|>'
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/LlamaEdge/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:Starling-LM-7B-alpha-Q5_K_M.gguf llama-api-server.wasm -p openchat -r '<|end_of_turn|>'
```

Open your browser to http://localhost:8080 to start the chat!

<b>Send an API request to the server</b>

Test the API server from another terminal using the following command

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'accept:application/json' \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "What is the capital of France?"}], "model":"Starling-LM-7B"}'
```

</details>

<details>
<summary> <b>DeepSeek-Coder-6.7B</b> </summary>
<hr/>
<b>Download the model</b>

```bash
curl -LO https://huggingface.co/second-state/Deepseek-Coder-6.7B-Instruct-GGUF/resolve/main/deepseek-coder-6.7b-instruct-Q5_K_M.gguf
```

Note that check the sha256 of `deepseek-coder-6.7b-instruct-Q5_K_M.gguf` after downloading.

```text
81c6fb56729ca9f95a73edd23ad58b4e64a27b53d0171a03716690b4bed8b2fc
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-chat.wasm

wasmedge --dir .:. --nn-preload default:GGML:AUTO:deepseek-coder-6.7b-instruct-Q5_K_M.gguf llama-chat.wasm -p deepseek-coder
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/LlamaEdge/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:deepseek-coder-6.7b-instruct-Q5_K_M.gguf llama-api-server.wasm -p deepseek-coder
```

Open your browser to http://localhost:8080 to start the chat!

<b>Send an API request to the server</b>

Test the API server from another terminal using the following command

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'accept:application/json' \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"system", "content": "You are an AI programming assistant."}, {"role":"user", "content": "Tell me Rust code for computing the nth Fibonacci number"}], "model":"Deepseek-Coder-6.7B"}'
```

</details>

<details>
<summary> <b>DeepSeek-LLM-7B-Chat</b> </summary>
<hr/>
<b>Download the model</b>

```bash
curl -LO https://huggingface.co/second-state/Deepseek-LLM-7B-Chat-GGUF/resolve/main/deepseek-llm-7b-chat-Q5_K_M.gguf
```

Note that check the sha256 of `deepseek-llm-7b-chat-Q5_K_M.gguf` after downloading.

```text
521dd4f2e740aaad46577dd0c85a2c4549968471ad626759bc685c8b6c557d78
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-chat.wasm

wasmedge --dir .:. --nn-preload default:GGML:AUTO:deepseek-llm-7b-chat-Q5_K_M.gguf llama-chat.wasm -p deepseek-chat
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/LlamaEdge/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:deepseek-llm-7b-chat-Q5_K_M.gguf llama-api-server.wasm -p deepseek-chat
```

Open your browser to http://localhost:8080 to start the chat!

<b>Send an API request to the server</b>

Test the API server from another terminal using the following command

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'accept:application/json' \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"system", "content": "You are an AI programming assistant."}, {"role":"user", "content": "What is the capital of Paris"}], "model":"Deepseek-LLM-7B"}'
```

</details>

<details>
<summary> <b>SOLAR-10.7B-Instruct-v1.0</b> </summary>
<hr/>
<b>Download the model</b>

```bash
curl -LO https://huggingface.co/second-state/SOLAR-10.7B-Instruct-v1.0-GGUF/resolve/main/SOLAR-10.7B-Instruct-v1.0-Q5_K_M.gguf
```

Note that check the sha256 of `solar-10.7b-instruct-v1.0.Q5_K_M.gguf` after downloading.

```text
715704d0c565664cf49dc6b4e0e087871724b7cb00ecf36a126df1d3de26b843
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-chat.wasm

wasmedge --dir .:. --nn-preload default:GGML:AUTO:SOLAR-10.7B-Instruct-v1.0-Q5_K_M.gguf llama-chat.wasm -p solar-instruct
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/LlamaEdge/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:SOLAR-10.7B-Instruct-v1.0-Q5_K_M.gguf llama-api-server.wasm -p solar-instruct
```

Open your browser to http://localhost:8080 to start the chat!

<b>Send an API request to the server</b>

Test the API server from another terminal using the following command

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'accept:application/json' \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"system", "content": "You are an AI programming assistant."}, {"role":"user", "content": "What is the capital of France?"}], "model":"SOLAR-10.7B-Instruct-v1.0"}'
```

</details>

<details>
<summary> <b>Mixtral-8x7B-Instruct-v0.1</b> </summary>
<hr/>
<b>Download the model</b>

```bash
curl -LO https://huggingface.co/second-state/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main/Mixtral-8x7B-Instruct-v0.1-Q5_K_M.gguf
```

Note that check the sha256 of `Mixtral-8x7B-Instruct-v0.1-Q5_K_M.gguf` after downloading.

```text
ffc48e5363b830341d157b7501374232badbf98c18384aecb93ff5adbfe0bdd7
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-chat.wasm
wasmedge --dir .:. --nn-preload default:GGML:AUTO:Mixtral-8x7B-Instruct-v0.1-Q5_K_M.gguf llama-chat.wasm -p mistral-instruct
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/LlamaEdge/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:Mixtral-8x7B-Instruct-v0.1-Q5_K_M.gguf llama-api-server.wasm -p mistral-instruct
```

Open your browser to http://localhost:8080 to start the chat!

<b>Send an API request to the server</b>

Test the API server from another terminal using the following command

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'accept:application/json' \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"system", "content": "You are an AI programming assistant."}, {"role":"user", "content": "What is the capital of Paris?"}], "model":"mixtral-8x7b-instruct-v0.1"}'
```

</details>

<details>
<summary> <b>Nous-Hermes-2-Mixtral-8x7B-DPO</b> </summary>
<hr/>
<b>Download the model</b>

```bash
curl -LO https://huggingface.co/second-state/Nous-Hermes-2-Mixtral-8x7B-DPO-GGUF/resolve/main/Nous-Hermes-2-Mixtral-8x7B-DPO-Q5_K_M.gguf
```

Note that check the sha256 of `Nous-Hermes-2-Mixtral-8x7B-DPO-Q5_K_M.gguf` after downloading.

```text
90c325215de925f47d76e391aee3a6bbac3859cdc03c744ff925b4ff9dd381e2
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-chat.wasm
wasmedge --dir .:. --nn-preload default:GGML:AUTO:Nous-Hermes-2-Mixtral-8x7B-DPO-Q5_K_M.gguf llama-chat.wasm -p chatml
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/LlamaEdge/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:Nous-Hermes-2-Mixtral-8x7B-DPO-Q5_K_M.gguf llama-api-server.wasm -p chatml
```

Open your browser to http://localhost:8080 to start the chat!

<b>Send an API request to the server</b>

Test the API server from another terminal using the following command

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'accept:application/json' \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"system", "content": "You are a sentient, superintelligent artificial general intelligence, here to teach and assist me."}, {"role":"user", "content": "Write a short story about Goku discovering kirby has teamed up with Majin Buu to destroy the world."}], "model":"Nous-Hermes-2-Mixtral-8x7B-DPO"}'
```

</details>

<details>
<summary> <b>Nous-Hermes-2-Mixtral-8x7B-SFT</b> </summary>
<hr/>
<b>Download the model</b>

```bash
curl -LO https://huggingface.co/second-state/Nous-Hermes-2-Mixtral-8x7B-SFT-GGUF/resolve/main/Nous-Hermes-2-Mixtral-8x7B-SFT-Q5_K_M.gguf
```

Note that check the sha256 of `Nous-Hermes-2-Mixtral-8x7B-SFT-Q5_K_M.gguf` after downloading.

```text
2599f102be866a80a86b5f03f75500704f6cd7de2dce51d27dd07293eb716770
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-chat.wasm
wasmedge --dir .:. --nn-preload default:GGML:AUTO:Nous-Hermes-2-Mixtral-8x7B-SFT-Q5_K_M.gguf llama-chat.wasm -p chatml
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/LlamaEdge/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:Nous-Hermes-2-Mixtral-8x7B-SFT-Q5_K_M.gguf llama-api-server.wasm -p chatml
```

Open your browser to http://localhost:8080 to start the chat!

<b>Send an API request to the server</b>

Test the API server from another terminal using the following command

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'accept:application/json' \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"system", "content": "You are a sentient, superintelligent artificial general intelligence, here to teach and assist me."}, {"role":"user", "content": "Write a short story about Goku discovering kirby has teamed up with Majin Buu to destroy the world."}], "model":"Nous-Hermes-2-Mixtral-8x7B-SFT"}'
```

</details>
