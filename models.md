# GGUF Models

You can find the model download link, the command to run the model, the command to create an OpenAI compatible API server for the model, and the sha256sum of the model.

<details>
<summary> <b>Llama-2-7B-Chat</b> </summary>
<hr/>
<b>Download the model</b>

```bash
curl -LO https://huggingface.co/second-state/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q5_K_M.gguf
```

Please check the sha256sum of the Downloaded model file to make sure it is correct.

```bash
shasum -a 256 llama-2-7b-chat.Q5_K_M.gguf
output: e0b99920cf47b94c78d2fb06a1eceb9ed795176dfa3f7feac64629f1b52b997f llama-2-7b-chat.Q5_K_M.gguf
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-chat.wasm
wasmedge --dir .:. --nn-preload default:GGML:AUTO:llama-2-7b-chat.Q5_K_M.gguf llama-chat.wasm -p llama-2-chat
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/second-state/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:llama-2-7b-chat.Q5_K_M.gguf llama-api-server.wasm -p llama-2-chat
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
curl -LO https://huggingface.co/second-state/Llama-2-13B-Chat-GGUF/resolve/main/llama-2-13b-chat.Q5_K_M.gguf
```

Please check the sha256sum of the Downloaded model file to make sure it is correct.

```bash
shasum -a 256 llama-2-13b-chat.Q5_K_M.gguf
output: ef36e090240040f97325758c1ad8e23f3801466a8eece3a9eac2d22d942f548a llama-2-13b-chat.Q5_K_M.gguf
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-chat.wasm
wasmedge --dir .:. --nn-preload default:GGML:AUTO:llama-2-13b-chat.Q5_K_M.gguf llama-chat.wasm -p llama-2-chat
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/second-state/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:llama-2-13b-chat.Q5_K_M.gguf llama-api-server.wasm -p llama-2-chat
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
curl -LO curl -LO https://huggingface.co/second-state/CodeLlama-13B-Instruct-GGUF/resolve/main/codellama-13b-instruct.Q4_0.gguf
```

Please check the sha256sum of the Downloaded model file to make sure it is correct:

```bash
shasum -a 256 codellama-13b-instruct.Q4_0.gguf
693021fa3a170a348b0a6104ab7d3a8c523331826a944dc0371fecd922df89dd codellama-13b-instruct.Q4_0.gguf
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-chat.wasm
wasmedge --dir .:. --nn-preload default:GGML:AUTO:codellama-13b-instruct.Q4_0.gguf llama-chat.wasm -p codellama-instruct
```

<b>This model isn't suitable for creating a API server</b>
</details>

<details>
<summary> <b>BELLE-Llama2-13B-Chat-0.4M</b> </summary>
<hr/>
<b>Download the model</b>

```bash
curl -LO https://huggingface.co/second-state/BELLE-Llama2-13B-Chat-0.4M-GGUF/resolve/main/BELLE-Llama2-13B-Chat-0.4M-ggml-model-q4_0.gguf
```

Please check the sha256sum of the Downloaded model file to make sure it is correct.

```bash
shasum -a 256 BELLE-Llama2-13B-Chat-0.4M-ggml-model-q4_0.gguf
output: 56879e1fd6ee6a138286730e121f2dba1be51b8f7e261514a594dea89ef32fe7 BELLE-Llama2-13B-Chat-0.4M-ggml-model-q4_0.gguf
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-chat.wasm
wasmedge --dir .:. --nn-preload default:GGML:AUTO:BELLE-Llama2-13B-Chat-0.4M-ggml-model-q4_0.gguf llama-chat.wasm -p belle-llama-2-chat --reverse-prompt "Human:"
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/second-state/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:BELLE-Llama2-13B-Chat-0.4M-ggml-model-q4_0.gguf llama-api-server.wasm -p belle-llama-2-chat --reverse-prompt "Human:"
```

Open your browser to http://localhost:8080 to start the chat!

<b>Send an API request to the server</b>

Test the API server from another terminal using the following command

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'accept:application/json' \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "What is the capital of France?"}], "model":"BELLE-Llama2-13B-Chat"}'
```

</details>

<details>
<summary> <b>Mistral-7B-Instruct-v0.1</b> </summary>
<hr/>
<b>Download the model</b>

```bash
curl -LO https://huggingface.co/second-state/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q5_K_M.gguf
```

Please check the sha256sum of the Downloaded model file to make sure it is correct:

```bash
shasum -a 256 mistral-7b-instruct-v0.1.Q5_K_M.gguf
output: c4b062ec7f0f160e848a0e34c4e291b9e39b3fc60df5b201c038e7064dbbdcdc mistral-7b-instruct-v0.1.Q5_K_M.gguf

shasum -a 256 mistral-7b-instruct-v0.1.Q4_K_M.gguf
output: 14466f9d658bf4a79f96c3f3f22759707c291cac4e62fea625e80c7d32169991 mistral-7b-instruct-v0.1.Q4_K_M.gguf
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-chat.wasm
wasmedge --dir .:. --nn-preload default:GGML:AUTO:mistral-7b-instruct-v0.1.Q5_K_M.gguf llama-chat.wasm -p mistral-instruct
```
<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/second-state/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:mistral-7b-instruct-v0.1.Q5_K_M.gguf llama-api-server.wasm -p mistral-instruct
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
curl -LO https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_0.gguf
```

Please check the sha256sum of the Downloaded model file to make sure it is correct:

```bash
shasum -a 256 mistral-7b-instruct-v0.2.Q4_0.gguf
output: 25d80b918e4432661726ef408b248005bebefe3f8e1ac722d55d0c5dcf2893e0 mistral-7b-instruct-v0.2.Q4_0.gguf
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-chat.wasm
wasmedge --dir .:. --nn-preload default:GGML:AUTO:mistral-7b-instruct-v0.2.Q4_0.gguf llama-chat.wasm -p mistral-instruct
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/second-state/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:mistral-7b-instruct-v0.2.Q4_0.gguf llama-api-server.wasm -p mistral-instruct
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
curl -LO https://huggingface.co/second-state/MistralLite-7B-GGUF/resolve/main/mistrallite.Q5_K_M.gguf
```

Please check the sha256sum of the Downloaded model file to make sure it is correct.

```bash
shasum -a 256 mistrallite.Q5_K_M.gguf
output: d06d149c24eea0446ea7aad596aca396fe7f3302441e9375d5bbd3fd9ba8ebea mistrallite.Q5_K_M.gguf
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-chat.wasm
wasmedge --dir .:. --nn-preload default:GGML:AUTO:mistrallite.Q5_K_M.gguf llama-chat.wasm -p mistrallite -r '</s>'
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/second-state/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:mistrallite.Q5_K_M.gguf llama-api-server.wasm -p mistrallite -r '</s>'
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

Please check the sha256sum of the Downloaded model file to make sure it is correct.

```bash
shasum -a 256 openchat-3.5-0106-Q5_K_M.gguf
output: c28f69693336ab63369451da7f1365e5003d79f3ac69566de72100a8299a967a
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-chat.wasm
wasmedge --dir .:. --nn-preload default:GGML:AUTO:openchat-3.5-0106-Q5_K_M.gguf llama-chat.wasm -p openchat -r '<|end_of_turn|>'
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/second-state/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
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
curl -LO https://huggingface.co/second-state/openchat-3.5-1210-GGUF/resolve/main/openchat-3.5-1210.Q5_K_M.gguf
```

Please check the sha256sum of the Downloaded model file to make sure it is correct.

```bash
shasum -a 256 openchat-3.5-1210.Q5_K_M.gguf
output: fbbb15ed13c630110b1790e0a0a621419fbf2f5564ead433c6ded9f8f196b95c
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-chat.wasm
wasmedge --dir .:. --nn-preload default:GGML:AUTO:openchat-3.5-1210.Q5_K_M.gguf llama-chat.wasm -p openchat -r '<|end_of_turn|>'
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/second-state/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:openchat-3.5-1210.Q5_K_M.gguf llama-api-server.wasm -p openchat -r '<|end_of_turn|>'
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
curl -LO https://huggingface.co/second-state/OpenChat-3.5-GGUF/resolve/main/openchat_3.5.Q5_K_M.gguf
```

Please check the sha256sum of the Downloaded model file to make sure it is correct.

```bash
shasum -a 256 openchat_3.5.Q5_K_M.gguf
output: 3abf26b0f2ff11394351a23f8d538a1404a2afb69465a6bbaba8836fef51899d openchat_3.5.Q5_K_M.gguf
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-chat.wasm
wasmedge --dir .:. --nn-preload default:GGML:AUTO:openchat_3.5.Q5_K_M.gguf llama-chat.wasm -p openchat -r '<|end_of_turn|>'
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/second-state/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:openchat_3.5.Q5_K_M.gguf llama-api-server.wasm -p openchat -r '<|end_of_turn|>'
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
<summary> <b>Wizard-Vicuna</b> </summary>
<hr/>
<b>Download the model</b>

```bash
curl -LO https://huggingface.co/second-state/wizard-vicuna-13B-GGUF/resolve/main/wizard-vicuna-13b-ggml-model-q8_0.gguf
```

Please check the sha256sum of the Downloaded model file to make sure it is correct.

```bash
shasum -a 256 wizard-vicuna-13b-ggml-model-q8_0.gguf
output: 681b6571e624fd211ae81308b573f24f0016f6352252ae98241b44983bb7e756 wizard-vicuna-13b-ggml-model-q8_0.gguf
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-chat.wasm
wasmedge --dir .:. --nn-preload default:GGML:AUTO:wizard-vicuna-13b-ggml-model-q8_0.gguf llama-chat.wasm -p vicuna-chat
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/second-state/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:wizard-vicuna-13b-ggml-model-q8_0.gguf llama-api-server.wasm -p vicuna-chat
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
<summary> <b>CausalLM-14B</b> </summary>
<hr/>
<b>Download the model</b>

```bash
curl -LO https://huggingface.co/second-state/CausalLM-14B-GGUF/resolve/main/causallm_14b.Q5_1.gguf
```

Please check the sha256sum of the Downloaded model file to make sure it is correct.

```bash
shasum -a 256 causallm_14b.Q5_1.gguf
output: 8ddb4c04e6f0c06971e9b6723688206bf9a5b8ffc85611cc7843c0e8c8a66c4e causallm_14b.Q5_1.gguf
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-chat.wasm
wasmedge --dir .:. --nn-preload default:GGML:AUTO:causallm_14b.Q5_1.gguf llama-chat.wasm -p chatml
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/second-state/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:causallm_14b.Q5_1.gguf llama-api-server.wasm -p chatml
```

Open your browser to http://localhost:8080 to start the chat!

<b>Send an API request to the server</b>

Test the API server from another terminal using the following command

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'accept:application/json' \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "What is the capital of France?"}], "model":"CausalLM-14B"}'
```

</details>

<details>
<summary> <b>TinyLlama-1.1B-Chat-v1.0</b> </summary>
<hr/>
<b>Download the model</b>

```bash
curl -LO https://huggingface.co/second-state/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf
```

Please check the sha256sum of the Downloaded model file to make sure it is correct.

```bash
shasum -a 256 tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf
output: aa54a5fb99ace5b964859cf072346631b2da6109715a805d07161d157c66ce7f
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-chat.wasm
wasmedge --dir .:. --nn-preload default:GGML:AUTO:tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf llama-chat.wasm -p chatml
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/second-state/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf llama-api-server.wasm -p chatml
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
<summary> <b>TinyLlama-1.1B-Chat-v0.3</b> </summary>
<hr/>
<b>Download the model</b>

```bash
curl -LO https://huggingface.co/second-state/TinyLlama-1.1B-Chat-v0.3-GGUF/resolve/main/tinyllama-1.1b-chat-v0.3.Q5_K_M.gguf
```

Please check the sha256sum of the Downloaded model file to make sure it is correct.

```bash
shasum -a 256 tinyllama-1.1b-chat-v0.3.Q5_K_M.gguf
output: 7c255febbf29c97b5d6f57cdf62db2f2bc95c0e541dc72c0ca29786ca0fa5eed
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-chat.wasm
wasmedge --dir .:. --nn-preload default:GGML:AUTO:tinyllama-1.1b-chat-v0.3.Q5_K_M.gguf llama-chat.wasm -p chatml
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/second-state/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:tinyllama-1.1b-chat-v0.3.Q5_K_M.gguf llama-api-server.wasm -p chatml
```

Open your browser to http://localhost:8080 to start the chat!

<b>Send an API request to the server</b>

Test the API server from another terminal using the following command

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'accept:application/json' \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "What is the capital of France?"}], "model":"TinyLlama-1.1B-Chat"}'
```

</details>

<details>
<summary> <b>Baichuan2-13B-Chat</b> </summary>
<hr/>
<b>Download the model</b>

```bash
curl -LO https://huggingface.co/second-state/Baichuan2-13B-Chat-GGUF/resolve/main/Baichuan2-13B-Chat-ggml-model-q4_0.gguf
```

Please check the sha256sum of the Downloaded model file to make sure it is correct.

```bash
shasum -a 256 Baichuan2-13B-Chat-ggml-model-q4_0.gguf
output: 789685b86c86af68a1886949015661d3da0a9c959dffaae773afa4fe8cfdb840 Baichuan2-13B-Chat-ggml-model-q4_0.gguf
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-chat.wasm
wasmedge --dir .:. --nn-preload default:GGML:AUTO:Baichuan2-13B-Chat-ggml-model-q4_0.gguf llama-chat.wasm -p baichuan-2 -r '用户:'
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/second-state/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:Baichuan2-13B-Chat-ggml-model-q4_0.gguf llama-api-server.wasm -p baichuan-2 -r '用户:'
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
curl -LO https://huggingface.co/second-state/OpenHermes-2.5-Mistral-7B-GGUF/resolve/main/openhermes-2.5-mistral-7b.Q5_K_M.gguf
```

Please check the sha256sum of the Downloaded model file to make sure it is correct.

```bash
shasum -a 256 openhermes-2.5-mistral-7b.Q5_K_M.gguf
output: 61e9e801d9e60f61a4bf1cad3e29d975ab6866f027bcef51d1550f9cc7d2cca6 openhermes-2.5-mistral-7b.Q5_K_M.gguf
```

<b>Chat with the model on the CLI</b>

```bash
wasmedge --dir .:. --nn-preload default:GGML:AUTO:openhermes-2.5-mistral-7b.Q5_K_M.gguf llama-chat.wasm -p chatml -r '<|im_end|>'
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/second-state/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:openhermes-2.5-mistral-7b.Q5_K_M.gguf llama-api-server.wasm -p chatml -r '<|im_end|>'
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
curl -LO https://huggingface.co/second-state/Dolphin-2.2-Yi-34B-GGUF/resolve/main/dolphin-2.2-yi-34b-ggml-model-q4_0.gguf
```

Please check the sha256sum of the Downloaded model file to make sure it is correct.

```bash
shasum -a 256 dolphin-2.2-yi-34b-ggml-model-q4_0.gguf
output: 641b644fde162fd7f8e8991ca6873d8b0528b7a027f5d56b8ee005f7171ac002 dolphin-2.2-yi-34b-ggml-model-q4_0.gguf
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-chat.wasm
wasmedge --dir .:. --nn-preload default:GGML:AUTO:dolphin-2.2-yi-34b-ggml-model-q4_0.gguf llama-chat.wasm -p chatml -r '<|im_end|>' -s 'You are a helpful AI assistant'
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/second-state/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:dolphin-2.2-yi-34b-ggml-model-q4_0.gguf llama-api-server.wasm -p chatml -r '<|im_end|>' -s 'You are a helpful AI assistant'
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
curl -LO https://huggingface.co/second-state/dolphin-2.6-mistral-7B-GGUF/resolve/main/dolphin-2.6-mistral-7b.Q5_K_M.gguf
```

Please check the sha256sum of the Downloaded model file to make sure it is correct.

```bash
shasum -a 256 dolphin-2.6-mistral-7b.Q5_K_M.gguf
output: e4ce9eabae27e45131c3d0d99223f133b96257301670073b3aee50f7627e20b2 dolphin-2.6-mistral-7b.Q5_K_M.gguf
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-chat.wasm
wasmedge --dir .:. --nn-preload default:GGML:AUTO:dolphin-2.6-mistral-7b.Q5_K_M.gguf llama-chat.wasm -p chatml
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/second-state/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:dolphin-2.6-mistral-7b.Q5_K_M.gguf llama-api-server.wasm -p chatml
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
<summary> <b>Dolphin-2.2-Mistral-7B</b> </summary>
<hr/>
<b>Download the model</b>

```bash
curl -LO https://huggingface.co/second-state/Dolphin-2.2-Mistral-7B-GGUF/resolve/main/dolphin-2.2-mistral-7b-ggml-model-q4_0.gguf
```

Please check the sha256sum of the Downloaded model file to make sure it is correct.

```bash
shasum -a 256 dolphin-2.2-mistral-7b-ggml-model-q4_0.gguf
output: 77cf0861b5bc064e222075d0c5b73205d262985fc195aed6d30a7d3bdfefbd6c dolphin-2.2-mistral-7b-ggml-model-q4_0.gguf
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-chat.wasm
wasmedge --dir .:. --nn-preload default:GGML:AUTO:dolphin-2.2-mistral-7b-ggml-model-q4_0.gguf llama-chat.wasm -p chatml -r '<|im_end|>'
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/second-state/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:dolphin-2.2-mistral-7b-ggml-model-q4_0.gguf llama-api-server.wasm -p chatml -r '<|im_end|>'
```

Open your browser to http://localhost:8080 to start the chat!

<b>Send an API request to the server</b>

Test the API server from another terminal using the following command

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'accept:application/json' \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "What is the capital of France?"}], "model":"Dolphin-2.2-Mistral-7B"}'
```

</details>

<details>
<summary> <b>Dolphin-2.2.1-Mistral-7B</b> </summary>
<hr/>
<b>Download the model</b>

```bash
curl -LO https://huggingface.co/second-state/Dolphin-2.2.1-Mistral-7B/resolve/main/dolphin-2.2.1-mistral-7b-ggml-model-q4_0.gguf
```

Please check the sha256sum of the Downloaded model file to make sure it is correct.

```bash
shasum -a 256 dolphin-2.2.1-mistral-7b-ggml-model-q4_0.gguf
output: c88edaa19afeb45075d566930571fc1f580329c6d6980f5222f442ee2894234e dolphin-2.2.1-mistral-7b-ggml-model-q4_0.gguf
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-chat.wasm
wasmedge --dir .:. --nn-preload default:GGML:AUTO:dolphin-2.2.1-mistral-7b-ggml-model-q4_0.gguf llama-chat.wasm -p chatml -r '<|im_end|>'
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/second-state/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:dolphin-2.2.1-mistral-7b-ggml-model-q4_0.gguf llama-api-server.wasm -p chatml -r '<|im_end|>'
```

Open your browser to http://localhost:8080 to start the chat!

<b>Send an API request to the server</b>

Test the API server from another terminal using the following command

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'accept:application/json' \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "What is the capital of France?"}], "model":"Dolphin-2.2.1-Mistral-7B"}'
```

</details>

<details>
<summary> <b>Samantha-1.2-Mistral-7B</b> </summary>
<hr/>
<b>Download the model</b>

```bash
curl -LO https://huggingface.co/second-state/Samantha-1.2-Mistral-7B/resolve/main/samantha-1.2-mistral-7b-ggml-model-q4_0.gguf
```

Please check the sha256sum of the Downloaded model file to make sure it is correct.

```bash
shasum -a 256 samantha-1.2-mistral-7b-ggml-model-q4_0.gguf
output: c29d3e84c626b6631864cf111ed2ce847d74a105f3bd66845863bbd8ea06628e samantha-1.2-mistral-7b-ggml-model-q4_0.gguf
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-chat.wasm
wasmedge --dir .:. --nn-preload default:GGML:AUTO:samantha-1.2-mistral-7b-ggml-model-q4_0.gguf llama-chat.wasm -p chatml -r '<|im_end|>'
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/second-state/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:samantha-1.2-mistral-7b-ggml-model-q4_0.gguf llama-api-server.wasm -p chatml -r '<|im_end|>'
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
<summary> <b>Dolphin-2.1-Mistral-7B</b> </summary>
<hr/>
<b>Download the model</b>

```bash
curl -LO https://huggingface.co/second-state/Dolphin-2.1-Mistral-7B-GGUF/resolve/main/dolphin-2.1-mistral-7b-ggml-model-q4_0.gguf
```

Please check the sha256sum of the Downloaded model file to make sure it is correct.

```bash
shasum -a 256 dolphin-2.1-mistral-7b-ggml-model-q4_0.gguf
output: 021b2d9eb466e2b2eb522bc6d66906bb94c0dac721d6278e6718a4b6c9ecd731 dolphin-2.1-mistral-7b-ggml-model-q4_0.gguf
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-chat.wasm
wasmedge --dir .:. --nn-preload default:GGML:AUTO:dolphin-2.1-mistral-7b-ggml-model-q4_0.gguf llama-chat.wasm -p chatml -r '<|im_end|>'
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/second-state/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:dolphin-2.1-mistral-7b-ggml-model-q4_0.gguf llama-api-server.wasm -p chatml -r '<|im_end|>'
```

Open your browser to http://localhost:8080 to start the chat!

<b>Send an API request to the server</b>

Test the API server from another terminal using the following command

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'accept:application/json' \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "What is the capital of France?"}], "model":"Dolphin-2.1-Mistral-7B"}'
```

</details>

<details>
<summary> <b>Dolphin-2.0-Mistral-7B</b> </summary>
<hr/>
<b>Download the model</b>

```bash
curl -LO https://huggingface.co/second-state/Dolphin-2.0-Mistral-7B-GGUF/resolve/main/dolphin-2.0-mistral-7b-ggml-model-q4_0.gguf
```

Please check the sha256sum of the Downloaded model file to make sure it is correct.

```bash
shasum -a 256 dolphin-2.0-mistral-7b-ggml-model-q4_0.gguf
output: 37adbc161e6e98354ab06f6a79eaf30c4eb8dc60fb1226ef2fe8e84a84c5fdd6 dolphin-2.0-mistral-7b-ggml-model-q4_0.gguf
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-chat.wasm
wasmedge --dir .:. --nn-preload default:GGML:AUTO:dolphin-2.0-mistral-7b-ggml-model-q4_0.gguf llama-chat.wasm -p chatml -r '<|im_end|>'
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/second-state/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:dolphin-2.0-mistral-7b-ggml-model-q4_0.gguf llama-api-server.wasm -p chatml -r '<|im_end|>'
```

Open your browser to http://localhost:8080 to start the chat!

<b>Send an API request to the server</b>

Test the API server from another terminal using the following command

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'accept:application/json' \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "What is the capital of France?"}], "model":"Dolphin-2.0-Mistral-7B"}'
```

</details>

<details>
<summary> <b>WizardLM-1.0-Uncensored-CodeLlama-34B</b> </summary>
<hr/>
<b>Download the model</b>

```bash
curl -LO https://huggingface.co/second-state/WizardLM-1.0-Uncensored-CodeLlama-34b/resolve/main/WizardLM-1.0-Uncensored-CodeLlama-34b-ggml-model-q4_0.gguf
```

Please check the sha256sum of the Downloaded model file to make sure it is correct.

```bash
shasum -a 256 WizardLM-1.0-Uncensored-CodeLlama-34b-ggml-model-q4_0.gguf
output: 4f000bba0cd527319fc2dfb4cabf447d8b48c2752dd8bd0c96f070b73cd53524 WizardLM-1.0-Uncensored-CodeLlama-34b-ggml-model-q4_0.gguf
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-chat.wasm
wasmedge --dir .:. --nn-preload default:GGML:AUTO:WizardLM-1.0-Uncensored-CodeLlama-34b-ggml-model-q4_0.gguf llama-chat.wasm -p vicuna-chat -s 'You are a helpful AI assistant.'
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/second-state/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:WizardLM-1.0-Uncensored-CodeLlama-34b-ggml-model-q4_0.gguf llama-api-server.wasm -p vicuna-chat -s 'You are a helpful AI assistant.'
```

Open your browser to http://localhost:8080 to start the chat!

<b>Send an API request to the server</b>

Test the API server from another terminal using the following command

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'accept:application/json' \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "What is the capital of France?"}], "model":"WizardLM-1.0-Uncensored-CodeLlama-34b"}'
```

</details>

<details>
<summary> <b>Samantha-1.11-CodeLlama-34B</b> </summary>
<hr/>
<b>Download the model</b>

```bash
curl -LO https://huggingface.co/second-state/Samantha-1.11-CodeLlama-34B-GGUF/resolve/main/Samantha-1.11-CodeLlama-34b-ggml-model-q4_0.gguf
```

Please check the sha256sum of the Downloaded model file to make sure it is correct.

```bash
shasum -a 256 Samantha-1.11-CodeLlama-34b-ggml-model-q4_0.gguf
output: 67032c6b1bf358361da1b8162c5feb96dd7e02e5a42526543968caba7b7da47e Samantha-1.11-CodeLlama-34b-ggml-model-q4_0.gguf
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-chat.wasm
wasmedge --dir .:. --nn-preload default:GGML:AUTO:Samantha-1.11-CodeLlama-34b-ggml-model-q4_0.gguf llama-chat.wasm -p vicuna-chat -s 'You are a helpful AI assistant.'
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/second-state/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:Samantha-1.11-CodeLlama-34b-ggml-model-q4_0.gguf llama-api-server.wasm -p vicuna-chat -s 'You are a helpful AI assistant.'
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
<summary> <b>Samantha-1.11-7B</b> </summary>
<hr/>
<b>Download the model</b>

```bash
curl -LO https://huggingface.co/second-state/Samantha-1.11-7B-GGUF/resolve/main/Samantha-1.11-7b-ggml-model-q4_0.gguf
```

Please check the sha256sum of the Downloaded model file to make sure it is correct.

```bash
shasum -a 256 Samantha-1.11-7b-ggml-model-q4_0.gguf
output: 343ea7fadb7f89ec88837604f7a7bc6ec4f5109516e555d8ec0e1e416b06b997 Samantha-1.11-7b-ggml-model-q4_0.gguf
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-chat.wasm
wasmedge --dir .:. --nn-preload default:GGML:AUTO:Samantha-1.11-7b-ggml-model-q4_0.gguf llama-chat.wasm -p vicuna-chat -s 'You are Samantha, a sentient AI companion.'
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/second-state/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:Samantha-1.11-7b-ggml-model-q4_0.gguf llama-api-server.wasm -p vicuna-chat -s 'You are Samantha, a sentient AI companion.'
```

Open your browser to http://localhost:8080 to start the chat!

<b>Send an API request to the server</b>

Test the API server from another terminal using the following command

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'accept:application/json' \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "What is the capital of France?"}], "model":"Samantha-1.11-7B"}'
```

</details>

<details>
<summary> <b>WizardCoder-Python-7B-V1.0</b> </summary>
<hr/>
<b>Download the model</b>

```bash
curl -LO https://huggingface.co/second-state/WizardCoder-Python-7B-V1.0/resolve/main/WizardCoder-Python-7B-V1.0-ggml-model-q4_0.gguf
```

Please check the sha256sum of the Downloaded model file to make sure it is correct.

```bash
shasum -a 256 WizardCoder-Python-7B-V1.0-ggml-model-q4_0.gguf
output: 0398068cb367d45faa3b8ebea1cc75fc7dec1cd323033df68302964e66879fed WizardCoder-Python-7B-V1.0-ggml-model-q4_0.gguf
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-chat.wasm
wasmedge --dir .:. --nn-preload default:GGML:AUTO:WizardCoder-Python-7B-V1.0-ggml-model-q4_0.gguf llama-chat.wasm -p wizard-coder -s 'Below is an instruction that describes a task. Write a response that appropriately completes the request.'
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/second-state/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:WizardCoder-Python-7B-V1.0-ggml-model-q4_0.gguf llama-api-server.wasm -p wizard-coder
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
curl -LO https://huggingface.co/second-state/Zephyr-7B-Alpha-GGUF/resolve/main/zephyr-7b-alpha.Q5_K_M.gguf
```

Please check the sha256sum of the Downloaded model file to make sure it is correct.

```bash
shasum -a 256 zephyr-7b-alpha.Q5_K_M.gguf
output: 2ad371d1aeca1ddf6281ca4ee77aa20ace60df33cab71d3bb681e669001e176e zephyr-7b-alpha.Q5_K_M.gguf
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-chat.wasm
wasmedge --dir .:. --nn-preload default:GGML:AUTO:zephyr-7b-alpha.Q5_K_M.gguf llama-chat.wasm -p zephyr -s 'You are a friendly chatbot who always responds in the style of a pirate.' -r '</s>'
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/second-state/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:zephyr-7b-alpha.Q5_K_M.gguf llama-api-server.wasm -p zephyr -r '</s>'
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
<summary> <b>WizardLM-7B-V1.0-Uncensored</b> </summary>
<hr/>
<b>Download the model</b>

```bash
curl -LO https://huggingface.co/second-state/WizardLM-7B-V1.0-Uncensored-GGUF/resolve/main/wizardlm-7b-v1.0-uncensored.Q5_K_M.gguf
```

Please check the sha256sum of the Downloaded model file to make sure it is correct.

```bash
shasum -a 256 wizardlm-7b-v1.0-uncensored.Q5_K_M.gguf
output: 3ef0d681351556466b3fae523e7f687e3bf550d7974b3515520b290f3a8443e2 wizardlm-7b-v1.0-uncensored.Q5_K_M.gguf
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-chat.wasm
wasmedge --dir .:. --nn-preload default:GGML:AUTO:wizardlm-7b-v1.0-uncensored.Q5_K_M.gguf llama-chat.wasm -p vicuna-chat -s 'You are a helpful AI assistant.'
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/second-state/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:wizardlm-7b-v1.0-uncensored.Q5_K_M.gguf llama-api-server.wasm -p vicuna-chat
```

Open your browser to http://localhost:8080 to start the chat!

<b>Send an API request to the server</b>

Test the API server from another terminal using the following command

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'accept:application/json' \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "What is the capital of France?"}], "model":"WizardLM-7B"}'
```

</details>

<details>
<summary> <b>WizardLM-13B-V1.0-Uncensored</b> </summary>
<hr/>
<b>Download the model</b>

```bash
curl -LO https://huggingface.co/second-state/WizardLM-13B-V1.0-Uncensored-GGUF/resolve/main/wizardlm-13b-v1.0-uncensored.Q5_K_M.gguf
```

Please check the sha256sum of the Downloaded model file to make sure it is correct.

```bash
shasum -a 256 wizardlm-13b-v1.0-uncensored.Q5_K_M.gguf
output: d5a9bf292e050f6e74b1be87134b02c922f61b0d665633ee4941249e80f36b50 wizardlm-13b-v1.0-uncensored.Q5_K_M.gguf
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-chat.wasm
wasmedge --dir .:. --nn-preload default:GGML:AUTO:wizardlm-13b-v1.0-uncensored.Q5_K_M.gguf llama-chat.wasm -p vicuna-chat -s 'You are a helpful AI assistant.'
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/second-state/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:wizardlm-13b-v1.0-uncensored.Q5_K_M.gguf llama-api-server.wasm -p vicuna-chat
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
curl -LO https://huggingface.co/second-state/Orca-2-13B-GGUF/resolve/main/Orca-2-13b-ggml-model-q4_0.gguf
```

Please check the sha256sum of the Downloaded model file to make sure it is correct.

```bash
shasum -a 256 Orca-2-13b-ggml-model-q4_0.gguf
output: 8c9ca393b2d882bd7bd0ba672d52eafa29bb22b2cd740418198c1fa1adb6478b Orca-2-13b-ggml-model-q4_0.gguf
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-chat.wasm
wasmedge --dir .:. --nn-preload default:GGML:AUTO:Orca-2-13b-ggml-model-q4_0.gguf llama-chat.wasm -p chatml -s 'You are Orca, an AI language model created by Microsoft. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior.'
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/second-state/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:Orca-2-13b-ggml-model-q4_0.gguf llama-api-server.wasm -p chatml
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
curl -LO https://huggingface.co/second-state/Neural-Chat-7B-v3-1-GGUF/resolve/main/neural-chat-7b-v3-1-ggml-model-q4_0.gguf
```

Please check the sha256sum of the Downloaded model file to make sure it is correct.

```bash
shasum -a 256 neural-chat-7b-v3-1-ggml-model-q4_0.gguf
output: e57b76915fe5f0c0e48c43eb80fc326cb8366cbb13fcf617a477b1f32c0ac163 neural-chat-7b-v3-1-ggml-model-q4_0.gguf
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-chat.wasm
wasmedge --dir .:. --nn-preload default:GGML:AUTO:neural-chat-7b-v3-1-ggml-model-q4_0.gguf llama-chat.wasm -p intel-neural
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/second-state/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:neural-chat-7b-v3-1-ggml-model-q4_0.gguf llama-api-server.wasm -p intel-neural
```

Open your browser to http://localhost:8080 to start the chat!

<b>Send an API request to the server</b>

Test the API server from another terminal using the following command

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'accept:application/json' \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "What is the capital of France?"}], "model":"Neural-Chat-7B"}'
```

</details>

<details>
<summary> <b>Yi-34B-Chat</b> </summary>
<hr/>
<b>Download the model</b>

```bash
curl -LO https://huggingface.co/second-state/Yi-34B-Chat-GGUF/resolve/main/Yi-34B-Chat-ggml-model-q4_0.gguf
```

Please check the sha256sum of the Downloaded model file to make sure it is correct.

```bash
shasum -a 256 Yi-34B-Chat-ggml-model-q4_0.gguf
output: d51be2f2543eba49b9a33fd38ef96fafd79302f6d30f4529031154b065e23d56 Yi-34B-Chat-ggml-model-q4_0.gguf
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-chat.wasm
wasmedge --dir .:. --nn-preload default:GGML:AUTO:Yi-34B-Chat-ggml-model-q4_0.gguf llama-chat.wasm -p chatml -r '<|im_end|>'
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/second-state/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz
wasmedge --dir .:. --nn-preload default:GGML:AUTO:Yi-34B-Chat-ggml-model-q4_0.gguf llama-api-server.wasm -p chatml -r '<|im_end|>'
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
curl -LO https://huggingface.co/second-state/Starling-LM-7B-alpha-GGUF/resolve/main/starling-lm-7b-alpha.Q5_K_M.gguf
```

Please check the sha256sum of the Downloaded model file to make sure it is correct.

```bash
shasum -a 256 starling-lm-7b-alpha.Q5_K_M.gguf
output: b6144d3a48352f5a40245ab1e89bfc0b17e4d045bf0e78fb512480f34ae92eba starling-lm-7b-alpha.Q5_K_M.gguf
```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-chat.wasm
wasmedge --dir .:. --nn-preload default:GGML:AUTO:starling-lm-7b-alpha.Q5_K_M.gguf llama-chat.wasm -p openchat -r '<|end_of_turn|>'
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/second-state/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:starling-lm-7b-alpha.Q5_K_M.gguf llama-api-server.wasm -p openchat -r '<|end_of_turn|>'
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
<summary> <b>Calm2-7B-Chat</b> </summary>
<hr/>
<b>Download the model</b>

```bash
curl -LO https://huggingface.co/second-state/Calm2-7B-Chat-GGUF/resolve/main/calm2-7b-chat.Q4_K_M.gguf
```

Note that check the sha256 of `calm2-7b-chat.Q4_K_M.gguf` after downloading.

  ```text
  42e829c19100c5d82c9432f0ee4c062e994fcf03966e8bfb2e92d1d91db12d56
  ```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-chat.wasm
wasmedge --dir .:. --nn-preload default:GGML:AUTO:calm2-7b-chat.Q4_K_M.gguf llama-chat.wasm -p vicuna-1.1-chat
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/second-state/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:calm2-7b-chat.Q4_K_M.gguf llama-api-server.wasm -p vicuna-1.1-chat
```

Open your browser to http://localhost:8080 to start the chat!

<b>Send an API request to the server</b>

Test the API server from another terminal using the following command

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'accept:application/json' \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "二日間の京都旅行計画"}], "model":"Calm2-7B-Chat"}'
```

</details>

<details>
<summary> <b>DeepSeek-Coder-6.7B</b> </summary>
<hr/>
<b>Download the model</b>

```bash
curl -LO https://huggingface.co/second-state/Deepseek-Coder-6.7B-Instruct-GGUF/resolve/main/deepseek-coder-6.7b-instruct.Q5_K_M.gguf
```

  Note that check the sha256 of `deepseek-coder-6.7b-instruct.Q5_K_M.gguf` after downloading.

  ```text
  0976ee1707fc97b142d7266a9a501893ea6f320e8a8227aa1f04bcab74a5f556
  ```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-chat.wasm
wasmedge --dir .:. --nn-preload default:GGML:AUTO:deepseek-coder-6.7b-instruct.Q5_K_M.gguf llama-chat.wasm -p deepseek-coder
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/second-state/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:deepseek-coder-6.7b-instruct.Q5_K_M.gguf llama-api-server.wasm -p deepseek-coder
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
curl -LO https://huggingface.co/second-state/Deepseek-LLM-7B-Chat-GGUF/resolve/main/deepseek-llm-7b-chat.Q5_K_M.gguf
```

  Note that check the sha256 of `deepseek-llm-7b-chat.Q5_K_M.gguf` after downloading.

  ```text
  e5bcd887cc97ff63dbd17b8b9feac261516e985b5e78f1f544eb49cf403caaf6
  ```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-chat.wasm
wasmedge --dir .:. --nn-preload default:GGML:AUTO:deepseek-llm-7b-chat.Q5_K_M.gguf llama-chat.wasm -p deepseek-chat
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/second-state/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:deepseek-llm-7b-chat.Q5_K_M.gguf llama-api-server.wasm -p deepseek-chat
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
curl -LO https://huggingface.co/second-state/SOLAR-10.7B-Instruct-v1.0-GGUF/resolve/main/solar-10.7b-instruct-v1.0.Q5_K_M.gguf
```

  Note that check the sha256 of `solar-10.7b-instruct-v1.0.Q5_K_M.gguf` after downloading.

  ```text
  4ade240f5dcc253272158f3659a56f5b1da8405510707476d23a7df943aa35f7
  ```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-chat.wasm
wasmedge --dir .:. --nn-preload default:GGML:AUTO:solar-10.7b-instruct-v1.0.Q5_K_M.gguf llama-chat.wasm -p solar-instruct
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/second-state/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:solar-10.7b-instruct-v1.0.Q5_K_M.gguf llama-api-server.wasm -p solar-instruct
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
curl -LO https://huggingface.co/second-state/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main/mixtral-8x7b-instruct-v0.1.Q4_0.gguf
```

  Note that check the sha256 of `mixtral-8x7b-instruct-v0.1.Q4_0.gguf` after downloading.

  ```text
  0c57465507f21bed4364fca37efd310bee92e25a4ce4f5678ef9b44e95830e4e
  ```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-chat.wasm
wasmedge --dir .:. --nn-preload default:GGML:AUTO:mixtral-8x7b-instruct-v0.1.Q4_0.gguf llama-chat.wasm -p mistral-instruct
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/second-state/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:mixtral-8x7b-instruct-v0.1.Q4_0.gguf llama-api-server.wasm -p mistral-instruct
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
<summary> <b>Dolphin-2.6-Phi-2</b> </summary>
<hr/>
<b>Download the model</b>

```bash
curl -LO https://huggingface.co/second-state/Dolphin-2.6-Phi-2-GGUF/resolve/main/dolphin-2_6-phi-2.Q5_K_M.gguf
```

  Note that check the sha256 of `dolphin-2_6-phi-2.Q5_K_M.gguf` after downloading.

  ```text
  acc43043793230038f39491de557e70c9d99efddc41f1254e7064cc48f9b5c1e
  ```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-chat.wasm
wasmedge --dir .:. --nn-preload default:GGML:AUTO:dolphin-2_6-phi-2.Q5_K_M.gguf llama-chat.wasm -p chatml
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/second-state/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:dolphin-2_6-phi-2.Q5_K_M.gguf llama-api-server.wasm -p chatml
```

Open your browser to http://localhost:8080 to start the chat!

<b>Send an API request to the server</b>

Test the API server from another terminal using the following command

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'accept:application/json' \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"system", "content": "You are an AI programming assistant."}, {"role":"user", "content": "What is the capital of Paris?"}], "model":"dolphin-2.6-phi-2"}'
```

</details>

<details>
<summary> <b>ELYZA-japanese-Llama-2-7b-instruct</b> </summary>
<hr/>
<b>Download the model</b>

```bash
curl -LO https://huggingface.co/second-state/ELYZA-japanese-Llama-2-7b-instruct-GGUF/resolve/main/ELYZA-japanese-Llama-2-7b-instruct-q5_K_M.gguf
```

  Note that check the sha256 of `ELYZA-japanese-Llama-2-7b-instruct-q5_K_M.gguf` after downloading.

  ```text
  53c0a17b0bba8aedc868e5dce72e5976cd99108966659b8466476957e99dc980
  ```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-chat.wasm
wasmedge --dir .:. --nn-preload default:GGML:AUTO:ELYZA-japanese-Llama-2-7b-instruct-q5_K_M.gguf llama-chat.wasm -p llama-2-chat
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/second-state/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:ELYZA-japanese-Llama-2-7b-instruct-q5_K_M.gguf llama-api-server.wasm -p llama-2-chat
```

Open your browser to http://localhost:8080 to start the chat!

<b>Send an API request to the server</b>

Test the API server from another terminal using the following command

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'accept:application/json' \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"system", "content": "あなたは誠実で優秀な日本人のアシスタントです。"}, {"role":"user", "content": "クマが海辺に行ってアザラシと友達になり、最終的には家に帰るというプロットの短編小説を書いてください。"}], "model":"ELYZA-japanese-Llama-2-7b-instruct"}'
```

</details>

<details>
<summary> <b>ELYZA-japanese-Llama-2-7b-fast-instruct</b> </summary>
<hr/>
<b>Download the model</b>

```bash
curl -LO https://huggingface.co/second-state/ELYZA-japanese-Llama-2-7b-fast-instruct-GGUF/resolve/main/ELYZA-japanese-Llama-2-7b-fast-instruct-q5_K_M.gguf
```

  Note that check the sha256 of `ELYZA-japanese-Llama-2-7b-instruct-q5_K_M.gguf` after downloading.

  ```text
  3dc1e83340c2ee25903ff286da79a62999ab2b2ade7ae2a7c0d6db9f47e14087
  ```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-chat.wasm
wasmedge --dir .:. --nn-preload default:GGML:AUTO:ELYZA-japanese-Llama-2-7b-fast-instruct-q5_K_M.gguf llama-chat.wasm -p llama-2-chat
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/second-state/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:ELYZA-japanese-Llama-2-7b-fast-instruct-q5_K_M.gguf llama-api-server.wasm -p llama-2-chat
```

Open your browser to http://localhost:8080 to start the chat!

<b>Send an API request to the server</b>

Test the API server from another terminal using the following command

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'accept:application/json' \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"system", "content": "あなたは誠実で優秀な日本人のアシスタントです。"}, {"role":"user", "content": "クマが海辺に行ってアザラシと友達になり、最終的には家に帰るというプロットの短編小説を書いてください。"}], "model":"ELYZA-japanese-Llama-2-7b-fast-instruct"}'
```

</details>

<details>
<summary> <b>Nous-Hermes-2-Mixtral-8x7B-DPO</b> </summary>
<hr/>
<b>Download the model</b>

```bash
curl -LO https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO-GGUF/blob/main/Nous-Hermes-2-Mixtral-8x7B-DPO.Q5_K_M.gguf
```

  Note that check the sha256 of `Nous-Hermes-2-Mixtral-8x7B-DPO.Q5_K_M.gguf` after downloading.

  ```text
  e09228d3f9d764c4dd4d57bd83edbfb3548a5477c990246f6648ee382f11ee8b
  ```

<b>Chat with the model on the CLI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-chat.wasm
wasmedge --dir .:. --nn-preload default:GGML:AUTO:Nous-Hermes-2-Mixtral-8x7B-DPO.Q5_K_M.gguf llama-chat.wasm -p chatml
```

<b>Chat with the model via a web UI</b>

```bash
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-api-server.wasm
curl -LO https://github.com/second-state/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

wasmedge --dir .:. --nn-preload default:GGML:AUTO:Nous-Hermes-2-Mixtral-8x7B-DPO.Q5_K_M.gguf llama-api-server.wasm -p chatml
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
