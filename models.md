# GGUF Models

You can find the model download link, the command to run the model, the command to create an OpenAI compatible API server for the model, and the sha256sum of the model.

- [Llama-2-7B-Chat](#llama-2-7b-chat)
- [Llama-2-13B-Chat](#llama-2-13b-chat)
- [CodeLlama-13B-Instruct](#codellama-13b-instruct)
- [BELLE-Llama2-13B-Chat](#belle-llama2-13b-chat)
- [Mistral-7B-Instruct-v0.1](#mistral-7b-instruct-v01)
- [MistralLite-7B](#mistrallite-7b)
- [OpenChat-3.5](#openchat-35)
- [Wizard-Vicuna](#wizard-vicuna)
- [CausalLM-14B](#causallm-14b)
- [TinyLlama-1.1B-Chat-v0.3](#tinyllama-11b-chat-v03)
- [Baichuan2-13B-Chat](#baichuan2-13b-chat)
- [Baichuan2-7B-Chat](#baichuan2-7b-chat)
- [OpenHermes-2.5-Mistral-7B](#openhermes-25-mistral-7b)
- [Dolphin-2.2-Yi-34B](#dolphin-22-yi-34b)
- [Dolphin-2.2-Mistral-7B](#dolphin-22-mistral-7b)
- [Dolphin-2.2.1-Mistral-7B](#dolphin-221-mistral-7b)
- [Samantha-1.2-Mistral-7B](#samantha-12-mistral-7b)
- [Dolphin-2.1-Mistral-7B](#dolphin-21-mistral-7b)
- [Dolphin-2.0-Mistral-7B](#dolphin-20-mistral-7b)
- [WizardLM-1.0-Uncensored-CodeLlama-34B](#wizardlm-10-uncensored-codellama-34b)
- [Samantha-1.11-CodeLlama-34B](#samantha-111-codellama-34b)
- [Samantha-1.11-7B](#samantha-111-7b)
- [WizardCoder-Python-7B-V1.0](#wizardcoder-python-7b-v10)
- [Zephyr-7B-Alpha](#zephyr-7b-alpha)
- [WizardLM-7B-V1.0-Uncensored](#wizardlm-7b-v10-uncensored)
- [WizardLM-13B-V1.0-Uncensored](#wizardlm-13b-v10-uncensored)
- [Orca-2-13B](#orca-2-13b)
- [Neural-Chat-7B-v3-1](#neural-chat-7b-v3-1)
- [Yi-34B-Chat](#yi-34b-chat)
- [Starling-LM-7B-alpha](#starling-lm-7b-alpha)


## Llama-2-7B-Chat

### Download the Llama-2-7B-Chat model

```console
curl -LO https://huggingface.co/second-state/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q5_K_M.gguf
```
### Command to run the model

```console
wasmedge --dir .:. --nn-preload default:GGML:AUTO:llama-2-7b-chat.Q5_K_M.gguf llama-chat.wasm -p llama-2-chat
```
### Command to create the API server for the Llama-2-7B-Chat model

Run the follwing command to create the API server

```
wasmedge --dir .:. --nn-preload default:GGML:AUTO:llama-2-7b-chat.Q5_K_M.gguf llama-api-server.wasm -p llama-2-chat
```

Test the API server from anothe terminal using the following command

```
curl -X POST http://localhost:8080/v1/chat/completions -H 'accept:application/json' -H 'Content-Type: application/json' -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "Who is Robert Oppenheimer?"}], "model":"llama-2-7b-chat"}'
```

### Get the sha256sum of the model

Please check the sha256sum of the Downloaded model file to make sure it is correct:

```
shasum -a 256 llama-2-7b-chat.Q5_K_M.gguf
output: e0b99920cf47b94c78d2fb06a1eceb9ed795176dfa3f7feac64629f1b52b997f llama-2-7b-chat.Q5_K_M.gguf
```

## Llama-2-13B-Chat

### Download the Llama-2-13B-Chat model

```
curl -LO https://huggingface.co/second-state/Llama-2-13B-Chat-GGUF/resolve/main/llama-2-13b-chat.Q5_K_M.gguf
```

### Command to run the model
```
wasmedge --dir .:. --nn-preload default:GGML:AUTO:llama-2-13b-chat.Q5_K_M.gguf llama-chat.wasm -p llama-2-chat
```
### Command to create the API server for the model

Run the follwing command to create the API server

```
wasmedge --dir .:. --nn-preload default:GGML:AUTO:llama-2-13b-chat.Q5_K_M.gguf llama-api-server.wasm -p llama-2-chat
```

Test the API server from anothe terminal using the following command

```
curl -X POST http://localhost:8080/v1/chat/completions -H 'accept:application/json' -H 'Content-Type: application/json' -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "Who is Robert Oppenheimer?"}], "model":"llama-2-13b-chat"}'
```

### Get the sha256sum of the model

Please check the sha256sum of the Downloaded model file to make sure it is correct:

```
shasum -a 256 llama-2-13b-chat.Q5_K_M.gguf
output: ef36e090240040f97325758c1ad8e23f3801466a8eece3a9eac2d22d942f548a llama-2-13b-chat.Q5_K_M.gguf
```

## CodeLlama-13B-Instruct

### Download the odeLlama-13B-Instruct model

```console
curl -LO curl -LO https://huggingface.co/second-state/CodeLlama-13B-Instruct-GGUF/resolve/main/codellama-13b-instruct.Q4_0.gguf
```
### Command to run the model
  
```console
wasmedge --dir .:. --nn-preload default:GGML:AUTO:codellama-13b-instruct.Q4_0.gguf llama-chat.wasm -p codellama-instruct
```

### Command to create the API server for the model

This model isn't suitable for creating a API server

### Get the sha256sum of the model

Please check the sha256sum of the Downloaded model file to make sure it is correct:

```
shasum -a 256 codellama-13b-instruct.Q4_0.gguf
693021fa3a170a348b0a6104ab7d3a8c523331826a944dc0371fecd922df89dd codellama-13b-instruct.Q4_0.gguf
```

## BELLE-Llama2-13B-Chat

### Download the BELLE-Llama2-13B-Chat-0.4M model

```console
curl -LO https://huggingface.co/second-state/BELLE-Llama2-13B-Chat-0.4M-GGUF/resolve/main/BELLE-Llama2-13B-Chat-0.4M-ggml-model-q4_0.gguf
```
### Command to run the model

```console
wasmedge --dir .:. --nn-preload default:GGML:AUTO:BELLE-Llama2-13B-Chat-0.4M-ggml-model-q4_0.gguf llama-chat.wasm -p belle-llama-2-chat
```
### Command to create the API server for the model

Run the follwing command to create the API server.

```
wasmedge --dir .:. --nn-preload default:GGML:AUTO:BELLE-Llama2-13B-Chat-0.4M-ggml-model-q4_0.gguf llama-api-server.wasm -p belle-llama-2-chat
```

Test the API server from anothe terminal using the following command.

```
curl -X POST http://localhost:8080/v1/chat/completions -H 'accept:application/json' -H 'Content-Type: application/json' -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "Who is Robert Oppenheimer?"}], "model":"BELLE-Llama2-13B-Chat"}'
```
### Get the sha256sum of the model

Please check the sha256sum of the Downloaded model file to make sure it is correct:

```
shasum -a 256 BELLE-Llama2-13B-Chat-0.4M-ggml-model-q4_0.gguf
output: 56879e1fd6ee6a138286730e121f2dba1be51b8f7e261514a594dea89ef32fe7 BELLE-Llama2-13B-Chat-0.4M-ggml-model-q4_0.gguf
```

## Mistral-7B-Instruct-v0.1

### Download the Mistral-7B-Instruct-v0.1 model

```console
curl -LO https://huggingface.co/second-state/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q5_K_M.gguf
```
### Command to run the model

```console
wasmedge --dir .:. --nn-preload default:GGML:AUTO:mistral-7b-instruct-v0.1.Q5_K_M.gguf llama-chat.wasm -p mistral-instruct-v0.1
```
### Command to create the API server for the model

This model isn't suitable for creating a API server.

### Get the sha256sum of the model

Please check the sha256sum of the Downloaded model file to make sure it is correct:

```text
shasum -a 256 mistral-7b-instruct-v0.1.Q5_K_M.gguf
output: c4b062ec7f0f160e848a0e34c4e291b9e39b3fc60df5b201c038e7064dbbdcdc mistral-7b-instruct-v0.1.Q5_K_M.gguf

shasum -a 256 mistral-7b-instruct-v0.1.Q4_K_M.gguf
output: 14466f9d658bf4a79f96c3f3f22759707c291cac4e62fea625e80c7d32169991 mistral-7b-instruct-v0.1.Q4_K_M.gguf
  ```

## MistralLite-7B

### Download the MistralLite-7B model

```console
curl -LO https://huggingface.co/second-state/MistralLite-7B-GGUF/resolve/main/mistrallite.Q5_K_M.gguf
```
### Command to run the model

```console
wasmedge --dir .:. --nn-preload default:GGML:AUTO:mistrallite.Q5_K_M.gguf llama-chat.wasm -p mistrallite -r '</s>'
```
### Command to create the API server for the model

Run the follwing command to create the API server.

```
wasmedge --dir .:. --nn-preload default:GGML:AUTO:mistrallite.Q5_K_M.gguf llama-api-server.wasm -p mistrallite 
```

Test the API server from anothe terminal using the following command.

```
curl -X POST http://localhost:8080/v1/chat/completions -H 'accept:application/json' -H 'Content-Type: application/json' -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "Who is Robert Oppenheimer?"}], "model":"MistralLite-7B"}'
```
### Get the sha256sum of the model

Please check the sha256sum of the Downloaded model file to make sure it is correct:

```
shasum -a 256 mistrallite.Q5_K_M.gguf
output: d06d149c24eea0446ea7aad596aca396fe7f3302441e9375d5bbd3fd9ba8ebea mistrallite.Q5_K_M.gguf
```

## OpenChat-3.5

### Download the OpenChat-3.5 model

```console
curl -LO https://huggingface.co/second-state/OpenChat-3.5-GGUF/resolve/main/openchat_3.5.Q5_K_M.gguf
```

### Command to run the model

```console
wasmedge --dir .:. --nn-preload default:GGML:AUTO:openchat_3.5.Q5_K_M.gguf llama-chat.wasm -p openchat -r '<|end_of_turn|>'
```

### Command to create the API server for the model

Run the follwing command to create the API server.

```
wasmedge --dir .:. --nn-preload default:GGML:AUTO:openchat_3.5.Q5_K_M.gguf llama-api-server.wasm -p openchat -r '<|end_of_turn|>'
```

Test the API server from anothe terminal using the following command.

```
curl -X POST http://localhost:8080/v1/chat/completions -H 'accept:application/json' -H 'Content-Type: application/json' -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "Who is Robert Oppenheimer?"}], "model":"OpenChat-3.5"}'
```

### Get the sha256sum of the model

Please check the sha256sum of the Downloaded model file to make sure it is correct:

```
shasum -a 256 openchat_3.5.Q5_K_M.gguf
output: 3abf26b0f2ff11394351a23f8d538a1404a2afb69465a6bbaba8836fef51899d openchat_3.5.Q5_K_M.gguf
```

## Wizard-Vicuna

### Download the Wizard-Vicuna model

```console
curl -LO https://huggingface.co/second-state/wizard-vicuna-13B-GGUF/resolve/main/wizard-vicuna-13b-ggml-model-q8_0.gguf
```

### Command to run the model

```console
wasmedge --dir .:. --nn-preload default:GGML:AUTO:wizard-vicuna-13b-ggml-model-q8_0.gguf llama-chat.wasm -p vicuna-chat
```
### Command to create the API server for the model

Run the follwing command to create the API server.

```
wasmedge --dir .:. --nn-preload default:GGML:AUTO:wizard-vicuna-13b-ggml-model-q8_0.gguf llama-api-server.wasm -p vicuna-chat
```

Test the API server from anothe terminal using the following command.

```
curl -X POST http://localhost:8080/v1/chat/completions -H 'accept:application/json' -H 'Content-Type: application/json' -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "Who is Robert Oppenheimer?"}], "model":"wizard-vicuna-13B"}'
```
### Get the sha256sum of the model

Please check the sha256sum of the Downloaded model file to make sure it is correct:

```
shasum -a 256 wizard-vicuna-13b-ggml-model-q8_0.gguf
output: 681b6571e624fd211ae81308b573f24f0016f6352252ae98241b44983bb7e756 wizard-vicuna-13b-ggml-model-q8_0.gguf
```

## CausalLM-14B

### Download the CausalLM-14B model

```console
curl -LO https://huggingface.co/second-state/CausalLM-14B-GGUF/resolve/main/causallm_14b.Q5_1.gguf
```
### Command to run the model

```console
wasmedge --dir .:. --nn-preload default:GGML:AUTO:causallm_14b.Q5_1.gguf llama-chat.wasm -p chatml
```
### Command to create the API server for the model

Run the follwing command to create the API server.

```
wasmedge --dir .:. --nn-preload default:GGML:AUTO:causallm_14b.Q5_1.gguf llama-api-server.wasm -p chatml
```

Test the API server from anothe terminal using the following command.

```
curl -X POST http://localhost:8080/v1/chat/completions -H 'accept:application/json' -H 'Content-Type: application/json' -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "Who is Robert Oppenheimer?"}], "model":"CausalLM-14B"}'
```

### Get the sha256sum of the model

Please check the sha256sum of the Downloaded model file to make sure it is correct:

```
shasum -a 256 causallm_14b.Q5_1.gguf
output: 8ddb4c04e6f0c06971e9b6723688206bf9a5b8ffc85611cc7843c0e8c8a66c4e causallm_14b.Q5_1.gguf
```

## TinyLlama-1.1B-Chat-v0.3

### Download the TinyLlama-1.1B-Chat-v0.3 model

```
curl -LO https://huggingface.co/second-state/TinyLlama-1.1B-Chat-v0.3-GGUF/resolve/main/tinyllama-1.1b-chat-v0.3.Q5_K_M.gguf
```
### Command to run the model

```console
wasmedge --dir .:. --nn-preload default:GGML:AUTO:tinyllama-1.1b-chat-v0.3.Q5_K_M.gguf llama-chat.wasm -p chatml
```
### Command to create the API server for the model

Run the follwing command to create the API server.

```
wasmedge --dir .:. --nn-preload default:GGML:AUTO:tinyllama-1.1b-chat-v0.3.Q5_K_M.gguf llama-api-server.wasm -p chatml
```

Test the API server from anothe terminal using the following command.

```
curl -X POST http://localhost:8080/v1/chat/completions -H 'accept:application/json' -H 'Content-Type: application/json' -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "Who is Robert Oppenheimer?"}], "model":"TinyLlama-1.1B-Chat"}'
```

### Get the sha256sum of the model

Please check the sha256sum of the Downloaded model file to make sure it is correct:

```
shasum -a 256 tinyllama-1.1b-chat-v0.3.Q5_K_M.gguf
output: 7c255febbf29c97b5d6f57cdf62db2f2bc95c0e541dc72c0ca29786ca0fa5eed
```

## Baichuan2-13B-Chat

### Download the Baichuan2-13B-Chat model

```console
curl -LO https://huggingface.co/second-state/Baichuan2-13B-Chat-GGUF/resolve/main/Baichuan2-13B-Chat-ggml-model-q4_0.gguf
```
### Command to run the model

```console
wasmedge --dir .:. --nn-preload default:GGML:AUTO:Baichuan2-13B-Chat-ggml-model-q4_0.gguf llama-chat.wasm -p baichuan-2 -r '用户:'
```

### Command to create the API server for the model

Run the follwing command to create the API server.

```
wasmedge --dir .:. --nn-preload default:GGML:AUTO:Baichuan2-13B-Chat-ggml-model-q4_0.gguf llama-api-server.wasm -p baichuan-2 -r '用户:'
```

Test the API server from anothe terminal using the following command.

```
curl -X POST http://localhost:8080/v1/chat/completions -H 'accept:application/json' -H 'Content-Type: application/json' -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "李白是谁"}], "model":"Baichuan2-13B-Chat"}'
```

### Get the sha256sum of the model

Please check the sha256sum of the Downloaded model file to make sure it is correct:

```
shasum -a 256 Baichuan2-13B-Chat-ggml-model-q4_0.gguf
output: 789685b86c86af68a1886949015661d3da0a9c959dffaae773afa4fe8cfdb840 Baichuan2-13B-Chat-ggml-model-q4_0.gguf
```

## Baichuan2-7B-Chat

### Download the Baichuan2-7B-Chat model

```console
curl -LO https://huggingface.co/second-state/Baichuan2-7B-Chat-GGUF/resolve/main/Baichuan2-7B-Chat-ggml-model-q4_0.gguf
```
### Command to run the model
  
```console
wasmedge --dir .:. --nn-preload default:GGML:AUTO:Baichuan2-7B-Chat-ggml-model-q4_0.gguf llama-chat.wasm -p baichuan-2 -r '用户:'
```

### Command to create the API server for the model

Run the follwing command to create the API server.

```
wasmedge --dir .:. --nn-preload default:GGML:AUTO:Baichuan2-7B-Chat-ggml-model-q4_0.gguf llama-api-server.wasm -p baichuan-2 -r '用户:'
```

Test the API server from anothe terminal using the following command.

```
curl -X POST http://localhost:8080/v1/chat/completions -H 'accept:application/json' -H 'Content-Type: application/json' -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "李白是谁?"}], "model":"Baichuan2-7B-Chat"}'
```

### Get the sha256sum of the model

Please check the sha256sum of the Downloaded model file to make sure it is correct:

```
shasum -a 256 Baichuan2-7B-Chat-ggml-model-q4_0.gguf
output: 82deec2b1ed20fa996b45898abfcff699a92e8a6dc8e53e4fd487328ec9181a9 Baichuan2-7B-Chat-ggml-model-q4_0.gguf
```

## OpenHermes-2.5-Mistral-7B

### Download OpenHermes-2.5-Mistral-7B Model

```console
curl -LO https://huggingface.co/second-state/OpenHermes-2.5-Mistral-7B-GGUF/resolve/main/openhermes-2.5-mistral-7b.Q5_K_M.gguf
```

#### Command to run the Model

After Downloading the model, you can run it with the following command:

```console
wasmedge --dir .:. --nn-preload default:GGML:AUTO:openhermes-2.5-mistral-7b.Q5_K_M.gguf llama-chat.wasm -p chatml -r '<|im_end|>'
```
### Command to create the API server for the model

Run the follwing command to create the API server.

```
wasmedge --dir .:. --nn-preload default:GGML:AUTO:openhermes-2.5-mistral-7b.Q5_K_M.gguf llama-api-server.wasm -p chatml -r '<|im_end|>'
```

Test the API server from anothe terminal using the following command.

```
curl -X POST http://localhost:8080/v1/chat/completions -H 'accept:application/json' -H 'Content-Type: application/json' -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "Who is Robert Oppenheimer?"}], "model":"OpenHermes-2.5-Mistral-7B"}'
```

#### Get the sha256sum of the model

Please check the sha256sum of the Downloaded model file to make sure it is correct:

```
shasum -a 256 openhermes-2.5-mistral-7b.Q5_K_M.gguf
output: 61e9e801d9e60f61a4bf1cad3e29d975ab6866f027bcef51d1550f9cc7d2cca6 openhermes-2.5-mistral-7b.Q5_K_M.gguf
```

## Dolphin-2.2-Yi-34B

### Download Dolphin-2.2-Yi-34B Model

```console
curl -LO https://huggingface.co/second-state/Dolphin-2.2-Yi-34B-GGUF/resolve/main/dolphin-2.2-yi-34b-ggml-model-q4_0.gguf
```

### Command to run the Model

After Downloading the model, you can run it with the following command:

```console
wasmedge --dir .:. --nn-preload default:GGML:AUTO:dolphin-2.2-yi-34b-ggml-model-q4_0.gguf llama-chat.wasm -p chatml -r '<|im_end|>' -s 'You are a helpful AI assistant'
```

### Command to create the API server for the model

Run the following command to create the API server.

```
wasmedge --dir .:. --nn-preload default:GGML:AUTO:dolphin-2.2-yi-34b-ggml-model-q4_0.gguf llama-api-server.wasm -p chatml
```

Test the API server from another terminal using the following command.

```
curl -X POST http://localhost:8080/v1/chat/completions -H 'accept:application/json' -H 'Content-Type: application/json' -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "Who is Robert Oppenheimer?"}], "model":"Dolphin-2.2-Yi-34B"}'
```

### Get the sha256sum of the model

Please check the sha256sum of the Downloaded model file to make sure it is correct:

```
shasum -a 256 dolphin-2.2-yi-34b-ggml-model-q4_0.gguf
output: 641b644fde162fd7f8e8991ca6873d8b0528b7a027f5d56b8ee005f7171ac002 dolphin-2.2-yi-34b-ggml-model-q4_0.gguf
```


## Dolphin-2.2-Mistral-7B

### Download Dolphin-2.2-Mistral-7B Model

```console
curl -LO https://huggingface.co/second-state/Dolphin-2.2-Mistral-7B-GGUF/resolve/main/dolphin-2.2-mistral-7b-ggml-model-q4_0.gguf
```
### Command to run the Model

After Downloading the model, you can run it with the following command:

```console
wasmedge --dir .:. --nn-preload default:GGML:AUTO:dolphin-2.2-mistral-7b-ggml-model-q4_0.gguf llama-chat.wasm -p chatml -r '<|im_end|>'
```

### Command to create the API server for the model

Run the following command to create the API server.

```
wasmedge --dir .:. --nn-preload default:GGML:AUTO:dolphin-2.2-mistral-7b-ggml-model-q4_0.gguf llama-api-server.wasm -p chatml
```

Test the API server from another terminal using the following command.

```
curl -X POST http://localhost:8080/v1/chat/completions -H 'accept:application/json' -H 'Content-Type: application/json' -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "Who is Robert Oppenheimer?"}], "model":"Dolphin-2.2-Mistral-7B"}'
```
### Get the sha256sum of the model

Please check the sha256sum of the Downloaded model file to make sure it is correct:

```
shasum -a 256 dolphin-2.2-mistral-7b-ggml-model-q4_0.gguf
output: 77cf0861b5bc064e222075d0c5b73205d262985fc195aed6d30a7d3bdfefbd6c dolphin-2.2-mistral-7b-ggml-model-q4_0.gguf
```

## Dolphin-2.2.1-Mistral-7B

### Download Dolphin-2.2.1-Mistral-7B Model

```console
curl -LO https://huggingface.co/second-state/Dolphin-2.2.1-Mistral-7B-GGUF/resolve/main/dolphin-2.2.1-mistral-7b-ggml-model-q4_0.gguf
```
### Command to run the Model

After Downloading the model, you can run it with the following command:

```console
wasmedge --dir .:. --nn-preload default:GGML:AUTO:dolphin-2.2.1-mistral-7b-ggml-model-q4_0.gguf llama-chat.wasm -p chatml -r '<|im_end|>'
```

### Command to create the API server for the model

Run the following command to create the API server.

```
wasmedge --dir .:. --nn-preload default:GGML:AUTO:dolphin-2.2.1-mistral-7b-ggml-model-q4_0.gguf llama-api-server.wasm -p chatml
```

Test the API server from another terminal using the following command.

```
curl -X POST http://localhost:8080/v1/chat/completions -H 'accept:application/json' -H 'Content-Type: application/json' -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "Who is Robert Oppenheimer?"}], "model":"Dolphin-2.2.1-Mistral-7B"}'
```
### Get the sha256sum of the model

Please check the sha256sum of the Downloaded model file to make sure it is correct:

```
shasum -a 256 dolphin-2.2.1-mistral-7b-ggml-model-q4_0.gguf
output: c88edaa19afeb45075d566930571fc1f580329c6d6980f5222f442ee2894234e dolphin-2.2.1-mistral-7b-ggml-model-q4_0.gguf
```

## Samantha-1.2-Mistral-7B

### Download Samantha-1.2-Mistral-7B Model

```console
curl -LO https://huggingface.co/second-state/Samantha-1.2-Mistral-7B/resolve/main/samantha-1.2-mistral-7b-ggml-model-q4_0.gguf
```
### Command to run the Model

After Downloading the model, you can run it with the following command:

```console
wasmedge --dir .:. --nn-preload default:GGML:AUTO:samantha-1.2-mistral-7b-ggml-model-q4_0.gguf llama-chat.wasm -p chatml -r '<|im_end|>'
```
### Command to create the API server for the model

Run the following command to create the API server.

```
wasmedge --dir .:. --nn-preload default:GGML:AUTO:samantha-1.2-mistral-7b-ggml-model-q4_0.ggu llama-api-server.wasm -p chatml
```

Test the API server from another terminal using the following command.

```
curl -X POST http://localhost:8080/v1/chat/completions -H 'accept:application/json' -H 'Content-Type: application/json' -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "Who is Robert Oppenheimer?"}], "model":"Samantha-1.2-Mistral-7B"}'
```
### Get the sha256sum of the model

Please check the sha256sum of the Downloaded model file to make sure it is correct:

```
shasum -a 256 samantha-1.2-mistral-7b-ggml-model-q4_0.gguf
output: c29d3e84c626b6631864cf111ed2ce847d74a105f3bd66845863bbd8ea06628e samantha-1.2-mistral-7b-ggml-model-q4_0.gguf
  ```

## Dolphin-2.1-Mistral-7B

### Download Dolphin-2.1-Mistral-7B Model

```console
curl -LO https://huggingface.co/second-state/Dolphin-2.1-Mistral-7B-GGUF/resolve/main/dolphin-2.1-mistral-7b-ggml-model-q4_0.gguf
```
### Command to run the Model

After Downloading the model, you can run it with the following command:

```console
wasmedge --dir .:. --nn-preload default:GGML:AUTO:dolphin-2.1-mistral-7b-ggml-model-q4_0.gguf llama-chat.wasm -p chatml -r '<|im_end|>'
```

### Command to create the API server for the model

Run the following command to create the API server.

```
wasmedge --dir .:. --nn-preload default:GGML:AUTO:dolphin-2.1-mistral-7b-ggml-model-q4_0.gguf llama-api-server.wasm -p chatml
```

Test the API server from another terminal using the following command.

```
curl -X POST http://localhost:8080/v1/chat/completions -H 'accept:application/json' -H 'Content-Type: application/json' -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "Who is Robert Oppenheimer?"}], "model":"Dolphin-2.1-Mistral-7B"}'
```
### Get the sha256sum of the model

Please check the sha256sum of the Downloaded model file to make sure it is correct:

```
shasum -a 256 dolphin-2.1-mistral-7b-ggml-model-q4_0.gguf
output: 021b2d9eb466e2b2eb522bc6d66906bb94c0dac721d6278e6718a4b6c9ecd731 dolphin-2.1-mistral-7b-ggml-model-q4_0.gguf
```

## Dolphin-2.0-Mistral-7B

### Download Dolphin-2.0-Mistral-7B Model

```console
curl -LO https://huggingface.co/second-state/Dolphin-2.0-Mistral-7B-GGUF/resolve/main/dolphin-2.0-mistral-7b-ggml-model-q4_0.gguf
```
### Command to run the Model

After Downloading the model, you can run it with the following command:

```console
wasmedge --dir .:. --nn-preload default:GGML:AUTO:dolphin-2.0-mistral-7b-ggml-model-q4_0.gguf llama-chat.wasm -p chatml -r '<|im_end|>'
```

### Command to create the API server for the model

Run the following command to create the API server.

```
wasmedge --dir .:. --nn-preload default:GGML:AUTO:dolphin-2.0-mistral-7b-ggml-model-q4_0.gguf llama-api-server.wasm -p chatml
```

Test the API server from another terminal using the following command.

```
curl -X POST http://localhost:8080/v1/chat/completions -H 'accept:application/json' -H 'Content-Type: application/json' -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "Who is Robert Oppenheimer?"}], "model":"Dolphin-2.0-Mistral-7B"}'
```
### Get the sha256sum of the model

Please check the sha256sum of the Downloaded model file to make sure it is correct:

```
shasum -a 256 dolphin-2.0-mistral-7b-ggml-model-q4_0.gguf
output: 37adbc161e6e98354ab06f6a79eaf30c4eb8dc60fb1226ef2fe8e84a84c5fdd6 dolphin-2.0-mistral-7b-ggml-model-q4_0.gguf
```

## WizardLM-1.0-Uncensored-CodeLlama-34B

### Download WizardLM-1.0-Uncensored-CodeLlama-34B Model

```console
curl -LO https://huggingface.co/second-state/WizardLM-1.0-Uncensored-CodeLlama-34b/resolve/main/WizardLM-1.0-Uncensored-CodeLlama-34b-ggml-model-q4_0.gguf
```
### Command to run the Model

```console
wasmedge --dir .:. --nn-preload default:GGML:AUTO:WizardLM-1.0-Uncensored-CodeLlama-34b-ggml-model-q4_0.gguf llama-chat.wasm -p vicuna-chat -s 'You are a helpful AI assistant.'
```
### Command to create the API server for the model

Run the following command to create the API server.

```
wasmedge --dir .:. --nn-preload default:GGML:AUTO:WizardLM-1.0-Uncensored-CodeLlama-34b-ggml-model-q4_0.gguf llama-api-server.wasm -p vicuna-chat 
```

Test the API server from another terminal using the following command.

```
curl -X POST http://localhost:8080/v1/chat/completions -H 'accept:application/json' -H 'Content-Type: application/json' -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "Who is Robert Oppenheimer?"}], "model":"WizardLM-1.0-Uncensored-CodeLlama-34b"}'
```
### Get the sha256sum of the model

Please check the sha256sum of the Downloaded model file to make sure it is correct:

```
shasum -a 256 WizardLM-1.0-Uncensored-CodeLlama-34b-ggml-model-q4_0.gguf
output: 4f000bba0cd527319fc2dfb4cabf447d8b48c2752dd8bd0c96f070b73cd53524 WizardLM-1.0-Uncensored-CodeLlama-34b-ggml-model-q4_0.gguf
```

## Samantha-1.11-CodeLlama-34B


### Download Samantha-1.11-CodeLlama-34B Model

```console
curl -LO https://huggingface.co/second-state/Samantha-1.11-CodeLlama-34B-GGUF/resolve/main/Samantha-1.11-CodeLlama-34b-ggml-model-q4_0.gguf
```

### Command to run the Model

```console
wasmedge --dir .:. --nn-preload default:GGML:AUTO:Samantha-1.11-CodeLlama-34b-ggml-model-q4_0.gguf llama-chat.wasm -p vicuna-chat -s 'You are a helpful AI assistant.' 
```

### Command to create the API server for the model

Run the following command to create the API server.

```
wasmedge --dir .:. --nn-preload default:GGML:AUTO:Samantha-1.11-CodeLlama-34b-ggml-model-q4_0.gguf llama-api-server.wasm -p vicuna-chat
```

Test the API server from another terminal using the following command.

```
curl -X POST http://localhost:8080/v1/chat/completions -H 'accept:application/json' -H 'Content-Type: application/json' -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "Who is Robert Oppenheimer?"}], "model":"Samantha-1.11-CodeLlama-34b"}'
```

### Get the sha256sum of the model

Please check the sha256sum of the Downloaded model file to make sure it is correct:

```
shasum -a 256 Samantha-1.11-CodeLlama-34b-ggml-model-q4_0.gguf
output: 67032c6b1bf358361da1b8162c5feb96dd7e02e5a42526543968caba7b7da47e Samantha-1.11-CodeLlama-34b-ggml-model-q4_0.gguf
```


## Samantha-1.11-7B

### Download Samantha-1.11-7B Model

```console
curl -LO https://huggingface.co/second-state/Samantha-1.11-7B-GGUF/resolve/main/Samantha-1.11-7b-ggml-model-q4_0.gguf
```

### Command to run the Model

After Downloading the model, you can run it with the following command:

```console
wasmedge --dir .:. --nn-preload default:GGML:AUTO:Samantha-1.11-7b-ggml-model-q4_0.gguf llama-chat.wasm -p vicuna-chat -s 'You are Samantha, a sentient AI companion.' 
```

### Command to create the API server for the model

Run the following command to create the API server.

```
wasmedge --dir .:. --nn-preload default:GGML:AUTO:Samantha-1.11-7b-ggml-model-q4_0.gguf llama-api-server.wasm -p vicuna-chat
```

Test the API server from another terminal using the following command.

```
curl -X POST http://localhost:8080/v1/chat/completions -H 'accept:application/json' -H 'Content-Type: application/json' -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "Who is Robert Oppenheimer?"}], "model":"Samantha-1.11-7B"}'
```

### Get the sha256sum of the model

Please check the sha256sum of the Downloaded model file to make sure it is correct:

```
shasum -a 256 Samantha-1.11-7b-ggml-model-q4_0.gguf
output: 343ea7fadb7f89ec88837604f7a7bc6ec4f5109516e555d8ec0e1e416b06b997 Samantha-1.11-7b-ggml-model-q4_0.gguf
```


## WizardCoder-Python-7B-V1.0

### Download WizardCoder-Python-7B-V1.0 Model

```console
curl -LO https://huggingface.co/second-state/WizardCoder-Python-7B-V1.0/resolve/main/WizardCoder-Python-7B-V1.0-ggml-model-q4_0.gguf
```

### Command to run the Model

After Downloading the model, you can run it with the following command:

```console
wasmedge --dir .:. --nn-preload default:GGML:AUTO:WizardCoder-Python-7B-V1.0-ggml-model-q4_0.gguf llama-chat.wasm -p wizard-coder -s 'Below is an instruction that describes a task. Write a response that appropriately completes the request.'
```

### Command to create the API server for the model

Run the following command to create the API server.

```
wasmedge --dir .:. --nn-preload default:GGML:AUTO:WizardCoder-Python-7B-V1.0-ggml-model-q4_0.gguf llama-api-server.wasm -p wizard-coder
```

Test the API server from another terminal using the following command.

```
curl -X POST http://localhost:8080/v1/chat/completions -H 'accept:application/json' -H 'Content-Type: application/json' -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "Who is Robert Oppenheimer?"}], "model":"WizardCoder-Python-7B"}'
```

#### Get the sha256sum of the model

Please check the sha256sum of the Downloaded model file to make sure it is correct:

```
shasum -a 256 WizardCoder-Python-7B-V1.0-ggml-model-q4_0.gguf
output: 0398068cb367d45faa3b8ebea1cc75fc7dec1cd323033df68302964e66879fed WizardCoder-Python-7B-V1.0-ggml-model-q4_0.gguf
```

## Zephyr-7B-Alpha

### Download Zephyr-7B-Alpha Model

```console
curl -LO https://huggingface.co/second-state/Zephyr-7B-Alpha-GGUF/resolve/main/zephyr-7b-alpha.Q5_K_M.gguf
```

### Command to run the Model

After Downloading the model, you can run it with the following command:

```console
wasmedge --dir .:. --nn-preload default:GGML:AUTO:zephyr-7b-alpha.Q5_K_M.gguf llama-chat.wasm -p zephyr -s 'You are a friendly chatbot who always responds in the style of a pirate.' -r '</s>'
```

### Command to create the API server for the model

Run the following command to create the API server.

```
wasmedge --dir .:. --nn-preload default:GGML:AUTO:zephyr-7b-alpha.Q5_K_M.gguf llama-api-server.wasm -p zephyr
```

Test the API server from another terminal using the following command.

```
curl -X POST http://localhost:8080/v1/chat/completions -H 'accept:application/json' -H 'Content-Type: application/json' -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "Who is Robert Oppenheimer?"}], "model":"Zephyr-7B"}'
```

### Get the sha256sum of the model

Please check the sha256sum of the Downloaded model file to make sure it is correct:

```
shasum -a 256 zephyr-7b-alpha.Q5_K_M.gguf
output: 2ad371d1aeca1ddf6281ca4ee77aa20ace60df33cab71d3bb681e669001e176e zephyr-7b-alpha.Q5_K_M.gguf
```

## WizardLM-7B-V1.0-Uncensored

### Download WizardLM-7B-V1.0-Uncensored Model

```console
curl -LO https://huggingface.co/second-state/WizardLM-7B-V1.0-Uncensored-GGUF/resolve/main/wizardlm-7b-v1.0-uncensored.Q5_K_M.gguf
```

#### Command to run the Model

After Downloading the model, you can run it with the following command:

```console
wasmedge --dir .:. --nn-preload default:GGML:AUTO:wizardlm-7b-v1.0-uncensored.Q5_K_M.gguf llama-chat.wasm -p vicuna-chat -s 'You are a helpful AI assistant.' 
```

### Command to create the API server for the model

Run the following command to create the API server.

```
wasmedge --dir .:. --nn-preload default:GGML:AUTO:wizardlm-7b-v1.0-uncensored.Q5_K_M.gguf llama-api-server.wasm -p vicuna-chat
```

Test the API server from another terminal using the following command.

```
curl -X POST http://localhost:8080/v1/chat/completions -H 'accept:application/json' -H 'Content-Type: application/json' -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "Who is Robert Oppenheimer?"}], "model":"WizardLM-7B"}'
```

#### Get the sha256sum of the model

Please check the sha256sum of the Downloaded model file to make sure it is correct:

```
shasum -a 256 wizardlm-7b-v1.0-uncensored.Q5_K_M.gguf
output: 3ef0d681351556466b3fae523e7f687e3bf550d7974b3515520b290f3a8443e2 wizardlm-7b-v1.0-uncensored.Q5_K_M.gguf
```

## WizardLM-13B-V1.0-Uncensored

### Download WizardLM-13B-V1.0-Uncensored Model

```console
curl -LO https://huggingface.co/second-state/WizardLM-13B-V1.0-Uncensored-GGUF/resolve/main/wizardlm-13b-v1.0-uncensored.Q5_K_M.gguf
```

### Command to run the Model

After Downloading the model, you can run it with the following command:

```console
wasmedge --dir .:. --nn-preload default:GGML:AUTO:wizardlm-13b-v1.0-uncensored.Q5_K_M.gguf llama-chat.wasm -p vicuna-chat -s 'You are a helpful AI assistant.' 
```

### Command to create the API server for the model

Run the following command to create the API server.

```
wasmedge --dir .:. --nn-preload default:GGML:AUTO:wizardlm-13b-v1.0-uncensored.Q5_K_M.gguf llama-api-server.wasm -p vicuna-chat
```

Test the API server from another terminal using the following command.

```
curl -X POST http://localhost:8080/v1/chat/completions -H 'accept:application/json' -H 'Content-Type: application/json' -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "Who is Robert Oppenheimer?"}], "model":"WizardLM-13B-V1.0-Uncensored"}'
```

### Get the sha256sum of the model

Please check the sha256sum of the Downloaded model file to make sure it is correct:

```
shasum -a 256 wizardlm-13b-v1.0-uncensored.Q5_K_M.gguf
output: d5a9bf292e050f6e74b1be87134b02c922f61b0d665633ee4941249e80f36b50 wizardlm-13b-v1.0-uncensored.Q5_K_M.gguf
```

## Orca-2-13B

### Download Orca-2-13B Model

```console
curl -LO https://huggingface.co/second-state/Orca-2-13B-GGUF/resolve/main/Orca-2-13b-ggml-model-q4_0.gguf
```

### Command to run the Model

After Downloading the model, you can run it with the following command:

```console
wasmedge --dir .:. --nn-preload default:GGML:AUTO:Orca-2-13b-ggml-model-q4_0.gguf llama-chat.wasm -p chatml -s 'You are Orca, an AI language model created by Microsoft. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior.' --stream-stdout
```

### Command to create the API server for the model

Run the following command to create the API server.

```
wasmedge --dir .:. --nn-preload default:GGML:AUTO:Orca-2-13b-ggml-model-q4_0.gguf llama-api-server.wasm -p chatml
```

Test the API server from another terminal using the following command.

```
curl -X POST http://localhost:8080/v1/chat/completions -H 'accept:application/json' -H 'Content-Type: application/json' -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "Who is Robert Oppenheimer?"}], "model":"Orca-2-13B"}'
```

### Get the sha256sum of the model

Please check the sha256sum of the Downloaded model file to make sure it is correct:

```
shasum -a 256 Orca-2-13b-ggml-model-q4_0.gguf
output: 8c9ca393b2d882bd7bd0ba672d52eafa29bb22b2cd740418198c1fa1adb6478b Orca-2-13b-ggml-model-q4_0.gguf
```

## Neural-Chat-7B-v3-1

### Download Neural-Chat-7B-v3-1 Model

```console
curl -LO https://huggingface.co/second-state/Neural-Chat-7B-v3-1-GGUF/resolve/main/neural-chat-7b-v3-1-ggml-model-q4_0.gguf
```

### Command to run the Model

After Downloading the model, you can run it with the following command:

```console
wasmedge --dir .:. --nn-preload default:GGML:AUTO:neural-chat-7b-v3-1-ggml-model-q4_0.gguf llama-chat.wasm -p intel-neural
```

### Command to create the API server for the model

Run the following command to create the API server.

```
wasmedge --dir .:. --nn-preload default:GGML:AUTO:neural-chat-7b-v3-1-ggml-model-q4_0.gguf llama-api-server.wasm -p intel-neural
```

Test the API server from another terminal using the following command.

```
curl -X POST http://localhost:8080/v1/chat/completions -H 'accept:application/json' -H 'Content-Type: application/json' -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "Who is Robert Oppenheimer?"}], "model":"Neural-Chat-7B"}'
```

### Get the sha256sum of the model

Please check the sha256sum of the Downloaded model file to make sure it is correct:

```
shasum -a 256 neural-chat-7b-v3-1-ggml-model-q4_0.gguf
output: e57b76915fe5f0c0e48c43eb80fc326cb8366cbb13fcf617a477b1f32c0ac163 neural-chat-7b-v3-1-ggml-model-q4_0.gguf
```

## Yi-34B-Chat

### Download Yi-34B-Chat Model

```console
curl -LO https://huggingface.co/second-state/Yi-34B-Chat-GGUF/resolve/main/Yi-34B-Chat-ggml-model-q4_0.gguf
```

### Command to run the Model

After Downloading the model, you can run it with the following command:

```console
wasmedge --dir .:. --nn-preload default:GGML:AUTO:Yi-34B-Chat-ggml-model-q4_0.gguf llama-chat.wasm -p chatml -r '<|im_end|>' --stream-stdout
```

### Command to create the API server for the model

Run the following command to create the API server.

```
wasmedge --dir .:. --nn-preload default:GGML:AUTO:neural-chat-7b-v3-1-ggml-model-q4_0.gguf llama-api-server.wasm -p chatml
```

Test the API server from another terminal using the following command.

```
curl -X POST http://localhost:8080/v1/chat/completions -H 'accept:application/json' -H 'Content-Type: application/json' -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "Who is Robert Oppenheimer?"}], "model":"Neural-Chat-7B"}'
```

#### Get the sha256sum of the model

Please check the sha256sum of the Downloaded model file to make sure it is correct:

```
shasum -a 256 Yi-34B-Chat-ggml-model-q4_0.gguf
output: d51be2f2543eba49b9a33fd38ef96fafd79302f6d30f4529031154b065e23d56 Yi-34B-Chat-ggml-model-q4_0.gguf
```



## Starling-LM-7B-alpha

### Download Starling-LM-7B-alpha Model

```console
curl -LO https://huggingface.co/second-state/Starling-LM-7B-alpha-GGUF/resolve/main/starling-lm-7b-alpha.Q5_K_M.gguf
```

### Command to run the Model

After Downloading the model, you can run it with the following command:

```console
wasmedge --dir .:. --nn-preload default:GGML:AUTO:starling-lm-7b-alpha.Q5_K_M.gguf llama-chat.wasm -p openchat -r '<|end_of_turn|>' --stream-stdout
```

### Command to create the API server for the model

Run the following command to create the API server.

```
wasmedge --dir .:. --nn-preload default:GGML:AUTO:starling-lm-7b-alpha.Q5_K_M.gguf llama-api-server.wasm -p openchat
```

Test the API server from another terminal using the following command.

```
curl -X POST http://localhost:8080/v1/chat/completions -H 'accept:application/json' -H 'Content-Type: application/json' -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "Who is Robert Oppenheimer?"}], "model":"Starling-LM-7B"}'
```

#### Get the sha256sum of the model

Please check the sha256sum of the Downloaded model file to make sure it is correct:

```
shasum -a 256 starling-lm-7b-alpha.Q5_K_M.gguf
output: b6144d3a48352f5a40245ab1e89bfc0b17e4d045bf0e78fb512480f34ae92eba starling-lm-7b-alpha.Q5_K_M.gguf
```


