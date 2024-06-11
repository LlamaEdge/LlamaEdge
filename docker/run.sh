#!/bin/bash

source /root/.wasmedge/env

chat_file_name=$1
prompt_template=$2
chat_ctx_size=$3
embedding_file_name=$4
embedding_ctx_size=$5

if [ -z "$embedding_file_name" ]; then
    wasmedge --dir .:. --nn-preload default:GGML:AUTO:/models/$chat_file_name llama-api-server.wasm --prompt-template $prompt_template --ctx-size $chat_ctx_size --model-name $chat_file_name --socket-addr 0.0.0.0:8080
else
    wasmedge --dir .:. --nn-preload default:GGML:AUTO:/models/$chat_file_name --nn-preload embedding:GGML:AUTO:/models/$embedding_file_name llama-api-server.wasm --prompt-template $prompt_template,embedding --ctx-size $chat_ctx_size,$embedding_ctx_size --model-name $chat_file_name,$embedding_file_name --socket-addr 0.0.0.0:8080
fi
