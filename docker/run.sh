#!/bin/bash

source /root/.wasmedge/env

chat_file_name=$1
embedding_file_name=$2
prompt_template=$3
chat_ctx_size=$4
embedding_ctx_size=$5

if [ -z "$chat_ctx_size" ]; then
    chat_ctx_size=512
fi

if [ -z "$embedding_ctx_size" ]; then
    embedding_ctx_size=256
fi

if [ "$embedding_ctx_size" -eq "0" ]; then
    wasmedge --dir .:. --nn-preload default:GGML:AUTO:/models/$chat_file_name llama-api-server.wasm --prompt-template $prompt_template --ctx-size $chat_ctx_size --model-name $chat_file_name --socket-addr 0.0.0.0:8080
else
    wasmedge --dir .:. --nn-preload default:GGML:AUTO:/models/$chat_file_name --nn-preload embedding:GGML:AUTO:/models/$embedding_file_name llama-api-server.wasm --prompt-template $prompt_template,embedding --ctx-size $chat_ctx_size,$embedding_ctx_size --model-name $chat_file_name,$embedding_file_name --socket-addr 0.0.0.0:8080
fi
