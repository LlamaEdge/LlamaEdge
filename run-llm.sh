#!/bin/bash

check_os() {
    printf "Checking the operating system (macOS and Linux supported) ...\n"

    # Check if the current operating system is macOS or Linux
    if [[ "$OSTYPE" != "linux-gnu"* && "$OSTYPE" != "darwin"* ]]; then
        echo "The OS should be macOS or Linux"
        exit 1
    fi
}

prereq() {
    printf "Checking prerequisites ...\n"

    # Check if git and curl are installed, if not, install them
    for cmd in git curl
    do
        if ! command -v $cmd &> /dev/null
        then
            printf "'$cmd' is required for installation.\n"
            exit 1
            # if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            #     sudo apt-get install $cmd
            # elif [[ "$OSTYPE" == "darwin"* ]]; then
            #     brew install $cmd
            # fi
        fi
    done

    # Check if libopenblas is installed
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if ! ldconfig -p | grep libopenblas &> /dev/null
        then
            printf "'libopenblas' is required for wasi-nn plugin to run.\n"
            exit 1
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # Todo check libopenblas in MacOS
        printf "" # Placeholder
    fi
}

install_wasmedge() {
    printf "Installing WasmEdge ...\n\n"

    # Check if WasmEdge has been installed
    reinstall_wasmedge=1
    if command -v wasmedge &> /dev/null
    then
        printf "'WasmEdge' has been installed, what do you want:\n\n"
        printf "      1) Reinstall WasmEdge for me\n"
        printf "      2) Keep my own WasmEdge\n"

        printf "\n      Please enter a number from the list above:"
        read reinstall_wasmedge
    fi

    while [[ "$reinstall_wasmedge" -ne 1 && "$reinstall_wasmedge" -ne 2 ]]; do
        printf "      Invalid number. Please enter number 1 or 2\n"
        read reinstall_wasmedge
    done


    if [[ "$reinstall_wasmedge" == "1" ]]; then
        # Run the command to install WasmEdge
        if curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | bash -s -- --plugins wasi_nn-ggml; then
            source $HOME/.wasmedge/env
            wasmedge_path=$(which wasmedge)
            printf "\n      The WasmEdge Runtime is installed in %s.\n\n      * To uninstall it, use the command 'bash <(curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/uninstall.sh) -q'\n" "$wasmedge_path"
        else
            echo "Failed to install WasmEdge"
            exit 1
        fi
    elif [[ "$reinstall_wasmedge" == "2" ]]; then
        printf "      * You need to download wasm_nn-ggml plugin from 'https://github.com/WasmEdge/WasmEdge/releases' and put it under plugins directory."
    fi
}

download_model() {
    printf "Downloading the gguf model ...\n\n"

    models="llama-2-7b-chat https://huggingface.co/second-state/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q5_K_M.gguf llama-2-chat \
    llama-2-13b-chat https://huggingface.co/second-state/Llama-2-13B-Chat-GGUF/resolve/main/llama-2-13b-chat.Q5_K_M.gguf llama-2-chat \
    Mistral-7b https://huggingface.co/second-state/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q5_K_M.gguf mistral-instruct-v0.1 \
    mistrallite https://huggingface.co/second-state/MistralLite-7B-GGUF/resolve/main/mistrallite.Q5_K_M.gguf mistrallite \
    TinyLlma https://huggingface.co/second-state/TinyLlama-1.1B-Chat-v0.3-GGUF/resolve/main/tinyllama-1.1b-chat-v0.3.Q5_K_M.gguf chatml \
    Orca-2-13b https://huggingface.co/second-state/Orca-2-13B-GGUF/resolve/main/Orca-2-13b-ggml-model-q4_0.gguf chatml \
    Codellama-13b https://huggingface.co/second-state/CodeLlama-13B-Instruct-GGUF/resolve/main/codellama-13b-instruct.Q4_0.gguf codellama-instruct \
    Baichuan2-7B https://huggingface.co/second-state/Baichuan2-7B-Chat-GGUF/resolve/main/Baichuan2-7B-Chat-ggml-model-q4_0.gguf baichuan-2 \
    OpenChat-3.5 https://huggingface.co/second-state/OpenChat-3.5-GGUF/resolve/main/openchat_3.5.Q5_K_M.gguf openchat"

    model_names="llama-2-7b-chat llama-2-13b-chat Mistral-7b mistrallite TinyLlma Orca-2-13b Codellama-13b Baichuan2-7B OpenChat-3.5"

    # Convert model_names to an array
    model_names_array=($model_names)

    # Print the models with their corresponding numbers
    for i in "${!model_names_array[@]}"; do
       printf "      %d) %s\n" $((i+1)) "${model_names_array[$i]}"
    done

    printf "\n      Please enter a number from the list above: "
    read model_number

    # Validate the input
    while [[ "$model_number" -lt 1 || "$model_number" -gt ${#model_names_array[@]} ]]; do
        printf "\n      Invalid number. Please enter a number between 1 and %d: " ${#model_names_array[@]}
        read model_number
    done

    # Get the model name from the array
    model=${model_names_array[$((model_number-1))]}

    # Change IFS to newline
    IFS=$'\n'

    # Check if the provided model name exists in the models string
    url=$(printf "%s\n" $models | awk -v model=$model '{for(i=1;i<=NF;i++)if($i==model)print $(i+1)}')

    if [ -z "$url" ]; then
        printf "\n      The URL for downloading the target gguf model does not exist.\n"
        exit 1
    fi

    filename=`basename $url`
    if ls $filename &> /dev/null
    then
        printf "\n      * You picked %s, whose model file already exist, skipping downloading\n" "$model"
    else
        printf "\n      You picked %s, downloading from %s\n" "$model" "$url"
        curl -LO $url -#
    fi

    model_file=$(basename $url)

    # Check if the provided model name exists in the models string
    prompt_template=$(printf "%s\n" $models | awk -v model=$model '{for(i=1;i<=NF;i++)if($i==model)print $(i+2)}')

    if [ -z "$prompt_template" ]; then
        printf "\n      The prompt template for the selected model does not exist.\n"
        exit 1
    fi
}

select_mode() {
    printf "Do you want to run the model via CLI or create an API server for it?\n"
    printf "      1) Run the model via CLI\n"
    printf "      2) Create an API server\n"

    printf "\n      Please enter a number from the list above:"
    read running_mode

    while [[ "$running_mode" -ne 1 && "$running_mode" -ne 2 ]]; do
        printf "      Invalid number. Please enter number 1 or 2\n"
        read running_mode
    done
}

select_log_level() {
    printf "Do you want to show the log info\n"
    printf "      1) Yes\n"
    printf "      2) No\n"

    printf "\n      Please enter a number from the list above:"
    read log_level

    while [[ "$log_level" -ne 1 && "$log_level" -ne 2 ]]; do
        printf "      Invalid number. Please enter number 1 or 2\n"
        read log_level
    done
}

download_server() {
    printf "Downloading 'llama-api-server' wasm app ...\n"

    # wasm_url="https://github.com/second-state/llama-utils/raw/main/api-server/llama-api-server.wasm"
    wasm_url="https://github.com/second-state/llama-utils/raw/all-in-one/llama-api-server.wasm"
    curl -LO $wasm_url -#
}

download_webui_files() {
    printf "Downloading frontend resources of 'chatbot-ui' ...\n"

    files_tarball="https://github.com/second-state/chatbot-ui/releases/download/v0.1.0/chatbot-ui.tar.gz"
    curl -LO $files_tarball -#
    tar xzf chatbot-ui.tar.gz
    rm chatbot-ui.tar.gz
}

start_server() {
    printf "Starting llama-api-server ...\n\n"

    wasmedge --dir .:. --nn-preload default:GGML:AUTO:$model_file llama-api-server.wasm -p $prompt_template
}

download_chat_wasm() {
    printf "Downloading 'llama-chat' wasm ...\n"

    wasm_url="https://github.com/second-state/llama-utils/raw/main/chat/llama-chat.wasm"
    curl -LO $wasm_url -#
}

start_chat() {
    printf "starting llama-chat ... \n\n"

    local log_stat=""
    if [[ "$log_level" == "1" ]]; then
        log_stat="--log-stat"
    fi

    wasmedge --dir .:. --nn-preload default:GGML:AUTO:$model_file llama-chat.wasm --stream-stdout --prompt-template $prompt_template $log_stat
}

main() {
    check_os    
    printf "\n"
    prereq
    printf "\n"
    install_wasmedge
    printf "\n"
    download_model
    printf "\n"
    select_mode
    printf "\n"
    if [[ "$running_mode" == "1" ]]; then
        select_log_level
        printf "\n"
        download_chat_wasm
        printf "\n"
        start_chat
    elif [[ "$running_mode" == "2" ]]; then
        download_server
        printf "\n"
        download_webui_files
        printf "\n"
        start_server
    fi
}

main "$@" || exit 1