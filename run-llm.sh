#!/bin/bash
#
# Helper script for deploying LlamaEdge API Server with a single Bash command
#
# - Works on Linux and macOS
# - Supports: CPU, CUDA, Metal, OpenCL
# - Can run GGUF models from https://huggingface.co/second-state/
#

set -e

# required utils: curl, git, make
if ! command -v curl &> /dev/null; then
    printf "[-] curl not found\n"
    exit 1
fi
if ! command -v git &> /dev/null; then
    printf "[-] git not found\n"
    exit 1
fi
if ! command -v make &> /dev/null; then
    printf "[-] make not found\n"
    exit 1
fi

# parse arguments
port=8080
repo=""
wtype=""
backend="cpu"
ctx_size=512
n_predict=1024
n_gpu_layers=100

# if macOS, use metal backend by default
if [[ "$OSTYPE" == "darwin"* ]]; then
    backend="metal"
elif command -v nvcc &> /dev/null; then
    backend="cuda"
fi

gpu_id=0
n_parallel=8
n_kv=4096
verbose=0
log_prompts=0
log_stat=0
# 0: server mode
# 1: local mode
# mode=0
# 0: non-interactive
# 1: interactive
interactive=0
model=""
# ggml version: latest or bxxxx
ggml_version="latest"

function print_usage {
    printf "Usage:\n"
    printf "  ./run-llm.sh [--port]\n\n"
    printf "  --model:        model name\n"
    printf "  --interactive:  run in interactive mode\n"
    printf "  --port:         port number, default is 8080\n"
    printf "  --ggml-version: ggml version, default is latest\n"
    printf "Example:\n\n"
    printf '  bash <(curl -sSfL 'https://code.flows.network/webhook/iwYN1SdN3AmPgR5ao5Gt/run-llm.sh')"\n\n'
}

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --model)
            model="$2"
            shift
            shift
            ;;
        --interactive)
            interactive=1
            shift
            ;;
        --port)
            port="$2"
            shift
            shift
            ;;
        --ggml-version)
            ggml_version="$2"
            shift
            shift
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $key"
            print_usage
            exit 1
            ;;
    esac
done

# available weights types
wtypes=("Q2_K" "Q3_K_L" "Q3_K_M" "Q3_K_S" "Q4_0" "Q4_K_M" "Q4_K_S" "Q5_0" "Q5_K_M" "Q5_K_S" "Q6_K" "Q8_0")

wfiles=()
for wt in "${wtypes[@]}"; do
    wfiles+=("")
done

ss_urls=(
    "https://huggingface.co/second-state/Gemma-2b-it-GGUF/resolve/main/gemma-2b-it-Q5_K_M.gguf"
    "https://huggingface.co/second-state/Phi-3-mini-4k-instruct-GGUF/resolve/main/Phi-3-mini-4k-instruct-Q4.gguf"
    "https://huggingface.co/second-state/Llama-2-7B-Chat-GGUF/resolve/main/Llama-2-7b-chat-hf-Q5_K_M.gguf"
    "https://huggingface.co/second-state/stablelm-2-zephyr-1.6b-GGUF/resolve/main/stablelm-2-zephyr-1_6b-Q5_K_M.gguf"
    "https://huggingface.co/second-state/OpenChat-3.5-0106-GGUF/resolve/main/openchat-3.5-0106-Q5_K_M.gguf"
    "https://huggingface.co/second-state/Yi-34B-Chat-GGUF/resolve/main/Yi-34B-Chat-Q5_K_M.gguf"
    "https://huggingface.co/second-state/Yi-34Bx2-MoE-60B-GGUF/resolve/main/Yi-34Bx2-MoE-60B-Q5_K_M.gguf"
    "https://huggingface.co/second-state/Deepseek-LLM-7B-Chat-GGUF/resolve/main/deepseek-llm-7b-chat-Q5_K_M.gguf"
    "https://huggingface.co/second-state/Deepseek-Coder-6.7B-Instruct-GGUF/resolve/main/deepseek-coder-6.7b-instruct-Q5_K_M.gguf"
    "https://huggingface.co/second-state/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/Mistral-7B-Instruct-v0.2-Q5_K_M.gguf"
    "https://huggingface.co/second-state/dolphin-2.6-mistral-7B-GGUF/resolve/main/dolphin-2.6-mistral-7b-Q5_K_M.gguf"
    "https://huggingface.co/second-state/Orca-2-13B-GGUF/resolve/main/Orca-2-13b-Q5_K_M.gguf"
    "https://huggingface.co/second-state/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/TinyLlama-1.1B-Chat-v1.0-Q5_K_M.gguf"
    "https://huggingface.co/second-state/SOLAR-10.7B-Instruct-v1.0-GGUF/resolve/main/SOLAR-10.7B-Instruct-v1.0-Q5_K_M.gguf"
)

# sample models
ss_models=(
    "gemma-2b-it"
    "phi-3-mini-4k"
    "llama-2-7b-chat"
    "stablelm-2-zephyr-1.6b"
    "openchat-3.5-0106"
    "yi-34b-chat"
    "yi-34bx2-moe-60b"
    "deepseek-llm-7b-chat"
    "deepseek-coder-6.7b-instruct"
    "mistral-7b-instruct-v0.2"
    "dolphin-2.6-mistral-7b"
    "orca-2-13b"
    "tinyllama-1.1b-chat-v1.0"
    "solar-10.7b-instruct-v1.0"
)

# prompt types
prompt_types=(
    "gemma-instruct"
    "phi-3-chat"
    "llama-2-chat"
    "chatml"
    "openchat"
    "zephyr"
    "codellama-instruct"
    "mistral-instruct"
    "mistrallite"
    "vicuna-chat"
    "vicuna-1.1-chat"
    "wizard-coder"
    "intel-neural"
    "deepseek-chat"
    "deepseek-coder"
    "solar-instruct"
    "belle-llama-2-chat"
    "human-assistant"
)


if [ -n "$model" ]; then
    printf "\n"

    # Check if the model is in the list of supported models
    if [[ ! " ${ss_models[@]} " =~ " ${model} " ]]; then

        printf "[+] ${model} is an invalid name or a unsupported model. Please check the model list:\n\n"

        for i in "${!ss_models[@]}"; do
            printf "    %2d) %s\n" "$((i+1))" "${ss_models[$i]}"
        done
        printf "\n"

        # ask for repo until index of sample repo is provided or an URL
        while [[ -z "$repo" ]]; do

            read -p "[+] Please select a number from the list above: " repo

            # check if the input is a number
            if [[ "$repo" -ge 1 && "$repo" -le ${#ss_models[@]} ]]; then
                ss_model="${ss_models[$repo-1]}"
                repo="${ss_urls[$repo-1]}"
            else
                printf "[-] Invalid repo index: %s\n" "$repo"
                repo=""
            fi
        done
    else
        # Find the index of the model in the list of supported models
        for i in "${!ss_models[@]}"; do
            if [[ "${ss_models[$i]}" = "${model}" ]]; then
                ss_model="${ss_models[$i]}"
                repo="${ss_urls[$i]}"

                break
            fi
        done

    fi

    # remove suffix
    repo=$(echo "$repo" | sed -E 's/\/tree\/main$//g')

    ss_url=$repo

    repo=${repo%/resolve/main/*}

    # check file if the model has been downloaded before
    wfile=$(basename "$ss_url")
    if [ -f "$wfile" ]; then
        printf "[+] Using cached model %s \n" "$wfile"
    else
        printf "[+] Downloading the selected model from %s\n" "$ss_url"

        # download the weights file
        curl -o "$wfile" -# -L "$ss_url"
    fi

    # * prompt type and reverse prompt

    readme_url="$repo/resolve/main/README.md"

    # Download the README.md file
    curl -s $readme_url -o README.md

    # Extract the "Prompt type: xxxx" line
    prompt_type_line=$(grep -i "Prompt type:" README.md)

    # Extract the xxxx part
    prompt_type=$(echo $prompt_type_line | cut -d'`' -f2 | xargs)

    printf "[+] Extracting prompt type: %s \n" "$prompt_type"

    # Check if "Reverse prompt" exists
    if grep -q "Reverse prompt:" README.md; then
        # Extract the "Reverse prompt: xxxx" line
        reverse_prompt_line=$(grep -i "Reverse prompt:" README.md)

        # Extract the xxxx part
        reverse_prompt=$(echo $reverse_prompt_line | cut -d'`' -f2 | xargs)

        printf "[+] Extracting reverse prompt: %s \n" "$reverse_prompt"
    else
        printf "[+] No reverse prompt required\n"
    fi

    # Clean up
    rm README.md

    # * install WasmEdge + wasi-nn_ggml plugin
    printf "[+] Install WasmEdge with wasi-nn_ggml plugin ...\n\n"

    if [ "$ggml_version" = "latest" ]; then
        if curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | bash -s -- -v 0.13.5 --plugins wasi_nn-ggml wasmedge_rustls; then
            source $HOME/.wasmedge/env
            wasmedge_path=$(which wasmedge)
            printf "\n    The WasmEdge Runtime is installed in %s.\n\n" "$wasmedge_path"
        else
            echo "Failed to install WasmEdge"
            exit 1
        fi
    else
        ggml_plugin="wasi_nn-ggml-$ggml_version"
        if curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | bash -s -- -v 0.13.5 --plugins $ggml_plugin wasmedge_rustls; then
            source $HOME/.wasmedge/env
            wasmedge_path=$(which wasmedge)
            printf "\n    The WasmEdge Runtime is installed in %s.\n\n" "$wasmedge_path"
        else
            echo "Failed to install WasmEdge"
            exit 1
        fi
    fi

    printf "\n"

    # * download llama-api-server.wasm
    printf "[+] Downloading the latest llama-api-server.wasm ...\n"
    curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-api-server.wasm
    printf "\n"

    # * download chatbot-ui
    printf "[+] Downloading Chatbot web app ...\n"
    files_tarball="https://github.com/second-state/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz"
    curl -LO $files_tarball
    if [ $? -ne 0 ]; then
        printf "    \nFailed to download ui tarball. Please manually download from https://github.com/second-state/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz and unzip the "chatbot-ui.tar.gz" to the current directory.\n"
        exit 1
    fi
    tar xzf chatbot-ui.tar.gz
    rm chatbot-ui.tar.gz
    printf "\n"

    model_name=${wfile%-Q*}

    cmd="wasmedge --dir .:. --nn-preload default:GGML:AUTO:$wfile llama-api-server.wasm --prompt-template ${prompt_type} --model-name ${model_name} --socket-addr 0.0.0.0:${port} --log-prompts --log-stat"

    # Add reverse prompt if it exists
    if [ -n "$reverse_prompt" ]; then
        cmd="$cmd --reverse-prompt \"${reverse_prompt}\""
    fi

    printf "\n"
    printf "[+] Will run the following command to start the server:\n\n"
    printf "    %s\n\n" "$cmd"
    printf "    Chatbot web app can be accessed at http://0.0.0.0:%s after the server is started\n\n\n" "$port"
    printf "*********************************** LlamaEdge API Server ********************************\n\n"
    printf "********************* [LOG: MODEL INFO (Load Model & Init Execution Context)] *********************"
    eval $cmd

elif [ "$interactive" -eq 0 ]; then

    printf "\n"
    # * install WasmEdge + wasi-nn_ggml plugin
    printf "[+] Installing WasmEdge with wasi-nn_ggml plugin ...\n\n"

    if [ "$ggml_version" = "latest" ]; then
        if curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | bash -s -- -v 0.13.5 --plugins wasi_nn-ggml wasmedge_rustls; then
            source $HOME/.wasmedge/env
            wasmedge_path=$(which wasmedge)
            printf "\n    The WasmEdge Runtime is installed in %s.\n\n" "$wasmedge_path"
        else
            echo "Failed to install WasmEdge"
            exit 1
        fi
    else
        ggml_plugin="wasi_nn-ggml-$ggml_version"
        if curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | bash -s -- -v 0.13.5 --plugins $ggml_plugin wasmedge_rustls; then
            source $HOME/.wasmedge/env
            wasmedge_path=$(which wasmedge)
            printf "\n    The WasmEdge Runtime is installed in %s.\n\n" "$wasmedge_path"
        else
            echo "Failed to install WasmEdge"
            exit 1
        fi
    fi

    printf "\n"

    # * download gemma-2b-it-Q5_K_M.gguf
    ss_url="https://huggingface.co/second-state/Gemma-2b-it-GGUF/resolve/main/gemma-2b-it-Q5_K_M.gguf"
    wfile=$(basename "$ss_url")
    if [ -f "$wfile" ]; then
        printf "[+] Using cached model %s \n" "$wfile"
    else
        printf "[+] Downloading %s ...\n" "$ss_url"

        # download the weights file
        curl -o "$wfile" -# -L "$ss_url"
    fi

    # * download llama-api-server.wasm
    printf "[+] Downloading the latest llama-api-server.wasm ...\n"
    curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-api-server.wasm
    printf "\n"

    # * download chatbot-ui
    printf "[+] Downloading Chatbot web app ...\n"
    files_tarball="https://github.com/second-state/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz"
    curl -LO $files_tarball
    if [ $? -ne 0 ]; then
        printf "    \nFailed to download ui tarball. Please manually download from https://github.com/second-state/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz and unzip the "chatbot-ui.tar.gz" to the current directory.\n"
        exit 1
    fi
    tar xzf chatbot-ui.tar.gz
    rm chatbot-ui.tar.gz
    printf "\n"

    # * start llama-api-server
    cmd="wasmedge --dir .:. --nn-preload default:GGML:AUTO:gemma-2b-it-Q5_K_M.gguf llama-api-server.wasm -p gemma-instruct -c 4096 --model-name gemma-2b-it --socket-addr 0.0.0.0:${port} --log-prompts --log-stat"

    printf "[+] Will run the following command to start the server:\n\n"
    printf "    %s\n\n" "$cmd"
    printf "    Chatbot web app can be accessed at http://0.0.0.0:%s after the server is started\n\n\n" "$port"
    printf "*********************************** LlamaEdge API Server ********************************\n\n"
    eval $cmd

elif [ "$interactive" -eq 1 ]; then

    printf "\n"
    printf "[I] This is a helper script for deploying LlamaEdge API Server on this machine.\n\n"
    printf "    The following tasks will be done:\n"
    printf "    - Download GGUF model\n"
    printf "    - Install WasmEdge Runtime and the wasi-nn_ggml plugin\n"
    printf "    - Download LlamaEdge API Server\n"
    printf "\n"
    printf "    Upon the tasks done, an HTTP server will be started and it will serve the selected\n"
    printf "    model.\n"
    printf "\n"
    printf "    Please note:\n"
    printf "\n"
    printf "    - All downloaded files will be stored in the current folder\n"
    printf "    - The server will be listening on all network interfaces\n"
    printf "    - The server will run with default settings which are not always optimal\n"
    printf "    - Do not judge the quality of a model based on the results from this script\n"
    printf "    - This script is only for demonstration purposes\n"
    printf "\n"
    printf "    During the whole process, you can press Ctrl-C to abort the current process at any time.\n"
    printf "\n"
    printf "    Press Enter to continue ...\n\n"

    read

    # * install WasmEdge + wasi-nn_ggml plugin

    printf "[+] Installing WasmEdge ...\n\n"

    # Check if WasmEdge has been installed
    reinstall_wasmedge=1
    if command -v wasmedge &> /dev/null
    then
        printf "    1) Install the latest version of WasmEdge and wasi-nn_ggml plugin (recommended)\n"
        printf "    2) Keep the current version\n\n"
        read -p "[+] Select a number from the list above: " reinstall_wasmedge
    fi

    while [[ "$reinstall_wasmedge" -ne 1 && "$reinstall_wasmedge" -ne 2 ]]; do
        printf "    Invalid number. Please enter number 1 or 2\n"
        read reinstall_wasmedge
    done

    if [[ "$reinstall_wasmedge" == "1" ]]; then
        # install WasmEdge + wasi-nn_ggml plugin
        if [ "$ggml_version" = "latest" ]; then
            if curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | bash -s -- -v 0.13.5 --plugins wasi_nn-ggml wasmedge_rustls; then
                source $HOME/.wasmedge/env
                wasmedge_path=$(which wasmedge)
                printf "\n    The WasmEdge Runtime is installed in %s.\n\n" "$wasmedge_path"
            else
                echo "Failed to install WasmEdge"
                exit 1
            fi
        else
            ggml_plugin="wasi_nn-ggml-$ggml_version"
            if curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | bash -s -- -v 0.13.5 --plugins $ggml_plugin wasmedge_rustls; then
                source $HOME/.wasmedge/env
                wasmedge_path=$(which wasmedge)
                printf "\n    The WasmEdge Runtime is installed in %s.\n\n" "$wasmedge_path"
            else
                echo "Failed to install WasmEdge"
                exit 1
            fi
        fi


    elif [[ "$reinstall_wasmedge" == "2" ]]; then
        wasmedge_path=$(which wasmedge)
        wasmedge_root_path=${wasmedge_path%"/bin/wasmedge"}

        found=0
        for file in "$wasmedge_root_path/plugin/libwasmedgePluginWasiNN."*; do
        if [[ -f $file ]]; then
            found=1
            break
        fi
        done

        if [[ $found -eq 0 ]]; then
            printf "\n    * Not found wasi-nn_ggml plugin. Please download it from https://github.com/WasmEdge/WasmEdge/releases/ and move it to %s. After that, please rerun the script. \n\n" "$wasmedge_root_path/plugin/"

            exit 1
        fi

    fi

    printf "[+] The most popular models at https://huggingface.co/second-state:\n\n"

    for i in "${!ss_models[@]}"; do
        printf "    %2d) %s\n" "$((i+1))" "${ss_models[$i]}"
    done

    # ask for repo until index of sample repo is provided or an URL
    while [[ -z "$repo" ]]; do
        printf "\n    Or choose one from: https://huggingface.co/models?sort=trending&search=gguf\n\n"

        read -p "[+] Please select a number from the list above or enter an URL: " repo

        # check if the input is a number
        if [[ "$repo" =~ ^[0-9]+$ ]]; then
            if [[ "$repo" -ge 1 && "$repo" -le ${#ss_models[@]} ]]; then
                ss_model="${ss_models[$repo-1]}"
                repo="${ss_urls[$repo-1]}"
            else
                printf "[-] Invalid repo index: %s\n" "$repo"
                repo=""
            fi
        elif [[ "$repo" =~ ^https?:// ]]; then
            repo="$repo"
        else
            printf "[-] Invalid repo URL: %s\n" "$repo"
            repo=""
        fi
    done


    # remove suffix
    repo=$(echo "$repo" | sed -E 's/\/tree\/main$//g')

    if [ -n "$ss_model" ]; then
        ss_url=$repo
        repo=${repo%/resolve/main/*}

        # check file if the model has been downloaded before
        wfile=$(basename "$ss_url")
        if [ -f "$wfile" ]; then
            printf "[+] Using cached model %s \n" "$wfile"
        else
            printf "[+] Downloading the selected model from %s\n" "$ss_url"

            # download the weights file
            curl -o "$wfile" -# -L "$ss_url"
        fi

    else

        printf "[+] Checking for GGUF model files in %s\n\n" "$repo"

        # find GGUF files in the source
        model_tree="${repo%/}/tree/main"
        model_files=$(curl -s "$model_tree" | grep -i "\\.gguf</span>" | sed -E 's/.*<span class="truncate group-hover:underline">(.*)<\/span><\/a>/\1/g')
        # Convert model_files into an array
        model_files_array=($model_files)

        while IFS= read -r line; do
            sizes+=("$line")
        done < <(curl -s "$model_tree" | awk -F 'download=true">' '/download=true">[0-9\.]+ (GB|MB)/ {print $2}' | awk '{print $1, $2}')

        # list all files in the provided git repo
        length=${#model_files_array[@]}
        for ((i=0; i<$length; i++)); do
            file=${model_files_array[i]}
            size=${sizes[i]}
            iw=-1
            is=0
            for wt in "${wtypes[@]}"; do
                # uppercase
                ufile=$(echo "$file" | tr '[:lower:]' '[:upper:]')
                if [[ "$ufile" =~ "$wt" ]]; then
                    iw=$is
                    break
                fi
                is=$((is+1))
            done

            if [[ $iw -eq -1 ]]; then
                continue
            fi

            wfiles[$iw]="$file"

            have=" "
            if [[ -f "$file" ]]; then
                have="*"
            fi

            printf "    %2d) %s %7s   %s\n" $iw "$have" "$size" "$file"
        done

        # ask for weights type until provided and available
        while [[ -z "$wtype" ]]; do
            printf "\n"
            read -p "[+] Please select a number from the list above: " wtype
            wfile="${wfiles[$wtype]}"

            if [[ -z "$wfile" ]]; then
                printf "[-] Invalid number: %s\n" "$wtype"
                wtype=""
            fi
        done

        url="${repo%/}/resolve/main/$wfile"

        # check file if the model has been downloaded before
        if [ -f "$wfile" ]; then
            printf "[+] Using cached model %s \n" "$wfile"
        else
            printf "[+] Downloading the selected model from %s\n" "$url"

            # download the weights file
            curl -o "$wfile" -# -L "$url"
        fi

    fi

    # * prompt type and reverse prompt

    if [[ $repo =~ ^https://huggingface\.co/second-state ]]; then
        readme_url="$repo/resolve/main/README.md"

        # Download the README.md file
        curl -s $readme_url -o README.md

        # Extract the "Prompt type: xxxx" line
        prompt_type_line=$(grep -i "Prompt type:" README.md)

        # Extract the xxxx part
        prompt_type=$(echo $prompt_type_line | cut -d'`' -f2 | xargs)

        printf "[+] Extracting prompt type: %s \n" "$prompt_type"

        # Check if "Reverse prompt" exists
        if grep -q "Reverse prompt:" README.md; then
            # Extract the "Reverse prompt: xxxx" line
            reverse_prompt_line=$(grep -i "Reverse prompt:" README.md)

            # Extract the xxxx part
            reverse_prompt=$(echo $reverse_prompt_line | cut -d'`' -f2 | xargs)

            printf "[+] Extracting reverse prompt: %s \n" "$reverse_prompt"
        else
            printf "[+] No reverse prompt required\n"
        fi

        # Clean up
        rm README.md
    else
        printf "[+] Please select a number from the list below:\n"
        printf "    The definitions of the prompt types below can be found at https://github.com/LlamaEdge/LlamaEdge/raw/main/api-server/chat-prompts/README.md\n\n"

        is=0
        for r in "${prompt_types[@]}"; do
            printf "    %2d) %s\n" $is "$r"
            is=$((is+1))
        done
        printf "\n"

        prompt_type_index=-1
        while ((prompt_type_index < 0 || prompt_type_index >= ${#prompt_types[@]})); do
            read -p "[+] Select prompt type: " prompt_type_index
            # Check if the input is a number
            if ! [[ "$prompt_type_index" =~ ^[0-9]+$ ]]; then
                echo "Invalid input. Please enter a number."
                prompt_type_index=-1
            fi
        done
        prompt_type="${prompt_types[$prompt_type_index]}"

        # Ask user if they need to set "reverse prompt"
        while [[ ! $need_reverse_prompt =~ ^[yYnN]$ ]]; do
            read -p "[+] Need reverse prompt? (y/n): " need_reverse_prompt
        done

        # If user answered yes, ask them to input a string
        if [[ "$need_reverse_prompt" == "y" || "$need_reverse_prompt" == "Y" ]]; then
            read -p "    Enter the reverse prompt: " reverse_prompt
            printf "\n"
        fi
    fi

    # * running mode

    printf "[+] Running mode: \n\n"

    running_modes=("API Server with Chatbot web app" "CLI Chat")

    for i in "${!running_modes[@]}"; do
        printf "    %2d) %s\n" "$((i+1))" "${running_modes[$i]}"
    done

    while [[ -z "$running_mode_index" ]]; do
        printf "\n"
        read -p "[+] Select a number from the list above: " running_mode_index
        running_mode="${running_modes[$running_mode_index - 1]}"

        if [[ -z "$running_mode" ]]; then
            printf "[-] Invalid number: %s\n" "$running_mode_index"
            running_mode_index=""
        fi
    done
    printf "[+] Selected running mode: %s (%s)\n" "$running_mode_index" "$running_mode"

    # * download llama-api-server.wasm or llama-chat.wasm

    repo="second-state/LlamaEdge"
    releases=$(curl -s "https://api.github.com/repos/$repo/releases")
    if [[ "$running_mode_index" == "1" ]]; then

        # * Download llama-api-server.wasm

        if [ -f "llama-api-server.wasm" ]; then
            # Ask user if they need to set "reverse prompt"
            while [[ ! $use_latest_version =~ ^[yYnN]$ ]]; do
                read -p "[+] You already have llama-api-server.wasm. Download the latest llama-api-server.wasm? (y/n): " use_latest_version
            done

            # If user answered yes, ask them to input a string
            if [[ "$use_latest_version" == "y" || "$use_latest_version" == "Y" ]]; then
                printf "[+] Downloading the latest llama-api-server.wasm ...\n"
                curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-api-server.wasm

                printf "\n"

            else
                printf "[+] Using cached llama-api-server.wasm\n"
            fi

        else
            printf "[+] Downloading the latest llama-api-server.wasm ...\n"
            curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-api-server.wasm

            printf "\n"
        fi

        # * chatbot-ui

        if [ -d "chatbot-ui" ]; then
            printf "[+] Using cached Chatbot web app\n"
        else
            printf "[+] Downloading Chatbot web app ...\n"
            files_tarball="https://github.com/second-state/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz"
            curl -LO $files_tarball
            if [ $? -ne 0 ]; then
                printf "    \nFailed to download ui tarball. Please manually download from https://github.com/second-state/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz and unzip the "chatbot-ui.tar.gz" to the current directory.\n"
                exit 1
            fi
            tar xzf chatbot-ui.tar.gz
            rm chatbot-ui.tar.gz
            printf "\n"
        fi

        model_name=${wfile%-Q*}

        cmd="wasmedge --dir .:. --nn-preload default:GGML:AUTO:$wfile llama-api-server.wasm --prompt-template ${prompt_type} --model-name ${model_name} --socket-addr 0.0.0.0:${port} --log-prompts --log-stat"

        # Add reverse prompt if it exists
        if [ -n "$reverse_prompt" ]; then
            cmd="$cmd --reverse-prompt \"${reverse_prompt}\""
        fi

        printf "[+] Will run the following command to start the server:\n\n"
        printf "    %s\n\n" "$cmd"

        # Ask user if they need to set "reverse prompt"
        while [[ ! $start_server =~ ^[yYnN]$ ]]; do
            read -p "[+] Confirm to start the server? (y/n): " start_server
        done

        # If user answered yes, ask them to input a string
        if [[ "$start_server" == "y" || "$start_server" == "Y" ]]; then
            printf "\n"
            printf "    Chatbot web app can be accessed at http://0.0.0.0:%s after the server is started\n\n\n" "$port"
            printf "*********************************** LlamaEdge API Server ********************************\n\n"
            eval $cmd

        fi

    elif [[ "$running_mode_index" == "2" ]]; then

        # * Download llama-chat.wasm

        if [ -f "llama-chat.wasm" ]; then
            # Ask user if they need to set "reverse prompt"
            while [[ ! $use_latest_version =~ ^[yYnN]$ ]]; do
                read -p "[+] You already have llama-chat.wasm. Download the latest llama-chat.wasm? (y/n): " use_latest_version
            done

            # If user answered yes, ask them to input a string
            if [[ "$use_latest_version" == "y" || "$use_latest_version" == "Y" ]]; then
                printf "[+] Downloading the latest llama-chat.wasm ...\n"
                curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-chat.wasm

                printf "\n"

            else
                printf "[+] Using cached llama-chat.wasm\n"
            fi

        else
            printf "[+] Downloading the latest llama-chat.wasm ...\n"
            curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-chat.wasm

            printf "\n"
        fi

        # * prepare the command

        cmd="wasmedge --dir .:. --nn-preload default:GGML:AUTO:$wfile llama-chat.wasm --prompt-template $prompt_type"

        # Add reverse prompt if it exists
        if [ -n "$reverse_prompt" ]; then
            cmd="$cmd --reverse-prompt \"${reverse_prompt}\""
        fi

        printf "[+] Will run the following command to start CLI Chat:\n\n"
        printf "    %s\n\n" "$cmd"

        # Ask user if they need to set "reverse prompt"
        while [[ ! $start_chat =~ ^[yYnN]$ ]]; do
            read -p "[+] Confirm to start CLI Chat? (y/n): " start_chat
        done

        # If user answered yes, ask them to input a string
        if [[ "$start_chat" == "y" || "$start_chat" == "Y" ]]; then
            printf "\n"

            # Execute the command
            printf "********************* LlamaEdge *********************\n\n"
            eval $cmd

        fi

    else
        printf "[-] Invalid running mode: %s\n" "$running_mode_index"
        exit 1
    fi

else
    echo "Invalid value for interactive"
fi

exit 0
