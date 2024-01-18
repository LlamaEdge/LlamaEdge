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
if ! command -v jq &> /dev/null; then
    printf "[-] jq not found\n"
    exit 1
fi

# parse arguments
port=8080
ctx_size=512
repo=""
wtype=""
backend="cpu"

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
mode=0

function print_usage {
    printf "Usage:\n"
    printf "  ./run-llm.sh [--port] [--repo] [--wtype] [--backend] [--gpu-id] [--n-parallel] [--n-kv] [--verbose]\n\n"
    printf "  --port:         port number, default is 8080\n"
    printf "  --ctx-size:     context size, default is 512\n"
    # printf "  --backend:    cpu, cuda, metal, opencl, depends on the OS\n"
    # printf "  --gpu-id:     gpu id, default is 0\n"
    # printf "  --verbose:    verbose output\n\n"
    # printf "  --mode:         running mode. 0: server mode (default) 1: local mode\n"
    # printf "  --log-prompts:  print prompt log message\n"
    # printf "  --log-stat:     print statistics log message\n\n"
    printf "Example:\n\n"
    printf '  bash <(curl -sSfL 'https://code.flows.network/webhook/iwYN1SdN3AmPgR5ao5Gt/run-llm.sh')"\n\n'
}

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --port)
            port="$2"
            shift
            shift
            ;;
        --ctx-size)
            ctx_size="$2"
            shift
            shift
            ;;
        # --mode)
        #     mode=1
        #     shift
        #     ;;
        # --log-prompts)
        #     log_prompts=1
        #     shift
        #     ;;
        # --log-stat)
        #     log_stat=1
        #     shift
        #     ;;
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

# sample repos
repos=(
    "https://huggingface.co/second-state/Llama-2-7B-Chat-GGUF"
    "https://huggingface.co/second-state/Mistral-7B-Instruct-v0.2-GGUF"
    "https://huggingface.co/second-state/dolphin-2.6-mistral-7B-GGUF"
    "https://huggingface.co/second-state/Orca-2-13B-GGUF"
    "https://huggingface.co/second-state/TinyLlama-1.1B-Chat-v1.0-GGUF"
    "https://huggingface.co/second-state/OpenChat-3.5-0106-GGUF"
    "https://huggingface.co/second-state/SOLAR-10.7B-Instruct-v1.0-GGUF"
    "https://huggingface.co/second-state/OpenHermes-2.5-Mistral-7B-GGUF"
)

# prompt types
prompt_types=(
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
)

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
printf "    If you don't know what you are doing, please press Ctrl-C to abort now\n"
printf "\n"
printf "    Press Enter to continue ...\n\n"

read

printf "[+] The most popular models at https://huggingface.co/second-state:\n\n"

is=0
for r in "${repos[@]}"; do
    printf "    %2d) %s\n" $is "$r"
    is=$((is+1))
done

# ask for repo until index of sample repo is provided or an URL
while [[ -z "$repo" ]]; do
    printf "\n    Or choose one from: https://huggingface.co/models?sort=trending&search=gguf\n\n"
    read -p "[+] Please select a number from the list above or enter an URL: " repo

    # check if the input is a number
    if [[ "$repo" =~ ^[0-9]+$ ]]; then
        if [[ "$repo" -ge 0 && "$repo" -lt ${#repos[@]} ]]; then
            repo="${repos[$repo]}"
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

printf "[+] Checking for GGUF model files in %s\n" "$repo"

# find GGUF files in the source
model_tree="${repo%/}/tree/main"
model_files=$(curl -s "$model_tree" | grep -i "\\.gguf</span>" | sed -E 's/.*<span class="truncate group-hover:underline">(.*)<\/span><\/a>/\1/g')
# Convert model_files into an array
model_files_array=($model_files)

# find GGUF file sizes in the source
sizes=()
while IFS= read -r line; do
    sizes+=("$line")
done < <(curl -s "$model_tree" | awk -F'[<>]' '/GB|MB/{print $3}')

# list all files in the provided git repo
printf "[+] Available models:\n\n"
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

# printf "[+] Selected model: %s (%s)\n" "$wtype" "$wfile"

url="${repo%/}/resolve/main/$wfile"

# check file if the model has been downloaded before
chk="$wfile.chk"

# check if we should download the file
# - if $wfile does not exist
# - if $wfile exists but $chk does not exist
# - if $wfile exists and $chk exists but $wfile is newer than $chk
# TODO: better logic using git lfs info

do_download=0

if [[ ! -f "$wfile" ]]; then
    do_download=1
elif [[ ! -f "$chk" ]]; then
    do_download=1
elif [[ "$wfile" -nt "$chk" ]]; then
    do_download=1
fi

if [[ $do_download -eq 1 ]]; then
    printf "[+] Downloading the selected model from %s\n" "$url"

    # download the weights file
    curl -o "$wfile" -# -L "$url"

    # create a check file if successful
    if [[ $? -eq 0 ]]; then
        printf "[+] Creating check file %s \n" "$chk"
        touch "$chk"
    fi
else
    printf "[+] Using cached model %s \n" "$wfile"
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

        printf "[+] Extracting reverse prompt: %s \n\n" "$reverse_prompt"
    else
        printf "[+] No reverse prompt required\n\n"
    fi

    # Clean up
    rm README.md
else
    printf "[+] Please select a number from the list below:\n"
    printf "    The definitions of the prompt types below can be found at https://github.com/second-state/LlamaEdge/raw/main/api-server/chat-prompts/README.md\n\n"

    is=0
    for r in "${prompt_types[@]}"; do
        printf "    %2d) %s\n" $is "$r"
        is=$((is+1))
    done

    printf "\n"
    read -p "[+] Select prompt type: " prompt_type_index
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

# * install WasmEdge + wasi-nn_ggml plugin

printf "[+] Installing WasmEdge ...\n"

# Check if WasmEdge has been installed
reinstall_wasmedge=1
if command -v wasmedge &> /dev/null
then
    printf "    Found WasmEdge in the current environment:\n\n"
    printf "     1) Reinstall WasmEdge and wasi-nn_ggml plugin (recommended)\n"
    printf "     2) Keep current version\n\n"
    read -p "[+] Select a number from the list above: " reinstall_wasmedge
fi

while [[ "$reinstall_wasmedge" -ne 1 && "$reinstall_wasmedge" -ne 2 ]]; do
    printf "    Invalid number. Please enter number 1 or 2\n"
    read reinstall_wasmedge
done

if [[ "$reinstall_wasmedge" == "1" ]]; then
    # uninstall WasmEdge
    if bash <(curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/uninstall.sh) -q; then

        # install WasmEdge + wasi-nn_ggml plugin
        if curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | bash -s -- --plugins wasi_nn-ggml; then
            source $HOME/.wasmedge/env
            wasmedge_path=$(which wasmedge)
            printf "\n    The WasmEdge Runtime is installed in %s.\n\n    * To uninstall it, use the command 'bash <(curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/uninstall.sh) -q'\n\n" "$wasmedge_path"
        else
            echo "Failed to install WasmEdge"
            exit 1
        fi

    else
        echo "Failed to uninstall WasmEdge"
        exit 1
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

# * select llama-edge server

repo="second-state/LlamaEdge"
releases=$(curl -s "https://api.github.com/repos/$repo/releases")

release_names=()
asset_urls=()

for i in {0..2}
do
    release_info=$(echo $releases | jq -r ".[$i]")
    release_name=$(echo $release_info | jq -r '.name')

    if [[ ! ${release_name} =~ ^LlamaEdge\ [0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        continue
    fi

    release_names+=("$release_name")

    asset_url=$(echo $release_info | jq -r '.assets[] | select(.name=="llama-api-server.wasm") | .browser_download_url')
    asset_urls+=("$asset_url")
done

# check if the current directory contains llama-api-server.wasm
if [ -f "llama-api-server.wasm" ]; then
    version_existed=$(wasmedge llama-api-server.wasm -V | cut -d' ' -f2)
fi

printf "[+] The latest three releases: \n\n"
for i in "${!release_names[@]}"; do
    if [[ ! ${release_names[$i]} =~ ^LlamaEdge\ [0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        continue
    fi

    release_info=${release_names[$i]}
    version=$(echo $release_info | cut -d' ' -f2)

    have=" "
    if [[ "$version" == "$version_existed" ]]; then
        have="*"
    fi

    printf "    %2d) %s %s\n" "$((i+1))" "$have" "${release_names[$i]}"
done

release_index=""
while [[ -z "$release_index" ]] || ! [[ "$release_index" =~ ^[0-9]+$ ]] || ((release_index < 1 || release_index > ${#release_names[@]})); do
    printf "\n"
    read -p "[+] Select a number from the list above: " release_index
done

release_name=${release_names[$release_index-1]}

# * Download llama-api-server.wasm

asset_url=${asset_urls[$((release_index-1))]}

version_selected=$(echo $release_name | cut -d' ' -f2)

if [[ "$version_selected" != "$version_existed" ]]; then
    printf "[+] Downloading llama-api-server.wasm to the current directory\n\n"
    curl -LO $asset_url
else
    printf "[+] Using cached llama-api-server.wasm\n\n"
fi

# * log options

printf "[+] Log options: \n\n"

log_options=("Show prompts" "Show execution statistics" "Show all" "Disable log")

for i in "${!log_options[@]}"; do
    printf "    %2d) %s\n" "$((i+1))" "${log_options[$i]}"
done

while [[ -z "$log_option_index" ]]; do
    printf "\n"
    read -p "[+] Select a number from the list above: " log_option_index
    log_option="${log_options[$log_option_index - 1]}"

    if [[ -z "$log_option" ]]; then
        printf "[-] Invalid number: %s\n" "$log_option_index"
        log_option_index=""
    fi
done
printf "[+] Selected log option: %s (%s)\n\n" "$log_option_index" "$log_option"

if [[ "$log_option_index" == "1" ]]; then
    log_prompts=1
elif [[ "$log_option_index" == "2" ]]; then
    log_stat=1
elif [[ "$log_option_index" == "3" ]]; then
    log_prompts=1
    log_stat=1
fi


# * start server
printf "[+] Start LlamaEdge server\n\n"

model_name=${wfile%-Q*}

cmd="wasmedge --dir .:. --nn-preload default:GGML:AUTO:$wfile llama-api-server.wasm -p $prompt_type -m \"${model_name}\" --socket-addr \"0.0.0.0:$port\""

# Add reverse prompt if it exists
if [ -n "$reverse_prompt" ]; then
    cmd="$cmd -r \"${reverse_prompt}\""
fi

# Add log prompts if log_prompts equals 1
if [ "$log_prompts" -eq 1 ]; then
    cmd="$cmd --log-prompts"
fi

# Add log stat if log_stat equals 1
if [ "$log_stat" -eq 1 ]; then
    cmd="$cmd --log-stat"
fi

# Execute the command
set -x
eval $cmd
set +x

exit 0