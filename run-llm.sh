#!/bin/bash

retry_download() {
    for i in {1..3}; do
        if [ $i -gt 1 ]; then
            echo "Retrying..."
        fi

        curl -LO $1 -#
        local r=$?
        if [ $r -eq 0 ]; then
            break
        fi
    done

    # Can only return the exit code
    # https://stackoverflow.com/questions/17336915/return-value-in-a-bash-function
    return $r
}

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

    models='
Llama-2-7B-Chat::https://code.flows.network/webhook/iwYN1SdN3AmPgR5ao5Gt/llama-2-7b-chat.Q5_K_M.gguf
Llama-2-13B-Chat::https://code.flows.network/webhook/iwYN1SdN3AmPgR5ao5Gt/llama-2-13b-chat.Q5_K_M.gguf
BELLE-Llama2-13B-Chat::https://code.flows.network/webhook/iwYN1SdN3AmPgR5ao5Gt/BELLE-Llama2-13B-Chat-0.4M-ggml-model-q4_0.gguf
MistralLite-7B::https://code.flows.network/webhook/iwYN1SdN3AmPgR5ao5Gt/mistrallite.Q5_K_M.gguf
Mistral-7B-Instruct-v0.1::https://code.flows.network/webhook/iwYN1SdN3AmPgR5ao5Gt/mistral-7b-instruct-v0.1.Q5_K_M.gguf
Mistral-7B-Instruct-v0.2::https://code.flows.network/webhook/iwYN1SdN3AmPgR5ao5Gt/mistral-7b-instruct-v0.2.Q4_0.gguf
OpenChat-3.5-0106::https://code.flows.network/webhook/iwYN1SdN3AmPgR5ao5Gt/openchat-3.5-0106-Q5_K_M.gguf
Wizard-Vicuna::https://code.flows.network/webhook/iwYN1SdN3AmPgR5ao5Gt/wizard-vicuna-13b-ggml-model-q8_0.gguf
CausalLM-14B::https://code.flows.network/webhook/iwYN1SdN3AmPgR5ao5Gt/causallm_14b.Q5_1.gguf
TinyLlama-1.1B-Chat-v1.0::https://code.flows.network/webhook/iwYN1SdN3AmPgR5ao5Gt/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf
TinyLlama-1.1B-Chat-v0.3::https://code.flows.network/webhook/iwYN1SdN3AmPgR5ao5Gt/tinyllama-1.1b-chat-v0.3.Q5_K_M.gguf
Baichuan2-13B-Chat::https://code.flows.network/webhook/iwYN1SdN3AmPgR5ao5Gt/Baichuan2-13B-Chat-ggml-model-q4_0.gguf
OpenHermes-2.5-Mistral-7B::https://code.flows.network/webhook/iwYN1SdN3AmPgR5ao5Gt/openhermes-2.5-mistral-7b.Q5_K_M.gguf
Dolphin-2.0-Mistral-7B::https://code.flows.network/webhook/iwYN1SdN3AmPgR5ao5Gt/dolphin-2.0-mistral-7b-ggml-model-q4_0.gguf
Dolphin-2.1-Mistral-7B::https://code.flows.network/webhook/iwYN1SdN3AmPgR5ao5Gt/dolphin-2.1-mistral-7b-ggml-model-q4_0.gguf
Dolphin-2.2-Yi-34B::https://code.flows.network/webhook/iwYN1SdN3AmPgR5ao5Gt/dolphin-2.2-yi-34b-ggml-model-q4_0.gguf
Dolphin-2.2-Mistral-7B::https://code.flows.network/webhook/iwYN1SdN3AmPgR5ao5Gt/dolphin-2.2-mistral-7b-ggml-model-q4_0.gguf
Dolphin-2.2.1-Mistral-7B::https://code.flows.network/webhook/iwYN1SdN3AmPgR5ao5Gt/dolphin-2.2.1-mistral-7b-ggml-model-q4_0.gguf
Samantha-1.2-Mistral-7B::https://code.flows.network/webhook/iwYN1SdN3AmPgR5ao5Gt/samantha-1.2-mistral-7b-ggml-model-q4_0.gguf
Samantha-1.11-CodeLlama-34B::https://code.flows.network/webhook/iwYN1SdN3AmPgR5ao5Gt/Samantha-1.11-CodeLlama-34b-ggml-model-q4_0.gguf
Samantha-1.11-7B::https://code.flows.network/webhook/iwYN1SdN3AmPgR5ao5Gt/Samantha-1.11-7b-ggml-model-q4_0.gguf
WizardLM-1.0-Uncensored-CodeLlama-34B::https://code.flows.network/webhook/iwYN1SdN3AmPgR5ao5Gt/WizardLM-1.0-Uncensored-CodeLlama-34b-ggml-model-q4_0.gguf
WizardLM-7B-V1.0-Uncensored::https://code.flows.network/webhook/iwYN1SdN3AmPgR5ao5Gt/wizardlm-7b-v1.0-uncensored.Q5_K_M.gguf
WizardLM-13B-V1.0-Uncensored::https://code.flows.network/webhook/iwYN1SdN3AmPgR5ao5Gt/wizardlm-13b-v1.0-uncensored.Q5_K_M.gguf
WizardCoder-Python-7B-V1.0::https://code.flows.network/webhook/iwYN1SdN3AmPgR5ao5Gt/WizardCoder-Python-7B-V1.0-ggml-model-q4_0.gguf
Zephyr-7B-Alpha::https://code.flows.network/webhook/iwYN1SdN3AmPgR5ao5Gt/zephyr-7b-alpha.Q5_K_M.gguf
Orca-2-13B::https://code.flows.network/webhook/iwYN1SdN3AmPgR5ao5Gt/Orca-2-13b-ggml-model-q4_0.gguf
Neural-Chat-7B-v3-1::https://code.flows.network/webhook/iwYN1SdN3AmPgR5ao5Gt/neural-chat-7b-v3-1-ggml-model-q4_0.gguf
Starling-LM-7B-alpha::https://code.flows.network/webhook/iwYN1SdN3AmPgR5ao5Gt/starling-lm-7b-alpha.Q5_K_M.gguf
Calm2-7B-Chat::https://code.flows.network/webhook/iwYN1SdN3AmPgR5ao5Gt/calm2-7b-chat.Q4_K_M.gguf
Deepseek-Coder-6.7B::https://code.flows.network/webhook/iwYN1SdN3AmPgR5ao5Gt/deepseek-coder-6.7b-instruct.Q5_K_M.gguf
Deepseek-LLM-7B-Chat::https://code.flows.network/webhook/iwYN1SdN3AmPgR5ao5Gt/deepseek-llm-7b-chat.Q5_K_M.gguf
SOLAR-10.7B-Instruct-v1.0::https://code.flows.network/webhook/iwYN1SdN3AmPgR5ao5Gt/solar-10.7b-instruct-v1.0.Q5_K_M.gguf
Mixtral-8x7B-Instruct-v0.1::https://code.flows.network/webhook/iwYN1SdN3AmPgR5ao5Gt/mixtral-8x7b-instruct-v0.1.Q4_0.gguf
dolphin-2.6-phi-2::https://code.flows.network/webhook/iwYN1SdN3AmPgR5ao5Gt/dolphin-2_6-phi-2.Q5_K_M.gguf
ELYZA-japanese-Llama-2-7b-instruct::https://code.flows.network/webhook/iwYN1SdN3AmPgR5ao5Gt/ELYZA-japanese-Llama-2-7b-instruct-q5_K_M.gguf
ELYZA-japanese-Llama-2-7b-fast-instruct::https://code.flows.network/webhook/iwYN1SdN3AmPgR5ao5Gt/ELYZA-japanese-Llama-2-7b-fast-instruct-q5_K_M.gguf
'

    sha256sums='
Llama-2-7B-Chat::e0b99920cf47b94c78d2fb06a1eceb9ed795176dfa3f7feac64629f1b52b997f
Llama-2-13B-Chat::ef36e090240040f97325758c1ad8e23f3801466a8eece3a9eac2d22d942f548a
BELLE-Llama2-13B-Chat::56879e1fd6ee6a138286730e121f2dba1be51b8f7e261514a594dea89ef32fe7
MistralLite-7B::d06d149c24eea0446ea7aad596aca396fe7f3302441e9375d5bbd3fd9ba8ebea
Mistral-7B-Instruct-v0.1::c4b062ec7f0f160e848a0e34c4e291b9e39b3fc60df5b201c038e7064dbbdcdc
Mistral-7B-Instruct-v0.2::25d80b918e4432661726ef408b248005bebefe3f8e1ac722d55d0c5dcf2893e0
OpenChat-3.5-0106::c28f69693336ab63369451da7f1365e5003d79f3ac69566de72100a8299a967a
Wizard-Vicuna::681b6571e624fd211ae81308b573f24f0016f6352252ae98241b44983bb7e756
CausalLM-14B::8ddb4c04e6f0c06971e9b6723688206bf9a5b8ffc85611cc7843c0e8c8a66c4e
TinyLlama-1.1B-Chat-v1.0::aa54a5fb99ace5b964859cf072346631b2da6109715a805d07161d157c66ce7f
TinyLlama-1.1B-Chat-v0.3::7c255febbf29c97b5d6f57cdf62db2f2bc95c0e541dc72c0ca29786ca0fa5eed
Baichuan2-13B-Chat::789685b86c86af68a1886949015661d3da0a9c959dffaae773afa4fe8cfdb840
OpenHermes-2.5-Mistral-7B::61e9e801d9e60f61a4bf1cad3e29d975ab6866f027bcef51d1550f9cc7d2cca6
Dolphin-2.0-Mistral-7B::37adbc161e6e98354ab06f6a79eaf30c4eb8dc60fb1226ef2fe8e84a84c5fdd6
Dolphin-2.1-Mistral-7B::021b2d9eb466e2b2eb522bc6d66906bb94c0dac721d6278e6718a4b6c9ecd731
Dolphin-2.2-Yi-34B::641b644fde162fd7f8e8991ca6873d8b0528b7a027f5d56b8ee005f7171ac002
Dolphin-2.2-Mistral-7B::77cf0861b5bc064e222075d0c5b73205d262985fc195aed6d30a7d3bdfefbd6c
Dolphin-2.2.1-Mistral-7B::c88edaa19afeb45075d566930571fc1f580329c6d6980f5222f442ee2894234e
Samantha-1.2-Mistral-7B::c29d3e84c626b6631864cf111ed2ce847d74a105f3bd66845863bbd8ea06628e
Samantha-1.11-CodeLlama-34B::67032c6b1bf358361da1b8162c5feb96dd7e02e5a42526543968caba7b7da47e
Samantha-1.11-7B::343ea7fadb7f89ec88837604f7a7bc6ec4f5109516e555d8ec0e1e416b06b997
WizardLM-1.0-Uncensored-CodeLlama-34B::4f000bba0cd527319fc2dfb4cabf447d8b48c2752dd8bd0c96f070b73cd53524
WizardLM-7B-V1.0-Uncensored::3ef0d681351556466b3fae523e7f687e3bf550d7974b3515520b290f3a8443e2
WizardLM-13B-V1.0-Uncensored::d5a9bf292e050f6e74b1be87134b02c922f61b0d665633ee4941249e80f36b50
WizardCoder-Python-7B-V1.0::0398068cb367d45faa3b8ebea1cc75fc7dec1cd323033df68302964e66879fed
Zephyr-7B-Alpha::2ad371d1aeca1ddf6281ca4ee77aa20ace60df33cab71d3bb681e669001e176e
Orca-2-13B::8c9ca393b2d882bd7bd0ba672d52eafa29bb22b2cd740418198c1fa1adb6478b
Neural-Chat-7B-v3-1::e57b76915fe5f0c0e48c43eb80fc326cb8366cbb13fcf617a477b1f32c0ac163
Starling-LM-7B-alpha::b6144d3a48352f5a40245ab1e89bfc0b17e4d045bf0e78fb512480f34ae92eba
Calm2-7B-Chat::42e829c19100c5d82c9432f0ee4c062e994fcf03966e8bfb2e92d1d91db12d56
Deepseek-Coder-6.7B::0976ee1707fc97b142d7266a9a501893ea6f320e8a8227aa1f04bcab74a5f556
Deepseek-LLM-7B-Chat::e5bcd887cc97ff63dbd17b8b9feac261516e985b5e78f1f544eb49cf403caaf6
SOLAR-10.7B-Instruct-v1.0::4ade240f5dcc253272158f3659a56f5b1da8405510707476d23a7df943aa35f7
Mixtral-8x7B-Instruct-v0.1::0c57465507f21bed4364fca37efd310bee92e25a4ce4f5678ef9b44e95830e4e
dolphin-2.6-phi-2::acc43043793230038f39491de557e70c9d99efddc41f1254e7064cc48f9b5c1e
ELYZA-japanese-Llama-2-7b-instruct::53c0a17b0bba8aedc868e5dce72e5976cd99108966659b8466476957e99dc980
ELYZA-japanese-Llama-2-7b-fast-instruct::3dc1e83340c2ee25903ff286da79a62999ab2b2ade7ae2a7c0d6db9f47e14087
'
    prompt_templates='
Llama-2-7B-Chat::llama-2-chat
Llama-2-13B-Chat::llama-2-chat
BELLE-Llama2-13B-Chat::belle-llama-2-chat
MistralLite-7B::mistrallite
Mistral-7B-Instruct-v0.1::mistral-instruct
Mistral-7B-Instruct-v0.2::mistral-instruct
OpenChat-3.5-0106::openchat
Wizard-Vicuna::vicuna-chat
CausalLM-14B::chatml
TinyLlama-1.1B-Chat-v1.0::chatml
TinyLlama-1.1B-Chat-v0.3::chatml
Baichuan2-13B-Chat::baichuan-2
OpenHermes-2.5-Mistral-7B::chatml
Dolphin-2.0-Mistral-7B::chatml
Dolphin-2.1-Mistral-7B::chatml
Dolphin-2.2-Yi-34B::chatml
Dolphin-2.2-Mistral-7B::chatml
Dolphin-2.2.1-Mistral-7B::chatml
Samantha-1.2-Mistral-7B::chatml
Samantha-1.11-CodeLlama-34B::vicuna-chat
Samantha-1.11-7B::vicuna-chat
WizardLM-1.0-Uncensored-CodeLlama-34B::vicuna-chat
WizardLM-7B-V1.0-Uncensored::vicuna-chat
WizardLM-13B-V1.0-Uncensored::vicuna-chat
WizardCoder-Python-7B-V1.0::wizard-coder
Zephyr-7B-Alpha::zephyr
Orca-2-13B::chatml
Neural-Chat-7B-v3-1::intel-neural
Starling-LM-7B-alpha::openchat
Calm2-7B-Chat::vicuna-1.1-chat
Deepseek-Coder-6.7B::deepseek-coder
Deepseek-LLM-7B-Chat::deepseek-chat
SOLAR-10.7B-Instruct-v1.0::solar-instruct
Mixtral-8x7B-Instruct-v0.1::mixtral-instruct
dolphin-2.6-phi-2::chatml
ELYZA-japanese-Llama-2-7b-instruct::llama-2-chat
ELYZA-japanese-Llama-2-7b-fast-instruct::llama-2-chat
'

    system_prompts='
Dolphin-2.2-Yi-34B::You are a helpful AI assistant
Samantha-1.11-CodeLlama-34B::You are a helpful AI assistant.
Samantha-1.11-7B::You are Samantha, a sentient AI companion.
WizardLM-1.0-Uncensored-CodeLlama-34B::You are a helpful AI assistant.
WizardLM-7B-V1.0-Uncensored::You are a helpful AI assistant.
WizardLM-13B-V1.0-Uncensored::You are a helpful AI assistant.
WizardCoder-Python-7B-V1.0::Below is an instruction that describes a task. Write a response that appropriately completes the request.
Zephyr-7B-Alpha::You are a friendly chatbot who always responds in the style of a pirate.
Orca-2-13B::You are Orca, an AI language model created by Microsoft. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior.
'

    reverse_prompts='
MistralLite-7B::</s>
OpenChat-3.5-0106::<|end_of_turn|>
Baichuan2-13B-Chat::用户:
OpenHermes-2.5-Mistral-7B::<|im_end|>
Dolphin-2.0-Mistral-7B::<|im_end|>
Dolphin-2.1-Mistral-7B::<|im_end|>
Dolphin-2.2-Yi-34B::<|im_end|>
Dolphin-2.2-Mistral-7B::<|im_end|>
Dolphin-2.2.1-Mistral-7B::<|im_end|>
Samantha-1.2-Mistral-7B::<|im_end|>
Zephyr-7B-Alpha::</s>
Starling-LM-7B-alpha::<|end_of_turn|>
'

    model_names="Llama-2-7B-Chat Llama-2-13B-Chat BELLE-Llama2-13B-Chat MistralLite-7B Mistral-7B-Instruct-v0.1 Mistral-7B-Instruct-v0.2 OpenChat-3.5-0106 Wizard-Vicuna CausalLM-14B TinyLlama-1.1B-Chat-v1.0 TinyLlama-1.1B-Chat-v0.3 Baichuan2-13B-Chat OpenHermes-2.5-Mistral-7B Dolphin-2.0-Mistral-7B Dolphin-2.1-Mistral-7B Dolphin-2.2-Yi-34B Dolphin-2.2-Mistral-7B Dolphin-2.2.1-Mistral-7B Samantha-1.2-Mistral-7B Samantha-1.11-CodeLlama-34B Samantha-1.11-7B WizardLM-1.0-Uncensored-CodeLlama-34B WizardLM-7B-V1.0-Uncensored WizardLM-13B-V1.0-Uncensored WizardCoder-Python-7B-V1.0 Zephyr-7B-Alpha Orca-2-13B Neural-Chat-7B-v3-1 Starling-LM-7B-alpha Calm2-7B-Chat Deepseek-Coder-6.7B Deepseek-LLM-7B-Chat SOLAR-10.7B-Instruct-v1.0 Mixtral-8x7B-Instruct-v0.1 ELYZA-japanese-Llama-2-7b-fast-instruct ELYZA-japanese-Llama-2-7b-instruct dolphin-2.6-phi-2"

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

    # Check if the provided model name exists in the models string
    url=$(echo "$models" | awk -F '::' -v model=$model '{for(i=1;i<=NF;i++)if($i==model)print $(i+1)}')

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
        retry_download $url
        if [ $? -ne 0 ]; then
            printf "\nFailed to download model file. Please try again\n"
            exit 1
        fi
    fi

    {
        if command -v sha256sum &> /dev/null
        then
            local cal_sum=$(sha256sum $filename | awk '{print $1}')
            local ori_sum=$(echo "$sha256sums" | awk -F '::' -v model=$model '{for(i=1;i<=NF;i++)if($i==model)print $(i+1)}')
            if [[ "$cal_sum" != "$ori_sum" ]]; then
                printf "\n\n**************************\n"
                printf "sha256sum of the model file $filename is not correct.\n"
                printf "Please remove the file then\n"
                printf "1) Manually download it from: $url\n"
                printf "or\n"
                printf "2) Ctrl+c to exit and restart this script\n"
                printf "**************************\n\n"
            fi
        fi
    }&

    model_file=$(basename $url)

    # Check if the provided model name exists in the models string
    prompt_template=$(echo "$prompt_templates" | awk -F '::' -v model=$model '{for(i=1;i<=NF;i++)if($i==model)print $(i+1)}')

    if [ -z "$prompt_template" ]; then
        printf "\n      The prompt template for the selected model does not exist.\n"
        exit 1
    fi

    system_prompt=$(echo "$system_prompts" | awk -F '::' -v model=$model '{for(i=1;i<=NF;i++)if($i==model)print $(i+1)}')

    reverse_prompt=$(echo "$reverse_prompts" | awk -F '::' -v model=$model '{for(i=1;i<=NF;i++)if($i==model)print $(i+1)}')
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

    wasm_url="https://code.flows.network/webhook/iwYN1SdN3AmPgR5ao5Gt/llama-api-server.wasm"
    retry_download $wasm_url
    if [ $? -ne 0 ]; then
        printf "\nFailed to download wasm file. Please try again\n"
        exit 1
    fi
}

download_webui_files() {
    printf "Downloading frontend resources of 'chatbot-ui' ...\n"

    files_tarball="https://github.com/second-state/chatbot-ui/releases/download/v0.1.0/chatbot-ui.tar.gz"
    retry_download $files_tarball
    if [ $? -ne 0 ]; then
        printf "\nFailed to download ui tarball. Please try again\n"
        exit 1
    fi
    tar xzf chatbot-ui.tar.gz
    rm chatbot-ui.tar.gz
}

start_server() {
    printf "Starting llama-api-server ...\n\n"

    set -x
    if [ -n "$reverse_prompt" ]; then
        wasmedge --dir .:. --nn-preload default:GGML:AUTO:$model_file llama-api-server.wasm -p $prompt_template -m "${model}" -r "${reverse_prompt[@]}"
    else
        wasmedge --dir .:. --nn-preload default:GGML:AUTO:$model_file llama-api-server.wasm -p $prompt_template -m "${model}"
    fi
    set +x
}

download_chat_wasm() {
    printf "Downloading 'llama-chat' wasm ...\n"

    wasm_url="https://code.flows.network/webhook/iwYN1SdN3AmPgR5ao5Gt/llama-chat.wasm"
    retry_download $wasm_url
    if [ $? -ne 0 ]; then
        printf "\nFailed to download wasm file. Please try again\n"
        exit 1
    fi
}

start_chat() {
    printf "starting llama-chat ... \n\n"

    local log_stat=""
    if [[ "$log_level" == "1" ]]; then
        log_stat="--log-stat"
    fi

    set -x
    if [ -n "$reverse_prompt" ] && [ -n "$system_prompt" ]; then
        wasmedge --dir .:. --nn-preload default:GGML:AUTO:$model_file llama-chat.wasm --prompt-template $prompt_template -r "${reverse_prompt[@]}" -s "${system_prompt[@]}" $log_stat
    elif [ -n "$reverse_prompt" ]; then
        wasmedge --dir .:. --nn-preload default:GGML:AUTO:$model_file llama-chat.wasm --prompt-template $prompt_template -r "${reverse_prompt[@]}" $log_stat
    elif [ -n "$system_prompt" ]; then
        wasmedge --dir .:. --nn-preload default:GGML:AUTO:$model_file llama-chat.wasm --prompt-template $prompt_template -s "${system_prompt[@]}" $log_stat
    else
        wasmedge --dir .:. --nn-preload default:GGML:AUTO:$model_file llama-chat.wasm --prompt-template $prompt_template $log_stat
    fi
    set +x
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
