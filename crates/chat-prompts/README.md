# Prompt Templates for LLMs

`chat-prompts` is part of [LlamaEdge API Server](https://github.com/LlamaEdge/LlamaEdge/tree/main/api-server) project. It provides a collection of prompt templates that are used to generate prompts for the LLMs (See models in [huggingface.co/second-state](https://huggingface.co/second-state)).

## Prompt Templates

The available prompt templates are listed below:

- `baichuan-2`
  - Prompt string

    ```text
    以下内容为人类用户与与一位智能助手的对话。

    用户:你好！
    助手:
    ```

  - Example: [second-state/Baichuan2-13B-Chat-GGUF](https://huggingface.co/second-state/Baichuan2-13B-Chat-GGUF)

- `codellama-instruct`
  - Prompt string

    ```text
    <s>[INST] <<SYS>>
    Write code to solve the following coding problem that obeys the constraints and passes the example test cases. Please wrap your code answer using ```: <</SYS>>

    {prompt} [/INST]
    ```

  - Example: [second-state/CodeLlama-13B-Instruct-GGUF](https://huggingface.co/second-state/CodeLlama-13B-Instruct-GGUF)

- `codellama-super-instruct`
  - Prompt string

    ```text
    <s>Source: system\n\n {system_prompt} <step> Source: user\n\n {user_message_1} <step> Source: assistant\n\n {ai_message_1} <step> Source: user\n\n {user_message_2} <step> Source: assistant\nDestination: user\n\n
    ```

  - Example: [second-state/CodeLlama-70b-Instruct-hf-GGUF](https://huggingface.co/second-state/CodeLlama-70b-Instruct-hf-GGUF)

- `chatml`
  - Prompt string

    ```text
    <|im_start|>system
    {system_message}<|im_end|>
    <|im_start|>user
    {prompt}<|im_end|>
    <|im_start|>assistant
    ```

  - Example: [second-state/Yi-34B-Chat-GGUF](https://huggingface.co/second-state/Yi-34B-Chat-GGUF)

- `chatml-think`
  - Prompt string

    ```text
    <|im_start|>system
    {system_message}<|im_end|>
    <|im_start|>user
    {prompt}<|im_end|>
    <|im_start|>assistant
    <|im_start|>think
    ```

    - Example: [Tessa-T1](https://huggingface.co/collections/second-state/tessa-t1-gguf-models-67e2abc86de4bbf715fa5448)

- `chatml-tool`
  - Prompt string

    ```text
    <|im_start|>system\n{system_message} Here are the available tools: <tools> [{tool_1}, {tool_2}] </tools> Use the following pydantic model json schema for each tool call you will make: {"properties": {"arguments": {"title": "Arguments", "type": "object"}, "name": {"title": "Name", "type": "string"}}, "required": ["arguments", "name"], "title": "FunctionCall", "type": "object"} For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:\n<tool_call>\n{"arguments": <args-dict>, "name": <function-name>}\n</tool_call><|im_end|>
    <|im_start|>user
    {user_message}<|im_end|>
    <|im_start|>assistant
    ```

    - Example

      ```text
      <|im_start|>system\nYou are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. Here are the available tools: <tools> [{"type":"function","function":{"name":"get_current_weather","description":"Get the current weather in a given location","parameters":{"type":"object","properties":{"location":{"type":"string","description":"The city and state, e.g. San Francisco, CA"},"format":{"type":"string","description":"The temperature unit to use. Infer this from the users location.","enum":["celsius","fahrenheit"]}},"required":["location","format"]}}},{"type":"function","function":{"name":"predict_weather","description":"Predict the weather in 24 hours","parameters":{"type":"object","properties":{"location":{"type":"string","description":"The city and state, e.g. San Francisco, CA"},"format":{"type":"string","description":"The temperature unit to use. Infer this from the users location.","enum":["celsius","fahrenheit"]}},"required":["location","format"]}}}] </tools> Use the following pydantic model json schema for each tool call you will make: {"properties": {"arguments": {"title": "Arguments", "type": "object"}, "name": {"title": "Name", "type": "string"}}, "required": ["arguments", "name"], "title": "FunctionCall", "type": "object"} For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:\n<tool_call>\n{"arguments": <args-dict>, "name": <function-name>}\n</tool_call><|im_end|>
      <|im_start|>user
      Hey! What is the weather like in Beijing?<|im_end|>
      <|im_start|>assistant
      ```

  - Example: [second-state/Hermes-2-Pro-Llama-3-8B-GGUF](https://huggingface.co/second-state/Hermes-2-Pro-Llama-3-8B-GGUF)

- `deepseek-chat`
  - Prompt string

    ```text
    User: {user_message_1}

    Assistant: {assistant_message_1}<｜end▁of▁sentence｜>User: {user_message_2}

    Assistant:
    ```

  - Example: [second-state/Deepseek-LLM-7B-Chat-GGUF](https://huggingface.co/second-state/Deepseek-LLM-7B-Chat-GGUF)

- `deepseek-chat-2`
  - Prompt string

    ```text
    <|begin_of_sentence|>{system_message}

    User: {user_message_1}

    Assistant: {assistant_message_1}<|end_of_sentence|>User: {user_message_2}

    Assistant:
    ```

  - Example: [second-state/DeepSeek-Coder-V2-Lite-Instruct-GGUF](https://huggingface.co/second-state/DeepSeek-Coder-V2-Lite-Instruct-GGUF)

- `deepseek-chat-25`
  - Prompt string

    ```text
    <|begin_of_sentence|>{system_message}<|User|>{user_message_1}<|Assistant|>{assistant_message_1}<|end_of_sentence|><|User|>{user_message_2}<|Assistant|>
    ```

  - Example: [second-state/DeepSeek-V2.5-GGUF](https://huggingface.co/second-state/DeepSeek-V2.5-GGUF)

- `deepseek-coder`
  - Prompt string

    ```text
    {system}
    ### Instruction:
    {question_1}
    ### Response:
    {answer_1}
    <|EOT|>
    ### Instruction:
    {question_2}
    ### Response:
    ```

  - Example: [second-state/Deepseek-Coder-6.7B-Instruct-GGUF](https://huggingface.co/second-state/Deepseek-Coder-6.7B-Instruct-GGUF)

- `embedding`
  - Prompt string
    This prompt template is only used for embedding models. It works as a placeholder, therefore, it has no concrete prompt string.

  - Example: [second-state/E5-Mistral-7B-Instruct-Embedding-GGUF](https://huggingface.co/second-state/E5-Mistral-7B-Instruct-Embedding-GGUF)

- `exaone-chat`
  - Prompt string

    ```text
    [|system|]{system_message}[|endofturn|]
    [|user|]{user_message_1}
    [|assistant|]{assistant_message_1}[|endofturn|]
    [|user|]{user_message_2}
    [|assistant|]
    ```

  - Example: [second-state/EXAONE-3.5-32B-Instruct-GGUF](https://huggingface.co/second-state/EXAONE-3.5-32B-Instruct-GGUF)

- `exaone-deep-chat`
  - Prompt string

    ```text
    [|system|]{system_message}[|endofturn|]
    [|user|]{user_message_1}
    [|assistant|]{assistant_message_1}[|endofturn|]
    [|user|]{user_message_2}
    [|assistant|]<thought>
    ```

  - Example: [second-state/EXAONE-Deep-32B-GGUF](https://huggingface.co/second-state/EXAONE-Deep-32B-GGUF)

- `falcon3`
  - Prompt string

    ```text
    <|system|>
    {system_message}
    <|user|>
    {user_message}
    <|assistant|>
    ```

  - Example: [second-state/Falcon3-7B-Instruct-GGUF](https://huggingface.co/second-state/Falcon3-7B-Instruct-GGUF)

- `functionary-31`

  - Prompt string

    ```text
    <|start_header_id|>system<|end_header_id|>

    Environment: ipython

    Cutting Knowledge Date: December 2023


    You have access to the following functions:

    Use the function 'get_current_weather' to 'Get the current weather'
    {"name":"get_current_weather","description":"Get the current weather","parameters":{"type":"object","properties":{"location":{"type":"string","description":"The city and state, e.g. San Francisco, CA"}},"required":["location"]}}


    Think very carefully before calling functions.
    If a you choose to call a function ONLY reply in the following format:
    <{start_tag}={function_name}>{parameters}{end_tag}
    where

    start_tag => `<function`
    parameters => a JSON dict with the function argument name as key and function argument value as value.
    end_tag => `</function>`

    Here is an example,
    <function=example_function_name>{"example_name": "example_value"}</function>

    Reminder:
    - If looking for real time information use relevant functions before falling back to brave_search
    - Function calls MUST follow the specified format, start with <function= and end with </function>
    - Required parameters MUST be specified
    - Only call one function at a time
    - Put the entire function call reply on one line

    <|eot_id|><|start_header_id|>user<|end_header_id|>

    What is the weather like in Beijing today?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    ```

  - Example: [second-state/functionary-small-v3.1-GGUF](https://huggingface.co/second-state/functionary-small-v3.1-GGUF)

- `functionary-32`

  - Prompt string

    ```text
    <|start_header_id|>system<|end_header_id|>

    You are capable of executing available function(s) if required.
    Only execute function(s) when absolutely necessary.
    Ask for the required input to:recipient==all
    Use JSON for function arguments.
    Respond in this format:
    >>>${recipient}
    ${content}
    Available functions:
    // Supported function definitions that should be called when necessary.
    namespace functions {

        // Get the current weather
        type get_current_weather = (_: {

            // The city and state, e.g. San Francisco, CA
            location: string,

        }) => any;


    } // namespace functions<|eot_id|><|start_header_id|>user<|end_header_id|>

    What is the weather like in Beijing today?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    ```

  - Example: [second-state/functionary-small-v3.2-GGUF](https://huggingface.co/second-state/functionary-small-v3.2-GGUF)

- `gemma-3`
  - Prompt string

    ```text
    <bos><start_of_turn>user
    {user_message_1}<end_of_turn>
    <start_of_turn>model
    {model_message_1}<end_of_turn>model
    <start_of_turn>user
    {system_prompt}
    {user_message_2}<end_of_turn>
    <start_of_turn>model
    ```

  - Example: [second-state/gemma-3-27b-it-GGUF](https://huggingface.co/second-state/gemma-3-27b-it-GGUF)

- `gemma-instruct`
  - Prompt string
    ```text
    <bos><start_of_turn>user
    {user_message}<end_of_turn>
    <start_of_turn>model
    {model_message}<end_of_turn>model
    ```

  - Example: [second-state/gemma-2-27b-it-GGUF](https://huggingface.co/second-state/gemma-2-27b-it-GGUF)

- `glm-4-chat`
  - Prompt string

    ```text
    [gMASK]<|system|>
    {system_message}<|user|>
    {user_message_1}<|assistant|>
    {assistant_message_1}
    ```

  - Example: [second-state/glm-4-9b-chat-GGUF](https://huggingface.co/second-state/glm-4-9b-chat-GGUF)

- `gpt-oss`
  - Prompt string

    ```text
    <|start|>system<|message|>
    You are ChatGPT, a large language model trained by OpenAI.
    Knowledge cutoff: 2024-06
    Current date: 2025-08-06
    Reasoning: medium
    # Valid channels: analysis, commentary, final. Channel must be included for every message.
    <|end|>

    <|start|>user<|message|>Hello!<|end|>
    <|start|>assistant<|channel|>final<|message|>Hi there!<|end|>
    <|start|>user<|message|>What's your favorite color?<|end|>
    <|start|>assistant
    ```

  - Example: [second-state/gpt-oss-20b-GGUF](https://huggingface.co/second-state/gpt-oss-20b-GGUF)

- `human-assistant`
  - Prompt string

    ```text
    Human: {input_1}\n\nAssistant:{output_1}Human: {input_2}\n\nAssistant:
    ```

  - Example: [second-state/OrionStar-Yi-34B-Chat-Llama-GGUF](https://huggingface.co/second-state/OrionStar-Yi-34B-Chat-Llama-GGUF)

- `intel-neural`
  - Prompt string

    ```text
    ### System:
    {system}
    ### User:
    {usr}
    ### Assistant:
    ```

  - Example: [second-state/Neural-Chat-7B-v3-3-GGUF](https://huggingface.co/second-state/Neural-Chat-7B-v3-3-GGUF)

- `llama-2-chat`
  - Prompt string

    ```text
    <s>[INST] <<SYS>>
    {system_message}
    <</SYS>>

    {user_message_1} [/INST] {assistant_message} </s><s>[INST] {user_message_2} [/INST]
    ```

- `llama-3-chat`
  - Prompt string

    ```text
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    {{ system_prompt }}<|eot_id|><|start_header_id|>user<|end_header_id|>

    {{ user_message_1 }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    {{ model_answer_1 }}<|eot_id|><|start_header_id|>user<|end_header_id|>

    {{ user_message_2 }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    ```

- `llama-3-tool`
  - Prompt string

    ```text
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    {system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>

    Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.

    Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}. Do not use variables.

    [{"type":"function","function":{"name":"get_current_weather","description":"Get the current weather in a given location","parameters":{"type":"object","properties":{"location":{"type":"string","description":"The city and state, e.g. San Francisco, CA"},"unit":{"type":"string","description":"The temperature unit to use. Infer this from the users location.","enum":["celsius","fahrenheit"]}},"required":["location","unit"]}}}]

    Question: {user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    ```

- `llama-4-chat`
  - Prompt string

    - Chat

      ```text
      <|begin_of_text|><|header_start|>system<|header_end|>

      {system_message}<|eot|>
      <|header_start|>user<|header_end|>

      {user_message_1}<|eot|>
      <|header_start|>assistant<|header_end|>

      {assistant_message_1}<|eot|>
      <|header_start|>user<|header_end|>

      {user_message_2}<|eot|><|header_start|>assistant<|header_end|>
      ```

    - Tool use

      - Prompt with user question and tool info

        <details> <summary> Expand to see the example </summary>

        ```text
        <|begin_of_text|><|header_start|>system<|header_end|>

        You are a helpful assistant.<|eot|>
        <|header_start|>user<|header_end|>

        Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.

        Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.
        Do not use variables.

        [
          {
            "type": "function",
            "function": {
              "name": "get_current_weather",
              "description": "Get the current weather in a given location",
              "parameters": {
                "type": "object",
                "properties": {
                  "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                  },
                  "unit": {
                    "type": "string",
                    "description": "The temperature unit to use. Infer this from the users location.",
                    "enum": [
                      "celsius",
                      "fahrenheit"
                    ]
                  }
                },
                "required": [
                  "location",
                  "unit"
                ]
              }
            }
          },
          {
            "type": "function",
            "function": {
              "name": "predict_weather",
              "description": "Predict the weather in 24 hours",
              "parameters": {
                "type": "object",
                "properties": {
                  "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                  },
                  "unit": {
                    "type": "string",
                    "description": "The temperature unit to use. Infer this from the users location.",
                    "enum": [
                      "celsius",
                      "fahrenheit"
                    ]
                  }
                },
                "required": [
                  "location",
                  "unit"
                ]
              }
            }
          },
          {
            "type": "function",
            "function": {
              "name": "sum",
              "description": "Calculate the sum of two numbers",
              "parameters": {
                "type": "object",
                "properties": {
                  "a": {
                    "type": "integer",
                    "description": "the left hand side number"
                  },
                  "b": {
                    "type": "integer",
                    "description": "the right hand side number"
                  }
                },
                "required": [
                  "a",
                  "b"
                ]
              }
            }
          }
        ]

        Question: How is the weather of Beijing, China in celsius?<|eot|><|header_start|>assistant<|header_end|>
        ```

        </details>

      - Prompt with tool results

        ```text
        <|begin_of_text|><|header_start|>system<|header_end|>

        You are a helpful assistant.<|eot|>
        <|header_start|>user<|header_end|>

        How is the weather of Beijing, China in celsius?<|eot|>
        <|header_start|>assistant<|header_end|>

        {"name":"get_current_weather","arguments":"{\"location\":\"Beijing, China\",\"unit\":\"celsius\"}"}<|eot|>
        <|header_start|>ipython<|header_end|>

        {"temperature":"30","unit":"celsius"}<|eot|><|header_start|>assistant<|header_end|>
        ```

  - Example: [second-state/Llama-4-Scout-17B-16E-Instruct-GGUF](https://huggingface.co/second-state/Llama-4-Scout-17B-16E-Instruct-GGUF)

- `mediatek-breeze`
  - Prompt string

    ```text
    <s>{system_message}  [INST] {user_message_1} [/INST] {assistant_message_1} [INST] {user_message_2} [/INST]
    ```

  - Example: [second-state/Breeze-7B-Instruct-v1_0-GGUF](https://huggingface.co/second-state/Breeze-7B-Instruct-v1_0-GGUF)

- `megrez`
  - Prompt string

    ```text
    <|role_start|>system<|role_end|>{system_message}<|turn_end|><|role_start|>user<|role_end|>{user_message}<|turn_end|><|role_start|>assistant<|role_end|>
    ```

  - Example: [second-state/Megrez-3B-Instruct-GGUF](https://huggingface.co/second-state/Megrez-3B-Instruct-GGUF)

- `mistral-instruct`
  - Prompt string

    ```text
    <s>[INST] {user_message_1} [/INST]{assistant_message_1}</s>[INST] {user_message_2} [/INST]{assistant_message_2}</s>
    ```

  - Example: [second-state/Mistral-7B-Instruct-v0.3-GGUF](https://huggingface.co/second-state/Mistral-7B-Instruct-v0.3-GGUF)

- `mistrallite`
  - Prompt string

    ```text
    <|prompter|>{user_message}</s><|assistant|>{assistant_message}</s>
    ```

  - Example: [second-state/MistralLite-7B-GGUF](https://huggingface.co/second-state/MistralLite-7B-GGUF)

- `mistral-small-chat`
  - Prompt string

    ```text
    <s>[SYSTEM_PROMPT]<system prompt>[/SYSTEM_PROMPT][INST]<user message>[/INST]<assistant response></s>[INST]<user message>[/INST]
    ```

  - Example: [second-state/Mistral-Small-24B-Instruct-2501-GGUF](https://huggingface.co/second-state/Mistral-Small-24B-Instruct-2501-GGUF)

- `mistral-small-tool`
  - Prompt string

    ```text
    <s>[INST] {user_message_1}[/INST][TOOL_CALLS] [{tool_call_1},{tool_call_2}]</s>[TOOL_RESULTS] {tool_result_1}[/TOOL_RESULTS] {assistant_message_1}</s>[AVAILABLE_TOOLS] [{tool_1},{tool_2}][/AVAILABLE_TOOLS][INST] {system_message}<0x0A><0x0A>{user_message_2}[/INST]
    ```

    - Example

      ```text
      [INST] What is the weather like in San Francisco in Celsius?[/INST][TOOL_CALLS] [{"id":"call_abc123","type":"function","function":{"name":"get_current_weather","arguments":"{\"location\":\"San Francisco, CA\",\"unit\":\"celsius\"}"}}]</s>[TOOL_RESULTS] {"id":"call_abc123","content":{"temperature":"30","unit":"celsius"}}[/TOOL_RESULTS] The current temperature in San Francisco is 30°C.</s>[INST] What is the weather like in San Francisco in the coming 24 hours in Celsius?[/INST][TOOL_CALLS] [{"id":"call_abc123","type":"function","function":{"name":"predict_weather","arguments":"{}"}}]</s>[TOOL_RESULTS] {"id":"call_abc124","content":{"temperature":"25","unit":"celsius"}}[/TOOL_RESULTS]
      ```

  - Example: [second-state/Mistral-Small-24B-Instruct-2501-GGUF](https://huggingface.co/second-state/Mistral-Small-24B-Instruct-2501-GGUF)

- `mistral-tool`
  - Prompt string

    ```text
    [INST] {user_message_1} [/INST][TOOL_CALLS] [{tool_call_1}]</s>[TOOL_RESULTS]{tool_result_1}[/TOOL_RESULTS]{assistant_message_1}</s>[AVAILABLE_TOOLS] [{tool_1},{tool_2}][/AVAILABLE_TOOLS][INST] {user_message_2} [/INST]
    ```

    - Example

      ```text
      [INST] Hey! What is the weather like in Beijing and Tokyo? [/INST][TOOL_CALLS] [{"name":"get_current_weather","arguments":{"location": "Beijing, CN", "format": "celsius"}}]</s>[TOOL_RESULTS]Fine, with a chance of showers.[/TOOL_RESULTS]Today in Auckland, the weather is expected to be partly cloudy with a high chance of showers. Be prepared for possible rain and carry an umbrella if you're venturing outside. Have a great day!</s>[AVAILABLE_TOOLS] [{"type":"function","function":{"name":"get_current_weather","description":"Get the current weather in a given location","parameters":{"type":"object","properties":{"location":{"type":"string","description":"The city and state, e.g. San Francisco, CA"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["location"]}}},{"type":"function","function":{"name":"predict_weather","description":"Predict the weather in 24 hours","parameters":{"type":"object","properties":{"location":{"type":"string","description":"The city and state, e.g. San Francisco, CA"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["location"]}}}][/AVAILABLE_TOOLS][INST] What is the weather like in Beijing now?[/INST]
      ```

  - Example: [second-state/Mistral-7B-Instruct-v0.3-GGUF](https://huggingface.co/second-state/Mistral-7B-Instruct-v0.3-GGUF)

- `moxin-chat`

  ```text
  <s> [INST] {system_message}

  {user_message_1} [/INST] {assistant_message_1}</s> [INST] {user_message_2} [/INST]
  ```

  - Example: [second-state/moxin-chat-7b-GGUF](https://huggingface.co/second-state/moxin-chat-7b-GGUF)

- `moxin-instruct`

  ```text
  <|system|>
  {system_message}
  <|user|>
  {user_message_1}
  <|assistant|>
  {assistant_message_1}<|endoftext|>
  <|user|>
  {user_message_2}
  <|assistant|>
  ```

  - Example: [second-state/moxin-reasoning-7b-GGUF](https://huggingface.co/second-state/moxin-reasoning-7b-GGUF)

- `nemotron-chat`

  ```text
  <extra_id_0>System
  {system_message}
  <extra_id_1>User
  {user_message_1}<extra_id_1>Assistant
  {assistant_message_1}
  <extra_id_1>User
  {user_message_2}<extra_id_1>Assistant
  {assistant_message_2}
  <extra_id_1>User
  {user_message_3}
  <extra_id_1>Assistant\n
  ```

  - Example: [second-state/Nemotron-Mini-4B-Instruct-GGUF](https://huggingface.co/second-state/Nemotron-Mini-4B-Instruct-GGUF)

- `nemotron-tool`

  ```text
  <extra_id_0>System
  {system_message}
  <tool> {tool_1} </tool>
  <tool> {tool_2} </tool>


  <extra_id_1>User
  {user_message_1}<extra_id_1>Assistant
  <toolcall> {tool_call_message} </toolcall>
  <extra_id_1>Tool
  {tool_result_message}
  <extra_id_1>Assistant\n
  ```

  - Example: [second-state/Nemotron-Mini-4B-Instruct-GGUF](https://huggingface.co/second-state/Nemotron-Mini-4B-Instruct-GGUF)

- `octopus`
  - Prompt string

    ```text
    {system_prompt}\n\nQuery: {input_text} \n\nResponse:
    ```

  - Example: [second-state/Octopus-v2-GGUF](https://huggingface.co/second-state/Octopus-v2-GGUF)

- `openchat`
  - Prompt string

    ```text
    GPT4 User: {prompt}<|end_of_turn|>GPT4 Assistant:
    ```

  - Example: [second-state/OpenChat-3.5-0106-GGUF](https://huggingface.co/second-state/OpenChat-3.5-0106-GGUF)

- `phi-2-instruct`
  - Prompt string

    ```text
    Instruct: <prompt>\nOutput:
    ```

  - Example: [second-state/phi-2-GGUF](https://huggingface.co/second-state/phi-2-GGUF)

- `phi-3-chat`
  - Prompt string

    ```text
    <|system|>
    {system_message}<|end|>
    <|user|>
    {user_message_1}<|end|>
    <|assistant|>
    {assistant_message_1}<|end|>
    <|user|>
    {user_message_2}<|end|>
    <|assistant|>
    ```

  - Example: [second-state/Phi-3-medium-4k-instruct-GGUF](https://huggingface.co/second-state/Phi-3-medium-4k-instruct-GGUF)

- `phi-4-chat`
  - Prompt string

    ```text
    <|im_start|>system<|im_sep|>
    {system_message}<|im_end|>
    <|im_start|>user<|im_sep|>
    {user_message}<|im_end|>
    <|im_start|>assistant<|im_sep|>
    ```

  - Example: [second-state/phi-4-GGUF](https://huggingface.co/second-state/phi-4-GGUF)

- `qwen3-no-think`
  - Prompt string

    ```text
    <|im_start|>system
    You are a helpful assistant. Answer questions as concisely as possible.<|im_end|>
    <|im_start|>user
    {user_message_1}<|im_end|>
    <|im_start|>assistant
    {assistant_message_1}<|im_end|>
    <|im_start|>user
    {user_message_2}<|im_end|>
    <|im_start|>assistant
    <think>

    </think>
    ```

  - Example: [second-state/Qwen3-4B-GGUF](https://huggingface.co/second-state/Qwen3-4B-GGUF)

- `seed-instruct`
  - Prompt string

    ```text
    <[begin▁of▁sentence]>system
    {system_message}

    <[end▁of▁sentence]><[begin▁of▁sentence]>user
    {user_message_1}<[end▁of▁sentence]><[begin▁of▁sentence]>assistant
    {assistant_message_1}<[end▁of▁sentence]><[begin▁of▁sentence]>user
    {user_message_2}<[end▁of▁sentence]><[begin▁of▁sentence]>assistant
    ```

  - Example: [second-state/Seed-Coder-8B-Instruct-GGUF](https://huggingface.co/second-state/Seed-Coder-8B-Instruct-GGUF)

- `seed-oss-think` and `seed-oss-no-think`
  - Prompt string (`seed-oss-think`)

    ```text
    <seed:bos>system
    You are Doubao, a helpful AI assistant.
    <seed:eos>

    <seed:bos>user
    {user_message_1}
    <seed:eos>

    <seed:bos>assistant
    <seed:think>{thinking_content}</seed:think>
    {assistant_message_1}
    <seed:eos>

    <seed:bos>user
    {user_message_2}
    <seed:eos>

    <seed:bos>assistant
    ```

  - Prompt string (`seed-oss-no-think`)

    ```text
    <seed:bos>system
    You are Doubao, a helpful AI assistant.
    <seed:eos>

    <seed:bos>system
    You are an intelligent assistant that can answer questions in one step without the need for reasoning and thinking, that is, your     thinking budget is 0. Next, please skip       the thinking process and directly start answering the user's questions.
    <seed:eos>

    <seed:bos>user
    {user_message_1}
    <seed:eos>

    <seed:bos>assistant
    {assistant_message_1}
    <seed:eos>

    <seed:bos>user
    {user_message_2}
    <seed:eos>

    <seed:bos>assistant
    ```

  - Example: [second-state/Seed-OSS-36B-Instruct-GGUF](https://huggingface.co/second-state/Seed-OSS-36B-Instruct-GGUF)

- `seed-reasoning`
  - Prompt string

    ```text
    <[begin▁of▁sentence]>user
    {user_message_1}<[end▁of▁sentence]><[begin▁of▁sentence]>assistant
    {assistant_message_1}<[end▁of▁sentence]><[begin▁of▁sentence]>user
    {user_message_2}<[end▁of▁sentence]><[begin▁of▁sentence]>assistant
    ```

  - Example: [second-state/Seed-Coder-8B-Reasoning-GGUF](https://huggingface.co/second-state/Seed-Coder-8B-Reasoning-GGUF)

- `smol-vision`
  - Prompt string

    ```text
    <|im_start|>
    User: {user_message_1}<image>
    Assistant: {assistant_message_1}
    User: {user_message_2}
    Assistant:
    ```

  - Example: [second-state/SmolVLM2-2.2B-Instruct-GGUF](https://huggingface.co/second-state/SmolVLM2-2.2B-Instruct-GGUF)

- `stablelm-zephyr`
  - Prompt string

    ```text
    <|user|>
    {prompt}<|endoftext|>
    <|assistant|>
    ```

  - Example: [second-state/stablelm-2-zephyr-1.6b-GGUF](https://huggingface.co/second-state/stablelm-2-zephyr-1.6b-GGUF)

- `vicuna-1.0-chat`
  - Prompt string

    ```text
    {system} USER: {prompt} ASSISTANT:
    ```

  - Example: [second-state/Wizard-Vicuna-13B-Uncensored-GGUF](https://huggingface.co/second-state/Wizard-Vicuna-13B-Uncensored-GGUF)

- `vicuna-1.1-chat`
  - Prompt string

    ```text
    USER: {prompt}
    ASSISTANT:
    ```

  - Example: [second-state/ChatAllInOne-Yi-34B-200K-V1-GGUF](https://huggingface.co/second-state/ChatAllInOne-Yi-34B-200K-V1-GGUF)

- `vicuna-llava`
  - Prompt string

    ```text
    <system_prompt>\nUSER:<image_embeddings>\n<textual_prompt>\nASSISTANT:
    ```

  - Example: [second-state/Llava-v1.6-Vicuna-7B-GGUF](https://huggingface.co/second-state/Llava-v1.6-Vicuna-7B-GGUF)

- `wizard-coder`
  - Prompt string

    ```text
    {system}

    ### Instruction:
    {instruction}

    ### Response:
    ```

  - Example: [second-state/WizardCoder-Python-7B-v1.0-GGUF](https://huggingface.co/second-state/WizardCoder-Python-7B-v1.0-GGUF)

- `zephyr`
  - Prompt string

    ```text
    <|system|>
    {system_prompt}</s>
    <|user|>
    {prompt}</s>
    <|assistant|>
    ```

  - Example: [second-state/Zephyr-7B-Beta-GGUF](https://huggingface.co/second-state/Zephyr-7B-Beta-GGUF)
