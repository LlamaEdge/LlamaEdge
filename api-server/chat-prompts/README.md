# Prompt templat table

You can learn the prompt template name and its corresponding string format, and models that LlamaEdge supports from the following table.

<table>
  
<tr>
<td>Prompt Template Name</td>
<td>Format</td>
<td>Models using this template</td>
</tr>

<tr>
<td>llama-2-chat</td>
<td>
  
```
<s>[INST] <<SYS>>
{{ system_prompt }}
<</SYS>>

{{ user_msg_1 }} [/INST] {{ model_answer_1 }} </s><s>[INST] {{ user_msg_2 }}   [/INST]
```

</td>
<td>Llama-2-7B-Chat, Llama-2-13B-Chat</td>
</tr>
    
<tr>
<td>chatml</td>
<td>
        
```text
<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
```
</td>
<td>TinyLlama-1.1B-Chat, i-34B-Chat, OpenHermes-2.5-Mistral-7B, Qwen, Dolphin-2.2-Yi-34B, Dolphin-2.6-Mistral-7B, Samantha-1.2-Mistral-7B, Orca-2-13B, Nous-Hermes-2-Mixtral-8x7B-DPO</td>
</tr>

<tr>
<td>openchat</td>
<td>
  
```text
GPT4 User: {prompt}<|end_of_turn|>GPT4 Assistant:
```

</td>
<td>OpenChat-3.5 series of models</td>
</tr>

<tr>
<td>zephyr</td>
<td>
  
```text
<|system|>
{system_prompt}</s>
<|user|>
{prompt}</s>
<|assistant|>
```
</td>
<td>Zephyr-7B-Alpha</td>
</tr>

<tr>
<td>codellama-instruct</td>
<td>

```text
[INST] Write code to solve the following coding problem that obeys the constraints and passes the example test cases. Please wrap your code answer using ```:
{prompt}
[/INST]
```

</td>
<td>Code-llama</td>
</tr>
    
<tr>
<td>mistral-instruct</td>
<td>

```text
<s>[INST] {prompt} [/INST]
```

</td>
<td>Mistral-7B-Instruct-v0.1, Mistral-7B-Instruct-v0.2, Mixtral-8x7B-Instruct-v0.1, miqu-2-70b</td>
</tr>

<tr>
<td>stabllm-zephyr</td>
<td>

```text
<|user|>
{prompt}<|endoftext|>
<|assistant|>
```

</td>
<td>stablelm-2-zephyr-1.6b</td>
</tr>

<tr>
<td>mistrallite</td>
<td>

```text
<|prompter|>{prompt}</s><|assistant|>
```

</td>
<td>MistralLite-7B</td>
</tr>

<tr>
<td>vicuna-chat</td>
<td>
  
```text
{system} USER: {prompt} ASSISTANT:
```
  
</td>
<td>Wizard-Vicuna-13B-Uncensored, Samantha-1.11-CodeLlama-34B, WizardLM-13B-V1.0-Uncensored</td>
</tr>

<tr>
<td>vicuna-1.1-chat</td>
<td>

```text
USER: {prompt}
ASSISTANT:
```

</td>
<td>CALM2-7B-Chat, Samantha-1.2-Mistral-7b</td>
</tr>

<tr>
<td>wizard-coder</td>
<td>

```text
{system}

### Instruction:
{instruction}

### Response:
```

</td>
<td>WizardCoder-Python-7B-V1.0</td>
</tr>

<tr>
<td>deepseek-chat</td>
<td>

```text
User: {user_message_1}

Assistant: {assistant_message_1}<｜end▁of▁sentence｜>User: {user_message_2}

Assistant:
```

</td>
<td>DeepSeek-LLM-7B-Chat</td>
</tr>

<tr>
<td>deepseek-coder</td>
<td>
  
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

</td>
<td>DeepSeek-Coder-6.7B</td>
</tr>

<tr>
<td>solar-instruct</td>
<td>

```text
### User:
{prompt}

### Assistant:
```

</td>
<td>SOLAR-10.7B-Instruct-v1.0</td>
</tr>

<tr>
<td>intel-neural</td>
<td>

```text
### System:
{system}
### User:
{usr}
### Assistant:
```
  
</td>
<td>Intel Nerual series of models</td>
</tr>

<tr>
<td>belle-llama-2-chat</td>
<td>

```text
Human: {input_1}\n\nAssistant:{output_1}Human: {input_2}\n\nAssistant:
```
  
</td>
<td>Belle-Llama-2-Chat</td>
</tr>

</table>
