# CHAT PROMPTS

## Prompt Types

- `llama-2-chat`

```text
<s>[INST] <<SYS>>
{{ system_prompt }}
<</SYS>>

{{ user_msg_1 }} [/INST] {{ model_answer_1 }} </s><s>[INST] {{ user_msg_2 }}   [/INST]
```

- `chatml`

```text
<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
```

- `openchat`

```text
GPT4 User: {prompt}<|end_of_turn|>GPT4 Assistant:
```

- `zephyr`

```text
<|system|>
{system_prompt}</s>
<|user|>
{prompt}</s>
<|assistant|>
```

- `codellama-instruct`

```text
[INST] Write code to solve the following coding problem that obeys the constraints and passes the example test cases. Please wrap your code answer using ```:
{prompt}
[/INST]
```

- `mistral-instruct`

```text
<s>[INST] {prompt} [/INST]
```

- `mistrallite`

```text
<|prompter|>{prompt}</s><|assistant|>
```

- `vicuna-chat`

```text
{system} USER: {prompt} ASSISTANT:
```

- `vicuna-1.1-chat`

```text
USER: {prompt}
ASSISTANT:
```

- `wizard-coder`

```text
{system}

### Instruction:
{instruction}

### Response:
```

- `intel-neural`

```text
### System:
{system}
### User:
{usr}
### Assistant:
```

- `deepseek-chat`

```text
User: {user_message_1}

Assistant: {assistant_message_1}<｜end▁of▁sentence｜>User: {user_message_2}

Assistant:
```

- `deepseek-coder`

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

- `solar-instruct`

```text
### User:
{prompt}

### Assistant:
```

- `belle-llama-2-chat`

```text
Human: {input_1}\n\nAssistant:{output_1}Human: {input_2}\n\nAssistant:
```
