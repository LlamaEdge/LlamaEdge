use super::BuildChatPrompt;
use crate::error::{PromptError, Result};
use endpoints::chat::{
    ChatCompletionAssistantMessage, ChatCompletionRequestMessage, ChatCompletionToolMessage,
    ChatCompletionUserMessage, ChatCompletionUserMessageContent, ContentPart, Tool,
};

const DEFAULT_GPT_OSS_SYSTEM_MESSAGE: &str = r#"You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: 2025-08-06"#;

const DEFAULT_GPT_OSS_SYSTEM_PROMPT: &str = r#"{system_message}

Reasoning: medium

# Valid channels: analysis, commentary, final. Channel must be included for every message.
"#;

const DEFAULT_GPT_OSS_SYSTEM_PROMPT_WITH_TOOLS: &str = r#"{system_message}

Reasoning: medium

# Tools
You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools}
</tools>

# Tool Call Format Requirement

When calling a function, you must output the function call **as a single-line compact JSON object**, wrapped inside a markdown code block with language `json`.

The JSON must contain:
- `"name"`: the function name
- `"arguments"`: a JSON object with function arguments

✅ The entire output must look exactly like this format:

```json
{"name":"sum","arguments":{"a":3,"b":5}}
```

# Tool Matching Policy

Before calling any tool, you MUST determine whether the user query semantically matches one of the available tools.

A tool is considered a match ONLY IF:
- The tool's purpose (from its "description") directly applies to the user query.
- All required input parameters for the tool are either explicitly present in the user query or can be reasonably inferred.
- The user's intent clearly aligns with the tool's function, and the tool is the correct way to fulfill the request.

If no tool matches the query, do NOT call any tool. Simply answer the user's question using natural language.

# Tool Usage Policy

- If a tool matches the user query (according to the matching policy), you MUST call the tool instead of replying directly.
- If no tool matches the query, answer the user naturally and directly.
- Do NOT mention tools, tool availability, or tool matching in your response.
- Do NOT say things like “this cannot be answered using the available tools” or “I will answer directly instead.”
- Only output the tool call if applicable, in the required format. Otherwise, provide a normal response.

# Valid channels: analysis, commentary, final. Channel must be included for every message.
"#;

/// Generate prompts for the models using ChatML template.
#[derive(Debug, Default, Clone)]
pub struct GptOssPrompt;
impl GptOssPrompt {
    /// Create a user prompt from a chat completion request message.
    fn append_user_message(
        &self,
        chat_history: impl AsRef<str>,
        system_prompt: impl AsRef<str>,
        message: &ChatCompletionUserMessage,
    ) -> String {
        let content = match message.content() {
            ChatCompletionUserMessageContent::Text(text) => text.to_string(),
            ChatCompletionUserMessageContent::Parts(parts) => {
                let mut content = String::new();
                for part in parts {
                    if let ContentPart::Text(text_content) = part {
                        content.push_str(text_content.text());
                        content.push('\n');
                    }
                }
                content
            }
        };

        match chat_history.as_ref().is_empty() {
            true => match system_prompt.as_ref().is_empty() {
                true => {
                    format!(
                        "<|im_start|>user<|message|>{user_message}<|im_end|>",
                        user_message = content.trim(),
                    )
                }
                false => {
                    format!(
                        "{system_prompt}\n\n<|im_start|>user<|message|>{user_message}<|im_end|>",
                        system_prompt = system_prompt.as_ref().trim(),
                        user_message = content.trim(),
                    )
                }
            },
            false => format!(
                "{chat_history}\n<|im_start|>user<|message|>{user_message}<|im_end|>",
                chat_history = chat_history.as_ref().trim(),
                user_message = content.trim(),
            ),
        }
    }

    /// create an assistant prompt from a chat completion request message.
    fn append_assistant_message(
        &self,
        chat_history: impl AsRef<str>,
        message: &ChatCompletionAssistantMessage,
    ) -> Result<String> {
        let content = match message.content() {
            Some(content) => content.to_string(),
            // Note that the content is optional if `tool_calls` is specified.
            None => match message.tool_calls().is_some() {
                true => String::new(),
                false => return Err(PromptError::NoAssistantMessage),
            },
        };

        Ok(format!(
            "{chat_history}\n<|im_start|>assistant<|channel|>final<|message|>{assistant_message}<|im_end|>",
            chat_history = chat_history.as_ref().trim(),
            assistant_message = content.trim(),
        ))
    }

    /// create a tool prompt from a chat completion request message.
    fn append_tool_message(
        &self,
        chat_history: impl AsRef<str>,
        message: &ChatCompletionToolMessage,
    ) -> String {
        format!(
            "{chat_history}\n<|im_start|>tool\n<tool_response>\n{tool_message}\n</tool_response>\n<|im_end|>",
            chat_history = chat_history.as_ref().trim(),
            tool_message = message.content().trim(),
        )
    }
}
impl BuildChatPrompt for GptOssPrompt {
    fn build(&self, messages: &mut Vec<ChatCompletionRequestMessage>) -> Result<String> {
        if messages.is_empty() {
            return Err(crate::error::PromptError::NoMessages);
        }

        // system prompt
        let system_message = match messages[0] {
            ChatCompletionRequestMessage::System(ref message) => {
                let content = message.content();
                match content.is_empty() {
                    true => DEFAULT_GPT_OSS_SYSTEM_PROMPT
                        .replace("{system_message}", DEFAULT_GPT_OSS_SYSTEM_MESSAGE),
                    false => DEFAULT_GPT_OSS_SYSTEM_PROMPT.replace("{system_message}", content),
                }
            }
            _ => DEFAULT_GPT_OSS_SYSTEM_PROMPT
                .replace("{system_message}", DEFAULT_GPT_OSS_SYSTEM_MESSAGE),
        };

        let system_prompt = format!("<|im_start|>system<|message|>\n{system_message}\n<|im_end|>");

        // append user/assistant messages
        let mut prompt = String::new();
        for message in messages {
            match message {
                ChatCompletionRequestMessage::User(message) => {
                    prompt = self.append_user_message(&prompt, &system_prompt, message);
                }
                ChatCompletionRequestMessage::Assistant(message) => {
                    prompt = self.append_assistant_message(&prompt, message)?;
                }
                _ => continue,
            }
        }

        prompt.push_str("\n<|im_start|>assistant");

        Ok(prompt)
    }

    fn build_with_tools(
        &self,
        messages: &mut Vec<ChatCompletionRequestMessage>,
        tools: Option<&[Tool]>,
    ) -> Result<String> {
        if messages.is_empty() {
            return Err(crate::error::PromptError::NoMessages);
        }

        // system prompt
        let system_message = match &messages[0] {
            ChatCompletionRequestMessage::System(message) => match tools {
                Some(tools) if !tools.is_empty() => {
                    let available_tools = serde_json::to_string(tools).unwrap();

                    let content = message.content();
                    match content.is_empty() {
                        true => DEFAULT_GPT_OSS_SYSTEM_PROMPT_WITH_TOOLS
                            .replace("{system_message}", DEFAULT_GPT_OSS_SYSTEM_MESSAGE)
                            .replace("{tools}", &available_tools),
                        false => DEFAULT_GPT_OSS_SYSTEM_PROMPT_WITH_TOOLS
                            .replace("{system_message}", content)
                            .replace("{tools}", &available_tools),
                    }
                }
                _ => {
                    let content = message.content();
                    match content.is_empty() {
                        true => DEFAULT_GPT_OSS_SYSTEM_PROMPT
                            .replace("{system_message}", DEFAULT_GPT_OSS_SYSTEM_MESSAGE),
                        false => DEFAULT_GPT_OSS_SYSTEM_PROMPT.replace("{system_message}", content),
                    }
                }
            },
            _ => match tools {
                Some(tools) if !tools.is_empty() => {
                    let available_tools = serde_json::to_string_pretty(tools).unwrap();

                    DEFAULT_GPT_OSS_SYSTEM_PROMPT_WITH_TOOLS
                        .replace("{system_message}", DEFAULT_GPT_OSS_SYSTEM_MESSAGE)
                        .replace("{tools}", &available_tools)
                }
                _ => DEFAULT_GPT_OSS_SYSTEM_PROMPT
                    .replace("{system_message}", DEFAULT_GPT_OSS_SYSTEM_MESSAGE),
            },
        };

        let system_prompt = format!("<|im_start|>system<|message|>\n{system_message}\n<|im_end|>");

        // append user/assistant messages
        let mut prompt = String::new();
        for message in messages {
            match message {
                ChatCompletionRequestMessage::User(message) => {
                    prompt = self.append_user_message(&prompt, &system_prompt, message);
                }
                ChatCompletionRequestMessage::Assistant(message) => {
                    prompt = self.append_assistant_message(&prompt, message)?;
                }
                ChatCompletionRequestMessage::Tool(message) => {
                    prompt = self.append_tool_message(&prompt, message);
                }
                _ => continue,
            }
        }

        prompt.push_str("\n<|im_start|>assistant");

        Ok(prompt)
    }
}
