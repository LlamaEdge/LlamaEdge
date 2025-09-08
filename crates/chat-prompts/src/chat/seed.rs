use crate::{
    error::{PromptError, Result},
    BuildChatPrompt,
};
use endpoints::chat::{
    ChatCompletionAssistantMessage, ChatCompletionRequestMessage, ChatCompletionSystemMessage,
    ChatCompletionToolMessage, ChatCompletionUserMessage, ChatCompletionUserMessageContent,
    ContentPart, Tool,
};

/// Generate prompts for the `ByteDance/Seed-instruct` models
#[derive(Debug, Default, Clone)]
pub struct SeedInstructPrompt;
impl SeedInstructPrompt {
    /// Create a system prompt from a chat completion request message.
    fn create_system_prompt(&self, message: &ChatCompletionSystemMessage) -> String {
        let content = message.content();
        match content.is_empty() {
            true => String::from("<[begin▁of▁sentence]>system\nYou are an AI programming assistant, utilizing the Seed-Coder model, developed by ByteDance Seed, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.\n\n<[end▁of▁sentence]>"),
            false => format!("<[begin▁of▁sentence]>system\n{content}\n\n<[end▁of▁sentence]>"),
        }
    }

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
                        "<[begin▁of▁sentence]>user\n{user_message}<[end▁of▁sentence]>",
                        user_message = content.trim(),
                    )
                }
                false => {
                    format!(
                        "{system_prompt}<[begin▁of▁sentence]>user\n{user_message}<[end▁of▁sentence]>",
                        system_prompt = system_prompt.as_ref().trim(),
                        user_message = content.trim(),
                    )
                }
            },
            false => format!(
                "{chat_history}<[begin▁of▁sentence]>user\n{user_message}<[end▁of▁sentence]>",
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
            "{chat_history}<[begin▁of▁sentence]>assistant\n{assistant_message}<[end▁of▁sentence]>",
            chat_history = chat_history.as_ref().trim(),
            assistant_message = content.trim(),
        ))
    }
}
impl BuildChatPrompt for SeedInstructPrompt {
    fn build(&self, messages: &mut Vec<ChatCompletionRequestMessage>) -> Result<String> {
        if messages.is_empty() {
            return Err(crate::error::PromptError::NoMessages);
        }

        // system prompt
        let system_prompt = match messages[0] {
            ChatCompletionRequestMessage::System(ref message) => self.create_system_prompt(message),
            _ => String::from("<[begin▁of▁sentence]>system\nYou are an AI programming assistant, utilizing the Seed-Coder model, developed by ByteDance Seed, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.\n\n<[end▁of▁sentence]>"),
        };

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

        prompt.push_str("<[begin▁of▁sentence]>assistant");

        Ok(prompt)
    }
}

/// Generate prompts for the `ByteDance/Seed-reasoning` models
#[derive(Debug, Default, Clone)]
pub struct SeedReasoningPrompt;
impl SeedReasoningPrompt {
    /// Create a user prompt from a chat completion request message.
    fn append_user_message(
        &self,
        chat_history: impl AsRef<str>,
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
            true => {
                format!(
                    "<[begin▁of▁sentence]>user\n{user_message}<[end▁of▁sentence]>",
                    user_message = content.trim(),
                )
            }
            false => format!(
                "{chat_history}<[begin▁of▁sentence]>user\n{user_message}<[end▁of▁sentence]>",
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
            "{chat_history}<[begin▁of▁sentence]>assistant\n{assistant_message}<[end▁of▁sentence]>",
            chat_history = chat_history.as_ref().trim(),
            assistant_message = content.trim(),
        ))
    }
}
impl BuildChatPrompt for SeedReasoningPrompt {
    fn build(&self, messages: &mut Vec<ChatCompletionRequestMessage>) -> Result<String> {
        if messages.is_empty() {
            return Err(crate::error::PromptError::NoMessages);
        }

        // append user/assistant messages
        let mut prompt = String::new();
        for message in messages {
            match message {
                ChatCompletionRequestMessage::User(message) => {
                    prompt = self.append_user_message(&prompt, message);
                }
                ChatCompletionRequestMessage::Assistant(message) => {
                    prompt = self.append_assistant_message(&prompt, message)?;
                }
                _ => continue,
            }
        }

        prompt.push_str("<[begin▁of▁sentence]>assistant");

        Ok(prompt)
    }
}

const DEFAULT_SEED_OSS_SYSTEM_MESSAGE: &str = "You are Doubao, a helpful AI assistant.";

const DEFAULT_SEED_OSS_SYSTEM_PROMPT_WITH_TOOLS: &str = r#"{system_message}

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

"#;

/// Generate prompts for the `ByteDance/Seed-instruct` models
#[derive(Debug, Default, Clone)]
pub struct SeedOssThinkPrompt;
impl SeedOssThinkPrompt {
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
                        "<seed:bos>user\n{user_message}\n<seed:eos>",
                        user_message = content.trim(),
                    )
                }
                false => {
                    format!(
                        "{system_prompt}\n\n<seed:bos>user\n{user_message}\n<seed:eos>",
                        system_prompt = system_prompt.as_ref().trim(),
                        user_message = content.trim(),
                    )
                }
            },
            false => format!(
                "{chat_history}\n\n<seed:bos>user\n{user_message}\n<seed:eos>",
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
            "{chat_history}\n\n<seed:bos>assistant\n{assistant_message}\n<seed:eos>",
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
            "{chat_history}\n\n<seed:bos>tool\n{tool_message}\n<seed:eos>",
            chat_history = chat_history.as_ref().trim(),
            tool_message = message.content().trim(),
        )
    }
}
impl BuildChatPrompt for SeedOssThinkPrompt {
    fn build(&self, messages: &mut Vec<ChatCompletionRequestMessage>) -> Result<String> {
        if messages.is_empty() {
            return Err(crate::error::PromptError::NoMessages);
        }

        // system prompt
        let system_message = match messages[0] {
            ChatCompletionRequestMessage::System(ref message) => {
                let content = message.content();
                match content.is_empty() {
                    true => DEFAULT_SEED_OSS_SYSTEM_MESSAGE.to_string(),
                    false => content.to_string(),
                }
            }
            _ => DEFAULT_SEED_OSS_SYSTEM_MESSAGE.to_string(),
        };

        let system_prompt = format!("<seed:bos>system\n{system_message}\n<seed:eos>");

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

        prompt.push_str("\n\n<seed:bos>assistant");

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
                        true => DEFAULT_SEED_OSS_SYSTEM_PROMPT_WITH_TOOLS
                            .replace("{system_message}", DEFAULT_SEED_OSS_SYSTEM_MESSAGE)
                            .replace("{tools}", &available_tools),
                        false => DEFAULT_SEED_OSS_SYSTEM_PROMPT_WITH_TOOLS
                            .replace("{system_message}", content)
                            .replace("{tools}", &available_tools),
                    }
                }
                _ => {
                    let content = message.content();
                    match content.is_empty() {
                        true => DEFAULT_SEED_OSS_SYSTEM_MESSAGE.to_string(),
                        false => content.to_string(),
                    }
                }
            },
            _ => match tools {
                Some(tools) if !tools.is_empty() => {
                    let available_tools = serde_json::to_string_pretty(tools).unwrap();

                    DEFAULT_SEED_OSS_SYSTEM_PROMPT_WITH_TOOLS
                        .replace("{system_message}", DEFAULT_SEED_OSS_SYSTEM_MESSAGE)
                        .replace("{tools}", &available_tools)
                }
                _ => DEFAULT_SEED_OSS_SYSTEM_MESSAGE.to_string(),
            },
        };

        let system_prompt = format!("<seed:bos>system\n{system_message}\n<seed:eos>");

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

        prompt.push_str("\n\n<seed:bos>assistant");

        Ok(prompt)
    }
}

/// Generate prompts for the `ByteDance/Seed-instruct` models
#[derive(Debug, Default, Clone)]
pub struct SeedOssNoThinkPrompt;
impl SeedOssNoThinkPrompt {
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
                        "<seed:bos>user\n{user_message}\n<seed:eos>",
                        user_message = content.trim(),
                    )
                }
                false => {
                    format!(
                        "{system_prompt}\n\n<seed:bos>user\n{user_message}\n<seed:eos>",
                        system_prompt = system_prompt.as_ref().trim(),
                        user_message = content.trim(),
                    )
                }
            },
            false => format!(
                "{chat_history}\n\n<seed:bos>user\n{user_message}\n<seed:eos>",
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
            "{chat_history}\n\n<seed:bos>assistant\n{assistant_message}\n<seed:eos>",
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
            "{chat_history}\n\n<seed:bos>tool\n{tool_message}\n<seed:eos>",
            chat_history = chat_history.as_ref().trim(),
            tool_message = message.content().trim(),
        )
    }
}
impl BuildChatPrompt for SeedOssNoThinkPrompt {
    fn build(&self, messages: &mut Vec<ChatCompletionRequestMessage>) -> Result<String> {
        if messages.is_empty() {
            return Err(crate::error::PromptError::NoMessages);
        }

        // system prompt
        let system_message = match messages[0] {
            ChatCompletionRequestMessage::System(ref message) => {
                let content = message.content();
                match content.is_empty() {
                    true => DEFAULT_SEED_OSS_SYSTEM_MESSAGE.to_string(),
                    false => content.to_string(),
                }
            }
            _ => DEFAULT_SEED_OSS_SYSTEM_MESSAGE.to_string(),
        };

        let system_prompt = format!("<seed:bos>system\n{system_message}\n<seed:eos>\n\n<seed:bos>system\nYou are an intelligent assistant that can answer questions in one step without the need for reasoning and thinking, that is, your thinking budget is 0. Next, please skip the thinking process and directly start answering the user's questions.\n<seed:eos>");

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

        prompt.push_str("\n\n<seed:bos>assistant");

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
                        true => DEFAULT_SEED_OSS_SYSTEM_PROMPT_WITH_TOOLS
                            .replace("{system_message}", DEFAULT_SEED_OSS_SYSTEM_MESSAGE)
                            .replace("{tools}", &available_tools),
                        false => DEFAULT_SEED_OSS_SYSTEM_PROMPT_WITH_TOOLS
                            .replace("{system_message}", content)
                            .replace("{tools}", &available_tools),
                    }
                }
                _ => {
                    let content = message.content();
                    match content.is_empty() {
                        true => DEFAULT_SEED_OSS_SYSTEM_MESSAGE.to_string(),
                        false => content.to_string(),
                    }
                }
            },
            _ => match tools {
                Some(tools) if !tools.is_empty() => {
                    let available_tools = serde_json::to_string_pretty(tools).unwrap();

                    DEFAULT_SEED_OSS_SYSTEM_PROMPT_WITH_TOOLS
                        .replace("{system_message}", DEFAULT_SEED_OSS_SYSTEM_MESSAGE)
                        .replace("{tools}", &available_tools)
                }
                _ => DEFAULT_SEED_OSS_SYSTEM_MESSAGE.to_string(),
            },
        };

        let system_prompt = format!("<seed:bos>system\n{system_message}\n<seed:eos>\n\n<seed:bos>system\nYou are an intelligent assistant that can answer questions in one step without the need for reasoning and thinking, that is, your thinking budget is 0. Next, please skip the thinking process and directly start answering the user's questions.\n<seed:eos>");

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

        prompt.push_str("\n\n<seed:bos>assistant");

        Ok(prompt)
    }
}
