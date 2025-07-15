use super::BuildChatPrompt;
use crate::{
    error::{PromptError, Result},
    utils::get_image_format,
};
use endpoints::chat::{
    ChatCompletionAssistantMessage, ChatCompletionRequestMessage, ChatCompletionSystemMessage,
    ChatCompletionToolMessage, ChatCompletionUserMessage, ChatCompletionUserMessageContent,
    ContentPart, Tool,
};

/// Smol-vl Prompt Template
#[derive(Debug, Default, Clone)]
pub struct SmolvlPrompt;
impl SmolvlPrompt {
    /// Create a user prompt from a chat completion request message.
    fn append_user_message(
        &self,
        chat_history: impl AsRef<str>,
        message: &ChatCompletionUserMessage,
    ) -> Result<String> {
        let prompt = match message.content() {
            ChatCompletionUserMessageContent::Text(content) => {
                match chat_history.as_ref().is_empty() {
                    true => {
                        format!(
                            "<|im_start|>\nUser: {user_message}<end_of_utterance>",
                            user_message = content.trim(),
                        )
                    }
                    false => format!(
                        "{chat_history}\nUser: {user_message}<end_of_utterance>",
                        chat_history = chat_history.as_ref().trim(),
                        user_message = content.trim(),
                    ),
                }
            }
            ChatCompletionUserMessageContent::Parts(parts) => {
                let mut content = String::new();
                let mut image_contents = vec![];
                for part in parts {
                    match part {
                        ContentPart::Text(text_content) => {
                            content.push_str(text_content.text());
                            content.push('\n');
                        }
                        ContentPart::Image(part) => {
                            let image_content = match part.image().is_url() {
                                true => String::from("<image>"),
                                false => {
                                    let base64_str = part.image().url.as_str();
                                    let format = get_image_format(base64_str)?;
                                    format!(
                                        r#"<img src="data:image/{format};base64,{base64_str}">"#
                                    )
                                }
                            };
                            image_contents.push(image_content);
                        }
                        ContentPart::Audio(_part) => {
                            let err_msg =
                                "Audio content is not supported for models that use the `smol-vision` prompt template.";
                            return Err(PromptError::UnsupportedContent(err_msg.to_string()));
                        }
                    }
                }

                let mut image_embeddings = String::new();
                for image_content in image_contents {
                    let image_embedding = format!(
                        "<|vision_start|>{image_content}<|vision_end|>",
                        image_content = image_content.trim(),
                    );
                    image_embeddings.push_str(&image_embedding);
                }

                match chat_history.as_ref().is_empty() {
                    true => format!(
                        "<|im_start|>\nUser: {user_message}{image_embeddings}<end_of_utterance>",
                        user_message = content.trim(),
                        image_embeddings = image_embeddings.trim(),
                    ),
                    false => format!(
                        "{chat_history}\nUser: {user_message}{image_embeddings}<end_of_utterance>",
                        chat_history = chat_history.as_ref().trim(),
                        user_message = content.trim(),
                        image_embeddings = image_embeddings.trim(),
                    ),
                }
            }
        };

        Ok(prompt)
    }

    /// create an assistant prompt from a chat completion request message.
    fn append_assistant_message(
        &self,
        chat_history: impl AsRef<str>,
        message: &ChatCompletionAssistantMessage,
    ) -> Result<String> {
        let content = match message.content() {
            Some(content) => content.to_string(),
            None => return Err(PromptError::NoAssistantMessage),
        };

        Ok(format!(
            "{chat_history}\nAssistant: {assistant_message}<end_of_utterance>",
            chat_history = chat_history.as_ref().trim(),
            assistant_message = content.trim(),
        ))
    }
}
impl BuildChatPrompt for SmolvlPrompt {
    fn build(&self, messages: &mut Vec<ChatCompletionRequestMessage>) -> Result<String> {
        if messages.is_empty() {
            return Err(crate::error::PromptError::NoMessages);
        }

        // append user/assistant messages
        let mut prompt = String::new();
        for message in messages {
            match message {
                ChatCompletionRequestMessage::User(message) => {
                    prompt = self.append_user_message(&prompt, message)?;
                }
                ChatCompletionRequestMessage::Assistant(message) => {
                    prompt = self.append_assistant_message(&prompt, message)?;
                }
                _ => continue,
            }
        }

        prompt.push_str("\nAssistant");

        Ok(prompt)
    }
}

/// Generate prompts for the `smol-3` models in no-think mode.
#[derive(Debug, Default, Clone)]
pub struct Smol3NoThinkPrompt;
impl Smol3NoThinkPrompt {
    /// Create a system prompt from a chat completion request message.
    fn create_system_prompt(&self, message: &ChatCompletionSystemMessage) -> String {
        let content = message.content();
        match content.is_empty() {
            true => {
                let content = r#"## Metadata

Knowledge Cutoff Date: June 2025
Today Date: 15 July 2025
Reasoning Mode: /no_think

## Custom Instructions

You are a helpful AI assistant named SmolLM, trained by Hugging Face."#;
                format!("<|im_start|>system\n{content}\n<|im_end|>")
            }
            false => {
                let content = r#"## Metadata

Knowledge Cutoff Date: June 2025
Today Date: 15 July 2025
Reasoning Mode: /no_think

## Custom Instructions"#;
                format!("<|im_start|>system\n{content}\n\n{content}\n<|im_end|>")
            }
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
                        "<|im_start|>user\n{user_message}\n<|im_end|>",
                        user_message = content.trim(),
                    )
                }
                false => {
                    format!(
                        "{system_prompt}\n<|im_start|>user\n{user_message}\n<|im_end|>",
                        system_prompt = system_prompt.as_ref().trim(),
                        user_message = content.trim(),
                    )
                }
            },
            false => format!(
                "{chat_history}\n<|im_start|>user\n{user_message}\n<|im_end|>",
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
            None => match message.tool_calls() {
                Some(tool_calls) if !tool_calls.is_empty() => {
                    let mut functions = vec![];
                    for tool_call in tool_calls {
                        functions.push(&tool_call.function);
                    }
                    serde_json::to_string(&functions).unwrap()
                }
                _ => return Err(PromptError::NoAssistantMessage),
            },
        };

        Ok(format!(
            "{chat_history}\n<|im_start|>assistant\n{assistant_message}\n<|im_end|>",
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
            "{chat_history}\n<|im_start|>user\n{tool_message}\n<|im_end|>",
            chat_history = chat_history.as_ref().trim(),
            tool_message = message.content().trim(),
        )
    }
}
impl BuildChatPrompt for Smol3NoThinkPrompt {
    fn build(&self, messages: &mut Vec<ChatCompletionRequestMessage>) -> Result<String> {
        if messages.is_empty() {
            return Err(crate::error::PromptError::NoMessages);
        }

        // system prompt
        let system_prompt = match messages[0] {
            ChatCompletionRequestMessage::System(ref message) => self.create_system_prompt(message),
            _ => String::from("<|im_start|>system\nYou are a helpful assistant. Answer questions as concisely as possible./no_think\n<|im_end|>"),
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
        let system_prompt = match messages[0] {
            ChatCompletionRequestMessage::System(ref message) => match tools {
                Some(tools) if !tools.is_empty() => {
                    let mut available_tools = String::new();
                    for tool in tools {
                        let tool_str = serde_json::to_string(&tool.function).unwrap();
                        available_tools.push_str(&tool_str);
                        available_tools.push('\n');
                    }

                    let tools = format!("<tools>\n{}\n</tools>", available_tools.trim());

                    let begin = r#"### Tools

You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:"#;

                    let end = r#"For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"#;

                    let tool_part = format!("{begin}\n\n{tools}\n\n{end}");

                    let content = message.content();
                    match content.is_empty() {
                        true => {
                            let content = r#"## Metadata

Knowledge Cutoff Date: June 2025
Today Date: 15 July 2025
Reasoning Mode: /no_think

## Custom Instructions

You are a helpful AI assistant named SmolLM, trained by Hugging Face."#;

                            format!("<|im_start|>system\n{content}\n\n{tool_part}\n\n/no_think\n<|im_end|>")
                        }
                        false => {
                            let metadata = r#"## Metadata

Knowledge Cutoff Date: June 2025
Today Date: 15 July 2025
Reasoning Mode: /no_think

## Custom Instructions"#;

                            format!("<|im_start|>system\n{metadata}\n\n{content}\n\n{tool_part}\n\n<|im_end|>")
                        }
                    }
                }
                _ => self.create_system_prompt(message),
            },
            _ => match tools {
                Some(tools) if !tools.is_empty() => {
                    let mut available_tools = String::new();
                    for tool in tools {
                        let tool_str = serde_json::to_string(&tool.function).unwrap();
                        available_tools.push_str(&tool_str);
                        available_tools.push('\n');
                    }

                    let tools = format!("<tools>\n{}\n</tools>", available_tools.trim());

                    let begin = r#"### Tools

You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:"#;

                    let end = r#"For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"#;

                    let tool_part = format!("{begin}\n\n{tools}\n\n{end}");

                    let content = r#"## Metadata

Knowledge Cutoff Date: June 2025
Today Date: 15 July 2025
Reasoning Mode: /no_think

## Custom Instructions

You are a helpful AI assistant named SmolLM, trained by Hugging Face."#;

                    format!("<|im_start|>system\n{content}\n\n{tool_part}\n\n<|im_end|>")
                }
                _ => {
                    let content = r#"## Metadata

Knowledge Cutoff Date: June 2025
Today Date: 15 July 2025
Reasoning Mode: /no_think

## Custom Instructions

You are a helpful AI assistant named SmolLM, trained by Hugging Face."#;

                    format!("<|im_start|>system\n{content}\n<|im_end|>")
                }
            },
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
