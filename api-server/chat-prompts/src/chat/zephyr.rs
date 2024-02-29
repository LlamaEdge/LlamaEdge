use super::BuildChatPrompt;
use crate::error::{PromptError, Result};
use endpoints::chat::{
    ChatCompletionAssistantMessage, ChatCompletionRequestMessage, ChatCompletionSystemMessage,
    ChatCompletionUserMessage, ChatCompletionUserMessageContent, ContentPart,
};

#[derive(Debug, Default, Clone)]
pub struct ZephyrChatPrompt;
impl ZephyrChatPrompt {
    /// Create a system prompt from a chat completion request message.
    fn create_system_prompt(&self, message: &ChatCompletionSystemMessage) -> String {
        let content = message.content();
        match content.is_empty() {
            true => String::from("<|system|>\nYou are a friendly chatbot.</s>"),
            false => format!("<|system|>\n{content}</s>"),
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
                        content.push_str("\n");
                    }
                }
                content
            }
        };

        match chat_history.as_ref().is_empty() {
            true => format!(
                "{system_prompt}\n<|user|>\n{user_message}</s>",
                system_prompt = system_prompt.as_ref().trim(),
                user_message = content.trim(),
            ),
            false => format!(
                "{chat_history}\n<|user|>\n{user_message}</s>",
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
            "{chat_history}\n<|assistant|>\n{assistant_message}</s>",
            chat_history = chat_history.as_ref().trim(),
            assistant_message = content.trim(),
        ))
    }
}
impl BuildChatPrompt for ZephyrChatPrompt {
    fn build(&self, messages: &mut Vec<ChatCompletionRequestMessage>) -> Result<String> {
        if messages.is_empty() {
            return Err(crate::error::PromptError::NoMessages);
        }

        // system prompt
        let system_prompt = match messages[0] {
            ChatCompletionRequestMessage::System(ref message) => {
                self.create_system_prompt(&message)
            }
            _ => String::from("<|system|>\nYou are a friendly chatbot.</s>"),
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
                ChatCompletionRequestMessage::System(_) => continue,
            }
        }

        prompt.push_str("\n<|assistant|>");

        Ok(prompt)
    }
}

#[derive(Debug, Default, Clone)]
pub struct StableLMZephyrChatPrompt;
impl StableLMZephyrChatPrompt {
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
                        content.push_str("\n");
                    }
                }
                content
            }
        };

        match chat_history.as_ref().is_empty() {
            true => format!(
                "<|user|>\n{user_message}<|endoftext|>",
                user_message = content.trim(),
            ),
            false => format!(
                "{chat_history}\n<|user|>\n{user_message}<|endoftext|>",
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
            "{chat_history}\n<|assistant|>\n{assistant_message}</s>",
            chat_history = chat_history.as_ref().trim(),
            assistant_message = content.trim(),
        ))
    }
}
impl BuildChatPrompt for StableLMZephyrChatPrompt {
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
                ChatCompletionRequestMessage::System(_) => continue,
            }
        }

        prompt.push_str("\n<|assistant|>");

        Ok(prompt)
    }
}
