use super::BuildChatPrompt;
use crate::error::{PromptError, Result};
use endpoints::chat::{
    ChatCompletionRequestMessage, ChatCompletionUserMessage, ChatCompletionUserMessageContent,
    ContentPart,
};

/// Generate prompts for the `Qwen2.5-Coder` model.
#[derive(Debug, Default, Clone)]
pub struct Qwen25CoderInstructPrompt;
impl Qwen25CoderInstructPrompt {
    /// Create a user prompt from a chat completion request message.
    fn append_user_message(&self, message: &ChatCompletionUserMessage) -> String {
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

        content.trim().to_owned()
    }
}
impl BuildChatPrompt for Qwen25CoderInstructPrompt {
    fn build(&self, messages: &mut Vec<ChatCompletionRequestMessage>) -> Result<String> {
        if messages.is_empty() {
            return Err(crate::error::PromptError::NoMessages);
        }

        // get the last message
        let last_message = messages.last().unwrap();
        match last_message {
            ChatCompletionRequestMessage::User(message) => {
                let prompt = self.append_user_message(message);
                Ok(prompt)
            }
            _ => Err(PromptError::BadMessages(
                "The last message is not a user message.".to_string(),
            )),
        }
    }
}
