use super::BuildChatPrompt;
use crate::error::Result;
use endpoints::chat::{ChatCompletionRequestMessage, ChatCompletionRole};

/// Generate instruct prompt for the `microsoft/phi-2` model.
#[derive(Debug, Default, Clone)]
pub struct Phi2InstructPrompt;
impl Phi2InstructPrompt {
    fn append_user_message(&self, content: impl AsRef<str>) -> String {
        format!(
            "Instruct: {user_message}",
            user_message = content.as_ref().trim(),
        )
    }
}
impl BuildChatPrompt for Phi2InstructPrompt {
    fn build(&self, messages: &mut Vec<ChatCompletionRequestMessage>) -> Result<String> {
        let mut prompt = if let Some(user_message) = messages.last() {
            if user_message.role == ChatCompletionRole::User {
                self.append_user_message(user_message.content.as_str())
            } else {
                return Err(crate::error::PromptError::NoUserMessage);
            }
        } else {
            return Err(crate::error::PromptError::NoMessages);
        };

        prompt.push_str("\nOutput:");

        Ok(prompt)
    }
}

/// Generate chat prompt for the `microsoft/phi-2` model.
#[derive(Debug, Default, Clone)]
pub struct Phi2ChatPrompt;
impl Phi2ChatPrompt {
    /// Create a user prompt from a chat completion request message.
    fn append_user_message(
        &self,
        chat_history: impl AsRef<str>,
        content: impl AsRef<str>,
    ) -> String {
        match chat_history.as_ref().is_empty() {
            true => format!(
                "Alice: {user_message}",
                user_message = content.as_ref().trim(),
            ),
            false => format!(
                "{chat_history}\nAlice: {user_message}",
                chat_history = chat_history.as_ref().trim(),
                user_message = content.as_ref().trim(),
            ),
        }
    }

    /// create an assistant prompt from a chat completion request message.
    fn append_assistant_message(
        &self,
        chat_history: impl AsRef<str>,
        content: impl AsRef<str>,
    ) -> String {
        format!(
            "{chat_history}\nBob: {assistant_message}",
            chat_history = chat_history.as_ref().trim(),
            assistant_message = content.as_ref().trim(),
        )
    }
}
impl BuildChatPrompt for Phi2ChatPrompt {
    fn build(&self, messages: &mut Vec<ChatCompletionRequestMessage>) -> Result<String> {
        if messages.is_empty() {
            return Err(crate::error::PromptError::NoMessages);
        }

        let mut prompt = String::new();
        for message in messages {
            match message.role {
                ChatCompletionRole::System => continue,
                ChatCompletionRole::User => {
                    prompt = self.append_user_message(&prompt, message.content.as_str());
                }
                ChatCompletionRole::Assistant => {
                    prompt = self.append_assistant_message(&prompt, message.content.as_str());
                }
                _ => {
                    return Err(crate::error::PromptError::UnknownRole(message.role));
                }
            }
        }

        prompt.push_str("\nBob:");

        Ok(prompt)
    }
}
