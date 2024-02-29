use super::BuildChatPrompt;
use crate::error::{PromptError, Result};
use endpoints::chat::{
    ChatCompletionAssistantMessage, ChatCompletionRequestMessage,
    ChatCompletionUserMessage, ChatCompletionUserMessageContent, ContentPart,
};

/// Generate prompts for the `gemma-7b-it` model.
#[derive(Debug, Default, Clone)]
pub struct GemmaInstructPrompt;
impl GemmaInstructPrompt {
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
                "<start_of_turn>user\n{user_message}<end_of_turn>\n<start_of_turn>model",
                user_message = content.trim(),
            ),
            false => format!(
                "{chat_history}\n<start_of_turn>user\n{user_message}<end_of_turn>\n<start_of_turn>model",
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
            "{chat_history}\n{assistant_message}<end_of_turn>model",
            chat_history = chat_history.as_ref().trim(),
            assistant_message = content.trim(),
        ))
    }
}
impl BuildChatPrompt for GemmaInstructPrompt {
    fn build(&self, messages: &mut Vec<ChatCompletionRequestMessage>) -> Result<String> {
        if messages.is_empty() {
            return Err(crate::error::PromptError::NoMessages);
        }

        // append user/assistant messages
        let mut prompt = String::new();
        for message in messages {
            match message {
                ChatCompletionRequestMessage::User(message) => {
                    prompt = self.append_user_message(&prompt, &message);
                }
                ChatCompletionRequestMessage::Assistant(message) => {
                    prompt = self.append_assistant_message(&prompt, message)?;
                }
                ChatCompletionRequestMessage::System(_) => continue,
            }
        }

        Ok(prompt)
    }
}
