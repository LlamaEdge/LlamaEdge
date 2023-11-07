use crate::{chat::BuildChatPrompt, error::Result};
use endpoints::chat::{ChatCompletionRequestMessage, ChatCompletionRole};

/// Generate prompts for the `Llama-2-chat` model.
#[derive(Debug, Default, Clone)]
pub struct Baichuan2ChatPrompt;
impl Baichuan2ChatPrompt {
    /// Create a system prompt from a chat completion request message.
    fn create_system_prompt(&self, system_message: &ChatCompletionRequestMessage) -> String {
        format!("{content}", content = system_message.content.as_str())
    }

    /// Create a user prompt from a chat completion request message.
    fn append_user_message(
        &self,
        chat_history: impl AsRef<str>,
        system_prompt: impl AsRef<str>,
        content: impl AsRef<str>,
    ) -> String {
        match chat_history.as_ref().is_empty() {
            true => match system_prompt.as_ref().is_empty() {
                true => {
                    format!(
                        "用户:{user_message}",
                        user_message = content.as_ref().trim(),
                    )
                }
                false => {
                    format!(
                        "{system_prompt}\n\n用户:{user_message}",
                        system_prompt = system_prompt.as_ref().trim(),
                        user_message = content.as_ref().trim(),
                    )
                }
            },
            false => format!(
                "{chat_history}用户:{user_message}",
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
            "{chat_history}\n助手:{assistant_message}",
            chat_history = chat_history.as_ref().trim(),
            assistant_message = content.as_ref().trim(),
        )
    }
}
impl BuildChatPrompt for Baichuan2ChatPrompt {
    fn build(&self, messages: &mut Vec<ChatCompletionRequestMessage>) -> Result<String> {
        if messages.is_empty() {
            return Ok(String::new());
        }

        // systemp prompt
        let system_prompt = if messages[0].role == ChatCompletionRole::System {
            let system_message = messages.remove(0);
            self.create_system_prompt(&system_message)
        } else {
            String::from("以下内容为人类用户与一位智能助手的对话。")
        };

        // append user/assistant messages
        if messages.is_empty() {
            return Err(crate::error::PromptError::NoMessages);
        }

        let mut prompt = String::new();
        for message in messages {
            if message.role == ChatCompletionRole::User {
                prompt =
                    self.append_user_message(&prompt, &system_prompt, message.content.as_str());
            } else if message.role == ChatCompletionRole::Assistant {
                prompt = self.append_assistant_message(&prompt, message.content.as_str());
            } else {
                return Err(crate::error::PromptError::UnknownRole(message.role));
            }
        }

        prompt.push_str("\n助手:");

        Ok(prompt)
    }
}
