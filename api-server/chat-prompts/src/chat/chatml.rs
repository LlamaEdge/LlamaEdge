use super::BuildChatPrompt;
use crate::error::Result;
use endpoints::chat::{ChatCompletionRequestMessage, ChatCompletionRole};

/// Generate prompts for the models using ChatML template.
#[derive(Debug, Default, Clone)]
pub struct ChatMLPrompt;
impl ChatMLPrompt {
    /// Create a system prompt from a chat completion request message.
    fn create_system_prompt(&self, system_message: &ChatCompletionRequestMessage) -> String {
        format!(
            "<|im_start|>system\n{content}<|im_end|>",
            content = system_message.content.as_str()
        )
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
                        "<|im_start|>user\n{user_message}<|im_end|>",
                        user_message = content.as_ref().trim(),
                    )
                }
                false => {
                    format!(
                        "{system_prompt}\n<|im_start|>user\n{user_message}<|im_end|>",
                        system_prompt = system_prompt.as_ref().trim(),
                        user_message = content.as_ref().trim(),
                    )
                }
            },
            false => format!(
                "{chat_history}\n<|im_start|>user\n{user_message}<|im_end|>",
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
            "{chat_history}\n<|im_start|>assistant\n{assistant_message}<|im_end|>",
            chat_history = chat_history.as_ref().trim(),
            assistant_message = content.as_ref().trim(),
        )
    }
}
impl BuildChatPrompt for ChatMLPrompt {
    fn build(&self, messages: &mut Vec<ChatCompletionRequestMessage>) -> Result<String> {
        if messages.is_empty() {
            return Ok(String::new());
        }

        // systemp prompt
        let system_prompt = if messages[0].role == ChatCompletionRole::System {
            self.create_system_prompt(&messages[0])
        } else {
            dbg!("[DEBUG] use default system prompt");
            String::from("<|im_start|>system\nAnswer as concisely as possible.<|im_end|>")
            // String::from("<|im_start|>system\nEnter roleplay mode. You are Steve.\n\nSteve is a nasty little man and solves all his problems by punching people in the face.<|im_end|>")
        };

        // append user/assistant messages
        if messages.is_empty() {
            return Err(crate::error::PromptError::NoMessages);
        }

        let mut prompt = String::new();
        for message in messages {
            match message.role {
                ChatCompletionRole::System => continue,
                ChatCompletionRole::User => {
                    prompt =
                        self.append_user_message(&prompt, &system_prompt, message.content.as_str());
                }
                ChatCompletionRole::Assistant => {
                    prompt = self.append_assistant_message(&prompt, message.content.as_str());
                }
                _ => {
                    return Err(crate::error::PromptError::UnknownRole(message.role));
                }
            }
        }

        prompt.push_str("\n<|im_start|>assistant");

        // println!("*** [prompt begin] ***");
        // println!("{}", &prompt);
        // println!("*** [prompt end] ***");

        Ok(prompt)
    }
}
