use super::BuildChatPrompt;
use crate::error::Result;
use xin::chat::{ChatCompletionRequestMessage, ChatCompletionRole};

/// Generate prompts for the `BELLE-Llama2-13B-chat` model.
#[derive(Debug, Default, Clone)]
pub struct BelleLlama2ChatPrompt;
impl BelleLlama2ChatPrompt {
    /// Create a user prompt from a chat completion request message.
    fn append_user_message(
        &self,
        chat_history: impl AsRef<str>,
        content: impl AsRef<str>,
    ) -> String {
        match chat_history.as_ref().is_empty() {
            true => format!(
                "Human: \n{user_message}",
                user_message = content.as_ref().trim(),
            ),
            false => format!(
                "{chat_history}\nHuman: \n{user_message}",
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
            "{prompt}\n\nAssistant:{assistant_message}",
            prompt = chat_history.as_ref().trim(),
            assistant_message = content.as_ref().trim(),
        )
    }
}
impl BuildChatPrompt for BelleLlama2ChatPrompt {
    fn build(&self, messages: &mut Vec<ChatCompletionRequestMessage>) -> Result<String> {
        if messages.is_empty() {
            return Ok(String::new());
        }

        let mut prompt = String::new();
        for message in messages {
            if message.role == ChatCompletionRole::User {
                prompt = self.append_user_message(&prompt, message.content.as_str());
            } else if message.role == ChatCompletionRole::Assistant {
                prompt = self.append_assistant_message(&prompt, message.content.as_str());
            } else {
                return Err(crate::error::PromptError::UnknownRole(message.role));
            }
        }

        prompt.push_str("\n\nAssistant:\n");

        println!("*** [prompt begin] ***");
        println!("{}", &prompt);
        println!("*** [prompt end] ***");

        Ok(prompt)
    }
}
