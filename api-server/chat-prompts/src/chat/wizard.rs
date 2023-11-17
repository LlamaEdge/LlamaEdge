use super::BuildChatPrompt;
use crate::error::Result;
use endpoints::chat::{ChatCompletionRequestMessage, ChatCompletionRole};

/// Generate prompts for the `wizard-vicuna` model.
#[derive(Debug, Default, Clone)]
pub struct WizardCoderPrompt;
impl WizardCoderPrompt {
    /// Create a system prompt from a chat completion request message.
    fn create_system_prompt(&self, system_message: &ChatCompletionRequestMessage) -> String {
        let content = system_message.content.as_str();
        match content.is_empty() {
            true => String::from("Below is an instruction that describes a task. Write a response that appropriately completes the request."),
            false => format!("{content}"),
        }
    }

    /// Create a user prompt from a chat completion request message.
    fn append_user_message(
        &self,
        system_prompt: impl AsRef<str>,
        content: impl AsRef<str>,
    ) -> String {
        format!(
            "{system_prompt}\n\n### Instruction:\n{user_message}",
            system_prompt = system_prompt.as_ref().trim(),
            user_message = content.as_ref().trim(),
        )
    }
}
impl BuildChatPrompt for WizardCoderPrompt {
    fn build(&self, messages: &mut Vec<ChatCompletionRequestMessage>) -> Result<String> {
        if messages.is_empty() {
            return Ok(String::new());
        }

        // systemp prompt
        let system_prompt = if messages[0].role == ChatCompletionRole::System {
            self.create_system_prompt(&messages[0])
        } else {
            String::from("Below is an instruction that describes a task. Write a response that appropriately completes the request.")
        };

        // append user message
        if messages.is_empty() {
            return Err(crate::error::PromptError::NoMessages);
        }

        let message = messages.last().unwrap();
        if message.role != ChatCompletionRole::User {
            return Err(crate::error::PromptError::NoMessages);
        }
        let mut prompt = self.append_user_message(&system_prompt, message.content.as_str());

        prompt.push_str("\n\n### Response:");

        Ok(prompt)
    }
}
