use super::BuildChatPrompt;
use crate::error::Result;
use endpoints::chat::{ChatCompletionRequestMessage, ChatCompletionRole};

/// Generate prompts for the `DeepSeek-LLM-Chat` model.
#[derive(Debug, Default, Clone)]
pub struct DeepseekChatPrompt;
impl DeepseekChatPrompt {
    /// Create a user prompt from a chat completion request message.
    fn append_user_message(
        &self,
        chat_history: impl AsRef<str>,
        content: impl AsRef<str>,
    ) -> String {
        match chat_history.as_ref().is_empty() {
            true => format!(
                "User: {user_message}",
                user_message = content.as_ref().trim(),
            ),
            false => format!(
                "{chat_history}User: {user_message}",
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
            "{chat_history}\n\nAssistant: {assistant_message}<|end_of_sentence|>",
            chat_history = chat_history.as_ref().trim(),
            assistant_message = content.as_ref().trim(),
        )
    }
}
impl BuildChatPrompt for DeepseekChatPrompt {
    fn build(&self, messages: &mut Vec<ChatCompletionRequestMessage>) -> Result<String> {
        // append user/assistant messages
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

        prompt.push_str("\n\nAssistant:");

        Ok(prompt)
    }
}

/// Generate prompts for the `DeepSeek-Coder` model.
#[derive(Debug, Default, Clone)]
pub struct DeepseekCoderPrompt;
impl DeepseekCoderPrompt {
    /// Create a system prompt from a chat completion request message.
    fn create_system_prompt(&self, system_message: &ChatCompletionRequestMessage) -> String {
        let content = system_message.content.as_str();
        match content.is_empty() {
            true => String::from("You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer."),
            false => format!(
                "{content}"
            )
        }
    }

    /// Create a user prompt from a chat completion request message.
    fn append_user_message(
        &self,
        chat_history: impl AsRef<str>,
        system_prompt: impl AsRef<str>,
        content: impl AsRef<str>,
    ) -> String {
        match chat_history.as_ref().is_empty() {
            true => format!(
                "{system_prompt}\n### Instruction:\n{user_message}",
                system_prompt = system_prompt.as_ref().trim(),
                user_message = content.as_ref().trim(),
            ),
            false => format!(
                "{chat_history}\n### Instruction:\n{user_message}",
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
            "{chat_histroy}\n### Response:\n{assistant_message}\n<|EOT|>",
            chat_histroy = chat_history.as_ref().trim(),
            assistant_message = content.as_ref().trim(),
        )
    }
}
impl BuildChatPrompt for DeepseekCoderPrompt {
    fn build(&self, messages: &mut Vec<ChatCompletionRequestMessage>) -> Result<String> {
        if messages.is_empty() {
            return Ok(String::new());
        }

        // systemp prompt
        let system_prompt = if messages[0].role == ChatCompletionRole::System {
            self.create_system_prompt(&messages[0])
        } else {
            String::from("You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.")
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

        prompt.push_str("\n### Response:");

        Ok(prompt)
    }
}
