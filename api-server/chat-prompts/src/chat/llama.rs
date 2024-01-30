use super::BuildChatPrompt;
use crate::error::Result;
use endpoints::chat::{ChatCompletionRequestMessage, ChatCompletionRole};

/// Generate prompts for the `Llama-2-chat` model.
#[derive(Debug, Default, Clone)]
pub struct Llama2ChatPrompt;
impl Llama2ChatPrompt {
    /// Create a system prompt from a chat completion request message.
    fn create_system_prompt(&self, system_message: &ChatCompletionRequestMessage) -> String {
        let content = system_message.content.as_str();
        match content.is_empty() {
            true => String::from("<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as short as possible, while being safe. <</SYS>>"),
            false =>format!(
                "<<SYS>>\n{content} <</SYS>>"
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
            true => match system_prompt.as_ref().is_empty() {
                true => {
                    format!(
                        "<s>[INST] {user_message} [/INST]",
                        user_message = content.as_ref().trim(),
                    )
                }
                false => {
                    format!(
                        "<s>[INST] {system_prompt}\n\n{user_message} [/INST]",
                        system_prompt = system_prompt.as_ref().trim(),
                        user_message = content.as_ref().trim(),
                    )
                }
            },
            false => format!(
                "{chat_history}<s>[INST] {user_message} [/INST]",
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
            "{prompt} {assistant_message} </s>",
            prompt = chat_history.as_ref().trim(),
            assistant_message = content.as_ref().trim(),
        )
    }
}
impl BuildChatPrompt for Llama2ChatPrompt {
    fn build(&self, messages: &mut Vec<ChatCompletionRequestMessage>) -> Result<String> {
        if messages.is_empty() {
            return Ok(String::new());
        }

        // systemp prompt
        let system_prompt = if messages[0].role == ChatCompletionRole::System {
            self.create_system_prompt(&messages[0])
        } else {
            String::from("<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as short as possible, while being safe. <</SYS>>")
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

        // println!("*** [prompt begin] ***");
        // println!("{}", &prompt);
        // println!("*** [prompt end] ***");

        Ok(prompt)
    }
}

/// Generate prompts for the `Codellama-instruct` model.
#[derive(Debug, Default, Clone)]
pub struct CodeLlamaInstructPrompt;
impl CodeLlamaInstructPrompt {
    /// Create a system prompt from a chat completion request message.
    fn create_system_prompt(&self, system_message: &ChatCompletionRequestMessage) -> String {
        let content = system_message.content.as_str();
        match content.is_empty() {
            true => String::from("<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as short as possible, while being safe. <</SYS>>"),
            false => format!(
                "<<SYS>>\n{content} <</SYS>>"
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
            true => match system_prompt.as_ref().is_empty() {
                true => {
                    format!(
                        "<s>[INST] {user_message} [/INST]",
                        user_message = content.as_ref().trim(),
                    )
                }
                false => {
                    format!(
                        "<s>[INST] {system_prompt}\n\n{user_message} [/INST]",
                        system_prompt = system_prompt.as_ref().trim(),
                        user_message = content.as_ref().trim(),
                    )
                }
            },
            false => format!(
                "{chat_history}<s>[INST] {user_message} [/INST]",
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
            "{prompt} {assistant_message} </s>",
            prompt = chat_history.as_ref().trim(),
            assistant_message = content.as_ref().trim(),
        )
    }
}
impl BuildChatPrompt for CodeLlamaInstructPrompt {
    fn build(&self, messages: &mut Vec<ChatCompletionRequestMessage>) -> Result<String> {
        if messages.is_empty() {
            return Ok(String::new());
        }

        // systemp prompt
        let system_prompt = if messages[0].role == ChatCompletionRole::System {
            self.create_system_prompt(&messages[0])
        } else {
            String::from("<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as short as possible, while being safe. <</SYS>>")
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

        // println!("*** [prompt begin] ***");
        // println!("{}", &prompt);
        // println!("*** [prompt end] ***");

        Ok(prompt)
    }
}

/// Generate prompts for the `Codellama-instruct` model.
#[derive(Debug, Default, Clone)]
pub struct CodeLlamaSuperInstructPrompt;
impl CodeLlamaSuperInstructPrompt {
    /// Create a system prompt from a chat completion request message.
    fn create_system_prompt(&self, system_message: &ChatCompletionRequestMessage) -> String {
        let content = system_message.content.as_str();
        match content.is_empty() {
            true => String::from("<s>Source: system\n\n You are a helpful, respectful and honest assistant. Always answer as short as possible, while being safe. <step>"),
            false => format!(
                "<s>Source: system\n\n {content} <step>"
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
                "{system_prompt} Source: user\n\n {user_message} <step>",
                system_prompt = system_prompt.as_ref().trim(),
                user_message = content.as_ref().trim(),
            ),
            false => format!(
                "{chat_history} Source: user\n\n {user_message} <step>",
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
            "{prompt} Source: assistant\n\n {assistant_message} <step>",
            prompt = chat_history.as_ref().trim(),
            assistant_message = content.as_ref().trim(),
        )
    }
}
impl BuildChatPrompt for CodeLlamaSuperInstructPrompt {
    fn build(&self, messages: &mut Vec<ChatCompletionRequestMessage>) -> Result<String> {
        if messages.is_empty() {
            return Ok(String::new());
        }

        // systemp prompt
        let system_prompt = if messages[0].role == ChatCompletionRole::System {
            self.create_system_prompt(&messages[0])
        } else {
            String::from("<s>Source: system\n\n You are a helpful, respectful and honest assistant. Always answer as short as possible, while being safe. <step>")
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

        prompt.push_str(" Source: assistant\nDestination: user\n\n ");

        Ok(prompt)
    }
}
