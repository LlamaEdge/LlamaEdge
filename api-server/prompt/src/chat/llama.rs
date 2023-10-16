use crate::{error::Result, BuildPrompt};
use xin::chat::{ChatCompletionRequestMessage, ChatCompletionRole};

/// Generate prompts for the `Llama-2-chat` model.
#[derive(Debug, Default, Clone)]
pub struct Llama2ChatPrompt;
impl Llama2ChatPrompt {
    /// Create a system prompt from a chat completion request message.
    fn create_system_prompt(&self, system_message: &ChatCompletionRequestMessage) -> String {
        format!(
            "<<SYS>>\n{content} <</SYS>>",
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
impl BuildPrompt for Llama2ChatPrompt {
    fn build(&self, messages: &mut Vec<ChatCompletionRequestMessage>) -> Result<String> {
        if messages.is_empty() {
            return Ok(String::new());
        }

        // systemp prompt
        let system_prompt = if messages[0].role == ChatCompletionRole::System {
            let system_message = messages.remove(0);
            self.create_system_prompt(&system_message)
        } else {
            String::new()
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

        // ! DO NOT REMOVE THE CODE BELOW
        // let len = messages.len();
        // let mut left = 0;
        // while left < len {
        //     if messages[left].role == ChatCompletionRole::User {
        //         let mut content: String = messages[left].content.clone();

        //         let mut right = left + 1;
        //         while right < len && messages[right].role == ChatCompletionRole::User {
        //             content = format!("{} {}", content, messages[right].content.as_str().trim());
        //             right += 1;
        //         }
        //         left = right;

        //         prompt = Llama2ChatPrompt::append_user_message(&prompt, &system_prompt, &content);
        //     } else if messages[left].role == ChatCompletionRole::Assistant {
        //         let mut content: String = messages[left].content.clone();

        //         let mut right = left + 1;
        //         while right < len && messages[right].role == ChatCompletionRole::Assistant {
        //             content = format!("{} {}", content, messages[right].content.as_str().trim());
        //             right += 1;
        //         }
        //         left = right;

        //         prompt = Llama2ChatPrompt::append_assistant_message(&prompt, &content);
        //     } else {
        //         return Err(crate::error::PromptError::UnknownRole(messages[left].role));
        //     }
        // }

        println!("*** [prompt begin] ***");
        println!("{}", &prompt);
        println!("*** [prompt end] ***");

        Ok(prompt)
    }
}

/// Generate prompts for the `Codellama-instruct` model.
#[derive(Debug, Default, Clone)]
pub struct CodeLlamaInstructPrompt;
impl CodeLlamaInstructPrompt {
    /// Create a system prompt from a chat completion request message.
    fn create_system_prompt(&self, system_message: &ChatCompletionRequestMessage) -> String {
        format!(
            "<<SYS>>\n{content} <</SYS>>",
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
impl BuildPrompt for CodeLlamaInstructPrompt {
    fn build(&self, messages: &mut Vec<ChatCompletionRequestMessage>) -> Result<String> {
        if messages.is_empty() {
            return Ok(String::new());
        }

        // systemp prompt
        let system_prompt = if messages[0].role == ChatCompletionRole::System {
            let system_message = messages.remove(0);
            let _system_prompt = self.create_system_prompt(&system_message);

            // ! debug
            String::from("<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as short as possible, while being safe. <</SYS>>")
        } else {
            String::new()
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

        println!("*** [prompt begin] ***");
        println!("{}", &prompt);
        println!("*** [prompt end] ***");

        Ok(prompt)
    }
}
