use super::BuildChatPrompt;
use crate::{
    error::{PromptError, Result},
    utils::get_image_format,
};
use endpoints::chat::{
    ChatCompletionAssistantMessage, ChatCompletionRequestMessage, ChatCompletionUserMessage,
    ChatCompletionUserMessageContent, ContentPart,
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
                        content.push('\n');
                    }
                }
                content
            }
        };

        match chat_history.as_ref().is_empty() {
            true => format!(
                "<bos><start_of_turn>user\n{user_message}<end_of_turn>\n<start_of_turn>model",
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
                    prompt = self.append_user_message(&prompt, message);
                }
                ChatCompletionRequestMessage::Assistant(message) => {
                    prompt = self.append_assistant_message(&prompt, message)?;
                }
                _ => continue,
            }
        }

        Ok(prompt)
    }
}

/// Generate prompts for the `gemma-3` model.
#[derive(Debug, Default, Clone)]
pub struct Gemma3Prompt;
impl Gemma3Prompt {
    /// Create a user prompt from a chat completion request message.
    fn append_user_message(
        &self,
        chat_history: impl AsRef<str>,
        system_prompt: impl AsRef<str>,
        message: &ChatCompletionUserMessage,
        last_user_message: bool,
    ) -> Result<String> {
        let prompt = match message.content() {
            ChatCompletionUserMessageContent::Text(content) => {
                match chat_history.as_ref().is_empty() {
                        true => {
                            match (last_user_message, system_prompt.as_ref().is_empty()) {
                                (true, false) => format!(
                                    "<bos><start_of_turn>user\n{system_prompt}\n\n{user_message}<end_of_turn>\n<start_of_turn>model",
                                    system_prompt = system_prompt.as_ref(),
                                    user_message = content.trim(),
                                ),
                                _ => format!(
                                    "<bos><start_of_turn>user\n{user_message}<end_of_turn>\n<start_of_turn>model",
                                    user_message = content.trim(),
                                ),
                            }
                        }
                        false => {
                            match (last_user_message, system_prompt.as_ref().is_empty()) {
                                (true, false) => format!(
                                    "{chat_history}\n<start_of_turn>user\n{system_prompt}\n\n{user_message}<end_of_turn>\n<start_of_turn>model",
                                    chat_history = chat_history.as_ref().trim(),
                                    system_prompt = system_prompt.as_ref(),
                                    user_message = content.trim(),
                                ),
                                _ => format!(
                                    "{chat_history}\n<start_of_turn>user\n{user_message}<end_of_turn>\n<start_of_turn>model",
                                    chat_history = chat_history.as_ref().trim(),
                                    user_message = content.trim(),
                                ),
                            }
                        }
                    }
            }
            ChatCompletionUserMessageContent::Parts(parts) => {
                let mut content = String::new();
                let mut image_contents = vec![];
                for part in parts {
                    match part {
                        ContentPart::Text(text_content) => {
                            content.push_str(text_content.text());
                            content.push('\n');
                        }
                        ContentPart::Image(part) => {
                            let image_content = match part.image().is_url() {
                                true => String::from("<image>"),
                                false => {
                                    let base64_str = part.image().url.as_str();
                                    let format = get_image_format(base64_str)?;
                                    format!(
                                        r#"<img src="data:image/{format};base64,{base64_str}">"#
                                    )
                                }
                            };
                            image_contents.push(image_content);
                        }
                        ContentPart::Audio(_part) => {
                            let err_msg = "Audio content is not supported for models that use the `gemma-3` prompt template.";
                            return Err(PromptError::UnsupportedContent(err_msg.to_string()));
                        }
                    }
                }

                let mut image_embeddings = String::new();
                for image_content in image_contents {
                    let image_embedding = format!(
                        "<|vision_start|>{image_content}<|vision_end|>",
                        image_content = image_content.trim(),
                    );
                    image_embeddings.push_str(&image_embedding);
                }

                match chat_history.as_ref().is_empty() {
                    true => {
                        match (last_user_message, system_prompt.as_ref().is_empty()) {
                            (true, false) => format!(
                                "<bos><start_of_turn>user\n{system_prompt}\n\n<start_of_image>{image_embeddings}<end_of_image>\n{user_message}<end_of_turn>\n<start_of_turn>model",
                                system_prompt = system_prompt.as_ref(),
                                image_embeddings = image_embeddings.trim(),
                                user_message = content.trim(),
                            ),
                            _ => format!(
                                "<bos><start_of_turn>user\n<start_of_image>{image_embeddings}<end_of_image>\n{user_message}<end_of_turn>\n<start_of_turn>model",
                                image_embeddings = image_embeddings.trim(),
                                user_message = content.trim(),
                            ),
                        }
                    }
                    false => {
                        match (last_user_message, system_prompt.as_ref().is_empty()) {
                            (true, false) => format!(
                                "{chat_history}\n<start_of_turn>user\n{system_prompt}\n\n<start_of_image>{image_embeddings}<end_of_image>\n{user_message}<end_of_turn>\n<start_of_turn>model",
                                chat_history = chat_history.as_ref().trim(),
                                system_prompt = system_prompt.as_ref(),
                                image_embeddings = image_embeddings.trim(),
                                user_message = content.trim(),
                            ),
                            _ => format!(
                                "{chat_history}\n<start_of_turn>user\n<start_of_image>{image_embeddings}<end_of_image>\n{user_message}<end_of_turn>\n<start_of_turn>model",
                                chat_history = chat_history.as_ref().trim(),
                                image_embeddings = image_embeddings.trim(),
                                user_message = content.trim(),
                            ),
                        }
                    }
                }
            }
        };

        Ok(prompt)
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
impl BuildChatPrompt for Gemma3Prompt {
    fn build(&self, messages: &mut Vec<ChatCompletionRequestMessage>) -> Result<String> {
        if messages.is_empty() {
            return Err(crate::error::PromptError::NoMessages);
        }

        // check if the first message is a system message
        let mut system_prompt = String::new();
        if let ChatCompletionRequestMessage::System(message) = &messages[0] {
            system_prompt = message.content().to_string();
        }

        // append user/assistant messages
        let mut prompt = String::new();
        for (idx, message) in messages.iter().enumerate() {
            match message {
                ChatCompletionRequestMessage::User(message) => {
                    let last_user_message = idx == messages.len() - 1;
                    prompt = self.append_user_message(
                        &prompt,
                        &system_prompt,
                        message,
                        last_user_message,
                    )?;
                }
                ChatCompletionRequestMessage::Assistant(message) => {
                    prompt = self.append_assistant_message(&prompt, message)?;
                }
                _ => continue,
            }
        }

        Ok(prompt)
    }
}
