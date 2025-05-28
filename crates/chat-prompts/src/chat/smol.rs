use super::BuildChatPrompt;
use crate::{
    error::{PromptError, Result},
    utils::get_image_format,
};
use endpoints::chat::{
    ChatCompletionAssistantMessage, ChatCompletionRequestMessage, ChatCompletionUserMessage,
    ChatCompletionUserMessageContent, ContentPart,
};

/// Smol-vl Prompt Template
#[derive(Debug, Default, Clone)]
pub struct SmolvlPrompt;
impl SmolvlPrompt {
    /// Create a user prompt from a chat completion request message.
    fn append_user_message(
        &self,
        chat_history: impl AsRef<str>,
        message: &ChatCompletionUserMessage,
    ) -> Result<String> {
        let prompt = match message.content() {
            ChatCompletionUserMessageContent::Text(content) => {
                match chat_history.as_ref().is_empty() {
                    true => {
                        format!(
                            "<|im_start|>\nUser: {user_message}<end_of_utterance>",
                            user_message = content.trim(),
                        )
                    }
                    false => format!(
                        "{chat_history}\nUser: {user_message}<end_of_utterance>",
                        chat_history = chat_history.as_ref().trim(),
                        user_message = content.trim(),
                    ),
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
                            let err_msg =
                                "Audio content is not supported for models that use the `smol-vision` prompt template.";
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
                    true => format!(
                        "<|im_start|>\nUser: {user_message}{image_embeddings}<end_of_utterance>",
                        user_message = content.trim(),
                        image_embeddings = image_embeddings.trim(),
                    ),
                    false => format!(
                        "{chat_history}\nUser: {user_message}{image_embeddings}<end_of_utterance>",
                        chat_history = chat_history.as_ref().trim(),
                        user_message = content.trim(),
                        image_embeddings = image_embeddings.trim(),
                    ),
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
            None => return Err(PromptError::NoAssistantMessage),
        };

        Ok(format!(
            "{chat_history}\nAssistant: {assistant_message}<end_of_utterance>",
            chat_history = chat_history.as_ref().trim(),
            assistant_message = content.trim(),
        ))
    }
}
impl BuildChatPrompt for SmolvlPrompt {
    fn build(&self, messages: &mut Vec<ChatCompletionRequestMessage>) -> Result<String> {
        if messages.is_empty() {
            return Err(crate::error::PromptError::NoMessages);
        }

        // append user/assistant messages
        let mut prompt = String::new();
        for message in messages {
            match message {
                ChatCompletionRequestMessage::User(message) => {
                    prompt = self.append_user_message(&prompt, message)?;
                }
                ChatCompletionRequestMessage::Assistant(message) => {
                    prompt = self.append_assistant_message(&prompt, message)?;
                }
                _ => continue,
            }
        }

        prompt.push_str("\nAssistant");

        Ok(prompt)
    }
}
