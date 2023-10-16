pub mod llama;
pub mod mistral;

use crate::error::Result;
use xin::chat::ChatCompletionRequestMessage;

pub trait BuildChatPrompt: Send {
    fn build(&self, messages: &mut Vec<ChatCompletionRequestMessage>) -> Result<String>;
}
