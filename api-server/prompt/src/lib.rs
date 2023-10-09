pub mod error;
pub mod llama;
pub mod mistral;

use error::Result;
use xin::chat::ChatCompletionRequestMessage;

pub trait BuildPrompt {
    fn build(messages: &mut Vec<ChatCompletionRequestMessage>) -> Result<String>;
}
