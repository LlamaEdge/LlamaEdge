use crate::{
    chat::{ChatCompletionRequest, ChatCompletionRequestMessage},
    embeddings::EmbeddingRequest,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagEmbeddingRequest {
    #[serde(rename = "embeddings")]
    pub embedding_request: EmbeddingRequest,
    #[serde(rename = "url")]
    pub qdrant_url: String,
    #[serde(rename = "collection_name")]
    pub qdrant_collection_name: String,
}

#[test]
fn test_rag_serialize_embedding_request() {
    let embedding_request = EmbeddingRequest {
        model: "model".to_string(),
        input: vec!["input".to_string()],
        encoding_format: None,
        user: None,
    };
    let qdrant_url = "http://localhost:6333".to_string();
    let qdrant_collection_name = "qdrant_collection_name".to_string();
    let rag_embedding_request = RagEmbeddingRequest {
        embedding_request,
        qdrant_url,
        qdrant_collection_name,
    };
    let json = serde_json::to_string(&rag_embedding_request).unwrap();
    assert_eq!(
        json,
        r#"{"embeddings":{"model":"model","input":["input"]},"url":"http://localhost:6333","collection_name":"qdrant_collection_name"}"#
    );
}

#[test]
fn test_rag_deserialize_embedding_request() {
    let json = r#"{"embeddings":{"model":"model","input":["input"]},"url":"http://localhost:6333","collection_name":"qdrant_collection_name"}"#;
    let rag_embedding_request: RagEmbeddingRequest = serde_json::from_str(json).unwrap();
    assert_eq!(rag_embedding_request.qdrant_url, "http://localhost:6333");
    assert_eq!(
        rag_embedding_request.qdrant_collection_name,
        "qdrant_collection_name"
    );
    assert_eq!(rag_embedding_request.embedding_request.model, "model");
    assert_eq!(
        rag_embedding_request.embedding_request.input,
        vec!["input".to_string()]
    );
}

#[derive(Debug, Deserialize, Serialize, Default)]
pub struct RagChatCompletionsRequest {
    /// The model to use for generating completions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_model: Option<String>,
    /// A list of messages comprising the conversation so far.
    pub messages: Vec<ChatCompletionRequestMessage>,
    /// ID of the embedding model to use.
    pub embedding_model: String,
    /// The format to return the embeddings in. Can be either float or base64.
    /// Defaults to float.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<String>,
    /// The URL of the Qdrant server.
    pub qdrant_url: String,
    /// The name of the collection in Qdrant.
    pub qdrant_collection_name: String,
    /// Max number of retrieved result.
    pub limit: u64,
    /// Adjust the randomness of the generated text. Between 0.0 and 2.0. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
    ///
    /// We generally recommend altering this or top_p but not both.
    /// Defaults to 1.0.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Limit the next token selection to a subset of tokens with a cumulative probability above a threshold P. The value should be between 0.0 and 1.0.
    ///
    /// Top-p sampling, also known as nucleus sampling, is another text generation method that selects the next token from a subset of tokens that together have a cumulative probability of at least p. This method provides a balance between diversity and quality by considering both the probabilities of tokens and the number of tokens to sample from. A higher value for top_p (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text.
    ///
    /// We generally recommend altering this or temperature but not both.
    /// Defaults to 1.0.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// How many chat completion choices to generate for each input message.
    /// Defaults to 1.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_choice: Option<i32>,
    /// Whether to stream the results as they are generated. Useful for chatbots.
    /// Defaults to false.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// A list of tokens at which to stop generation. If None, no stop tokens are used. Up to 4 sequences where the API will stop generating further tokens.
    /// Defaults to None
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    /// The maximum number of tokens to generate. The value should be no less than 1.
    /// Defaults to 16.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
    /// Defaults to 0.0.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
    /// Defaults to 0.0.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    /// Modify the likelihood of specified tokens appearing in the completion.
    ///
    /// Accepts a json object that maps tokens (specified by their token ID in the tokenizer) to an associated bias value from -100 to 100. Mathematically, the bias is added to the logits generated by the model prior to sampling. The exact effect will vary per model, but values between -1 and 1 should decrease or increase likelihood of selection; values like -100 or 100 should result in a ban or exclusive selection of the relevant token.
    /// Defaults to None.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<HashMap<String, f64>>,
    /// A unique identifier representing your end-user.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}
impl RagChatCompletionsRequest {
    pub fn as_chat_completions_request(&self) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: self.chat_model.clone(),
            messages: self.messages.clone(),
            temperature: self.temperature,
            top_p: self.top_p,
            n_choice: self.n_choice,
            stream: self.stream,
            stop: self.stop.clone(),
            max_tokens: self.max_tokens,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            logit_bias: self.logit_bias.clone(),
            user: self.user.clone(),
            functions: None,
            function_call: None,
        }
    }
}
