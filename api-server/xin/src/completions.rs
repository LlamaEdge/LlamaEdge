use super::common::{LlamaCppLogitBiasType, Usage};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub struct CompletionRequestBuilder {
    req: CompletionRequest,
}
impl CompletionRequestBuilder {
    pub fn new(model: impl Into<String>, prompt: Vec<String>) -> Self {
        Self {
            req: CompletionRequest {
                model: model.into(),
                prompt,
                suffix: None,
                max_tokens: None,
                temperature: None,
                top_p: None,
                n_choice: None,
                stream: None,
                logprobs: None,
                echo: None,
                stop: None,
                presence_penalty: None,
                frequency_penalty: None,
                best_of: None,
                logit_bias: None,
                user: None,
                llama_cpp_top_k: 0,
                llama_cpp_repeat_penalty: 0.0,
                llama_cpp_logit_bias_type: None,
            },
        }
    }

    pub fn build(self) -> CompletionRequest {
        self.req
    }
}

/// Creates a completion for the provided prompt and parameters.
#[derive(Debug, Deserialize, Serialize)]
pub struct CompletionRequest {
    /// ID of the model to use.
    model: String,
    /// The prompt(s) to generate completions for, encoded as a string, array of strings, array of tokens, or array of token arrays.
    prompt: Vec<String>,
    /// The suffix that comes after a completion of inserted text.
    /// Defaults to None.
    #[serde(skip_serializing_if = "Option::is_none")]
    suffix: Option<String>,
    /// The maximum number of tokens to generate in the completion.
    ///
    /// The token count of your prompt plus max_tokens cannot exceed the model's context length.
    /// Defaults to 16.
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    /// Adjust the randomness of the generated text. Between 0.0 and 2.0. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
    ///
    /// We generally recommend altering this or top_p but not both.
    /// Defaults to 1.0.
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    /// Limit the next token selection to a subset of tokens with a cumulative probability above a threshold P. The value should be between 0.0 and 1.0.
    ///
    /// Top-p sampling, also known as nucleus sampling, is another text generation method that selects the next token from a subset of tokens that together have a cumulative probability of at least p. This method provides a balance between diversity and quality by considering both the probabilities of tokens and the number of tokens to sample from. A higher value for top_p (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text.
    /// Defaults to 1.0.
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    /// How many chat completion choices to generate for each input message.
    /// Defaults to 1.
    #[serde(skip_serializing_if = "Option::is_none")]
    n_choice: Option<u32>,
    /// Whether to stream the results as they are generated. Useful for chatbots.
    /// Defaults to false.
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    /// Include the log probabilities on the logprobs most likely tokens, as well the chosen tokens. For example, if logprobs is 5, the API will return a list of the 5 most likely tokens. The API will always return the logprob of the sampled token, so there may be up to logprobs+1 elements in the response.
    ///
    /// The maximum value for logprobs is 5.
    /// Defaults to None.
    #[serde(skip_serializing_if = "Option::is_none")]
    logprobs: Option<u32>,
    /// Echo back the prompt in addition to the completion.
    /// Defaults to false.
    #[serde(skip_serializing_if = "Option::is_none")]
    echo: Option<bool>,
    /// Up to 4 sequences where the API will stop generating further tokens. The returned text will not contain the stop sequence.
    /// Defaults to None.
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
    /// Defaults to 0.0.
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
    /// Defaults to 0.0.
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    /// Generates `best_of` completions server-side and returns the "best" (the one with the highest log probability per token). Results cannot be streamed.When used with `n_choice`, `best_of` controls the number of candidate completions and `n_choice` specifies how many to return – `best_of` must be greater than `n_choice`.
    /// Defaults to 1.
    #[serde(skip_serializing_if = "Option::is_none")]
    best_of: Option<u32>,
    /// Modify the likelihood of specified tokens appearing in the completion.
    /// Accepts a json object that maps tokens (specified by their token ID in the GPT tokenizer) to an associated bias value from -100 to 100. You can use this tokenizer tool (which works for both GPT-2 and GPT-3) to convert text to token IDs. Mathematically, the bias is added to the logits generated by the model prior to sampling. The exact effect will vary per model, but values between -1 and 1 should decrease or increase likelihood of selection; values like -100 or 100 should result in a ban or exclusive selection of the relevant token.
    /// As an example, you can pass {"50256": -100} to prevent the <|endoftext|> token from being generated.
    /// Defaults to None.
    #[serde(skip_serializing_if = "Option::is_none")]
    logit_bias: Option<HashMap<String, f32>>,
    /// A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<String>,

    //* llama.cpp specific parameters
    llama_cpp_top_k: i32,
    llama_cpp_repeat_penalty: f64,
    llama_cpp_logit_bias_type: Option<LlamaCppLogitBiasType>,
}

/// Represents a completion response from the API.
///
/// Note: both the streamed and non-streamed response objects share the same shape (unlike the chat endpoint).
#[derive(Debug, Deserialize, Serialize)]
pub struct CompletionResponse {
    /// A unique identifier for the completion.
    id: String,
    /// The object type, which is always "text_completion".
    object: String,
    /// The Unix timestamp (in seconds) of when the completion was created.
    created: u32,
    /// The model used for completion.
    model: String,
    /// The list of completion choices the model generated for the input prompt.
    choices: Vec<CompletionChoice>,
    /// Usage statistics for the completion request.
    usage: Vec<Usage>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct CompletionChoice {
    text: String,
    /// The index of the choice in the list of choices.
    index: u32,
    /// A chat completion delta generated by streamed model responses.
    logprobs: Option<LogprobResult>,
    /// The reason the model stopped generating tokens. This will be `stop` if the model hit a natural stop point or a provided stop sequence, `length` if the maximum number of tokens specified in the request was reached, or `function_call` if the model called a function.
    finish_reason: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct LogprobResult {
    pub tokens: Vec<String>,
    pub token_logprobs: Vec<f32>,
    pub top_logprobs: Vec<HashMap<String, f32>>,
    pub text_offset: Vec<i32>,
}
