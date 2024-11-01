//! Define types for the `completions` endpoint.

use super::common::{FinishReason, Usage};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Creates a completion for the provided prompt and parameters.
#[derive(Debug, Deserialize, Serialize)]
pub struct CompletionRequest {
    /// ID of the model to use.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,

    /// The prompt(s) to generate completions for, encoded as a string, array of strings, array of tokens, or array of token arrays.
    pub prompt: CompletionPrompt,
    /// Generates `best_of` completions server-side and returns the "best" (the one with the highest log probability per token). Results cannot be streamed.When used with `n_choice`, `best_of` controls the number of candidate completions and `n_choice` specifies how many to return – `best_of` must be greater than `n_choice`.
    /// Defaults to 1.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub best_of: Option<u32>,
    /// Echo back the prompt in addition to the completion.
    /// Defaults to false.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub echo: Option<bool>,
    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
    /// Defaults to 0.0.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    /// Modify the likelihood of specified tokens appearing in the completion.
    /// Accepts a json object that maps tokens (specified by their token ID in the GPT tokenizer) to an associated bias value from -100 to 100. You can use this tokenizer tool (which works for both GPT-2 and GPT-3) to convert text to token IDs. Mathematically, the bias is added to the logits generated by the model prior to sampling. The exact effect will vary per model, but values between -1 and 1 should decrease or increase likelihood of selection; values like -100 or 100 should result in a ban or exclusive selection of the relevant token.
    /// As an example, you can pass {"50256": -100} to prevent the <|endoftext|> token from being generated.
    /// Defaults to None.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<HashMap<String, f32>>,
    /// Include the log probabilities on the logprobs most likely tokens, as well the chosen tokens. For example, if logprobs is 5, the API will return a list of the 5 most likely tokens. The API will always return the logprob of the sampled token, so there may be up to logprobs+1 elements in the response.
    ///
    /// The maximum value for logprobs is 5.
    /// Defaults to None.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<u32>,
    /// The maximum number of tokens to generate in the completion.
    ///
    /// The token count of your prompt plus max_tokens cannot exceed the model's context length.
    /// Defaults to 16.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// How many completions to generate for each prompt.
    /// Defaults to 1.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
    /// Defaults to 0.0.

    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    /// Up to 4 sequences where the API will stop generating further tokens. The returned text will not contain the stop sequence.
    /// Defaults to None.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    /// Whether to stream the results as they are generated. Useful for chatbots.
    /// Defaults to false.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// The suffix that comes after a completion of inserted text.
    /// Defaults to None.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suffix: Option<String>,
    /// Adjust the randomness of the generated text. Between 0.0 and 2.0. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
    ///
    /// We generally recommend altering this or top_p but not both.
    /// Defaults to 1.0.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Limit the next token selection to a subset of tokens with a cumulative probability above a threshold P. The value should be between 0.0 and 1.0.
    ///
    /// Top-p sampling, also known as nucleus sampling, is another text generation method that selects the next token from a subset of tokens that together have a cumulative probability of at least p. This method provides a balance between diversity and quality by considering both the probabilities of tokens and the number of tokens to sample from. A higher value for top_p (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text.
    /// Defaults to 1.0.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    // //* llama.cpp specific parameters
    // llama_cpp_top_k: i32,
    // llama_cpp_repeat_penalty: f64,
    // llama_cpp_logit_bias_type: Option<LlamaCppLogitBiasType>,
}

#[test]
fn test_serialize_completion_request() {
    {
        let request = CompletionRequest {
            model: Some("text-davinci-003".to_string()),
            parallel_tool_calls: Some(false),
            prompt: CompletionPrompt::SingleText("Once upon a time".to_string()),
            best_of: Some(1),
            echo: Some(false),
            frequency_penalty: Some(0.0),
            logit_bias: Some(HashMap::new()),
            logprobs: Some(5),
            max_tokens: Some(16),
            n: Some(1),
            presence_penalty: Some(0.0),
            stop: Some(vec!["\n".to_string()]),
            stream: Some(false),
            suffix: Some("".to_string()),
            temperature: Some(1.0),
            top_p: Some(1.0),
            user: Some("user-123".to_string()),
        };

        let actual = serde_json::to_string(&request).unwrap();
        let expected = r#"{"model":"text-davinci-003","prompt":"Once upon a time","best_of":1,"echo":false,"frequency_penalty":0.0,"logit_bias":{},"logprobs":5,"max_tokens":16,"n":1,"presence_penalty":0.0,"stop":["\n"],"stream":false,"suffix":"","temperature":1.0,"top_p":1.0,"user":"user-123"}"#;
        assert_eq!(actual, expected);
    }

    {
        let request = CompletionRequest {
            model: None,
            parallel_tool_calls: None,
            prompt: CompletionPrompt::MultiText(vec![
                "Once upon a time".to_string(),
                "There was a cat".to_string(),
            ]),
            best_of: None,
            echo: None,
            frequency_penalty: None,
            logit_bias: None,
            logprobs: None,
            max_tokens: None,
            n: None,
            presence_penalty: None,
            stop: None,
            stream: None,
            suffix: None,
            temperature: None,
            top_p: None,
            user: None,
        };

        let actual = serde_json::to_string(&request).unwrap();
        let expected = r#"{"prompt":["Once upon a time","There was a cat"]}"#;
        assert_eq!(actual, expected);
    }
}

#[test]
fn test_deserialize_completion_request() {
    {
        let json = r#"{"model":"text-davinci-003","prompt":"Once upon a time","best_of":1,"echo":false,"frequency_penalty":0.0,"logit_bias":{},"logprobs":5,"max_tokens":16,"n":1,"presence_penalty":0.0,"stop":["\n"],"stream":false,"suffix":"","temperature":1.0,"top_p":1.0,"user":"user-123"}"#;
        let request: CompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.model, Some("text-davinci-003".to_string()));
        assert_eq!(
            request.prompt,
            CompletionPrompt::SingleText("Once upon a time".to_string())
        );
        assert_eq!(request.best_of, Some(1));
        assert_eq!(request.echo, Some(false));
        assert_eq!(request.frequency_penalty, Some(0.0));
        assert_eq!(request.logit_bias, Some(HashMap::new()));
        assert_eq!(request.logprobs, Some(5));
        assert_eq!(request.max_tokens, Some(16));
        assert_eq!(request.n, Some(1));
        assert_eq!(request.presence_penalty, Some(0.0));
        assert_eq!(request.stop, Some(vec!["\n".to_string()]));
        assert_eq!(request.stream, Some(false));
        assert_eq!(request.suffix, Some("".to_string()));
        assert_eq!(request.temperature, Some(1.0));
        assert_eq!(request.top_p, Some(1.0));
        assert_eq!(request.user, Some("user-123".to_string()));
    }

    {
        let json = r#"{"prompt":["Once upon a time","There was a cat"]}"#;
        let request: CompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.model, None);
        assert_eq!(
            request.prompt,
            CompletionPrompt::MultiText(vec![
                "Once upon a time".to_string(),
                "There was a cat".to_string()
            ])
        );
        assert_eq!(request.best_of, None);
        assert_eq!(request.echo, None);
        assert_eq!(request.frequency_penalty, None);
        assert_eq!(request.logit_bias, None);
        assert_eq!(request.logprobs, None);
        assert_eq!(request.max_tokens, None);
        assert_eq!(request.n, None);
        assert_eq!(request.presence_penalty, None);
        assert_eq!(request.stop, None);
        assert_eq!(request.stream, None);
        assert_eq!(request.suffix, None);
        assert_eq!(request.temperature, None);
        assert_eq!(request.top_p, None);
        assert_eq!(request.user, None);
    }
}

/// Defines the types of a user message content.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum CompletionPrompt {
    /// A single text prompt.
    SingleText(String),
    /// Multiple text prompts.
    MultiText(Vec<String>),
}

/// Represents a completion response from the API.
///
/// Note: both the streamed and non-streamed response objects share the same shape (unlike the chat endpoint).
#[derive(Debug, Deserialize, Serialize)]
pub struct CompletionObject {
    /// A unique identifier for the completion.
    pub id: String,
    /// The list of completion choices the model generated for the input prompt.
    pub choices: Vec<CompletionChoice>,
    /// The Unix timestamp (in seconds) of when the completion was created.
    pub created: u64,
    /// The model used for completion.
    pub model: String,
    /// The object type, which is always "text_completion".
    pub object: String,
    /// Usage statistics for the completion request.
    pub usage: Usage,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct CompletionChoice {
    /// The reason the model stopped generating tokens. This will be `stop` if the model hit a natural stop point or a provided stop sequence, `length` if the maximum number of tokens specified in the request was reached, or `function_call` if the model called a function.
    pub finish_reason: FinishReason,
    /// The index of the choice in the list of choices.
    pub index: u32,
    /// A chat completion delta generated by streamed model responses.
    pub logprobs: Option<LogprobResult>,
    pub text: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct LogprobResult {
    pub tokens: Vec<String>,
    pub token_logprobs: Vec<f32>,
    pub top_logprobs: Vec<HashMap<String, f32>>,
    pub text_offset: Vec<i32>,
}
