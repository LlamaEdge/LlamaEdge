use crate::common::Usage;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub struct ChatCompletionRequestBuilder {
    req: ChatCompletionRequest,
}
impl ChatCompletionRequestBuilder {
    /// Creates a new builder with the given model.
    ///
    /// # Arguments
    ///
    /// * `model` - ID of the model to use.
    ///
    /// * `messages` - A list of messages comprising the conversation so far.
    ///
    /// * `sampling` - The sampling method to use.
    pub fn new(model: impl Into<String>, messages: Vec<ChatCompletionRequestMessage>) -> Self {
        Self {
            req: ChatCompletionRequest {
                model: model.into(),
                messages,
                temperature: None,
                top_p: None,
                n_choice: None,
                stream: None,
                stop: None,
                max_tokens: None,
                presence_penalty: None,
                frequency_penalty: None,
                logit_bias: None,
                user: None,
                functions: None,
                function_call: None,
            },
        }
    }

    pub fn with_sampling(mut self, sampling: ChatCompletionRequestSampling) -> Self {
        let (temperature, top_p) = match sampling {
            ChatCompletionRequestSampling::Temperature(t) => (t, 1.0),
            ChatCompletionRequestSampling::TopP(p) => (1.0, p),
        };
        self.req.temperature = Some(temperature);
        self.req.top_p = Some(top_p);
        self
    }

    /// Sets the number of chat completion choices to generate for each input message.
    ///
    /// # Arguments
    ///
    /// * `n` - How many chat completion choices to generate for each input message. If `n` is less than 1, then sets to `1`.
    pub fn with_n_choices(mut self, n: i32) -> Self {
        let n_choice = if n < 1 { 1 } else { n };
        self.req.n_choice = Some(n_choice);
        self
    }

    pub fn with_stream(mut self, flag: bool) -> Self {
        self.req.stream = Some(flag);
        self
    }

    pub fn with_stop(mut self, stop: Vec<String>) -> Self {
        self.req.stop = Some(stop);
        self
    }

    /// Sets the maximum number of tokens to generate in the chat completion. The total length of input tokens and generated tokens is limited by the model's context length.
    ///
    /// # Argument
    ///
    /// * `max_tokens` - The maximum number of tokens to generate in the chat completion. If `max_tokens` is less than 1, then sets to `16`.
    pub fn with_max_tokens(mut self, max_tokens: i32) -> Self {
        let max_tokens = if max_tokens < 1 { 16 } else { max_tokens };
        self.req.max_tokens = Some(max_tokens);
        self
    }

    /// Sets the presence penalty. Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
    pub fn with_presence_penalty(mut self, penalty: f32) -> Self {
        self.req.presence_penalty = Some(penalty);
        self
    }

    /// Sets the frequency penalty. Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
    pub fn with_frequency_penalty(mut self, penalty: f32) -> Self {
        self.req.frequency_penalty = Some(penalty);
        self
    }

    pub fn with_logits_bias(mut self, map: HashMap<String, f64>) -> Self {
        self.req.logit_bias = Some(map);
        self
    }

    pub fn with_user(mut self, user: impl Into<String>) -> Self {
        self.req.user = Some(user.into());
        self
    }

    pub fn with_functions(mut self, functions: Vec<ChatCompletionRequestFunction>) -> Self {
        self.req.functions = Some(functions);
        self
    }

    pub fn with_function_call(mut self, function_call: impl Into<String>) -> Self {
        self.req.function_call = Some(function_call.into());
        self
    }

    pub fn build(self) -> ChatCompletionRequest {
        self.req
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ChatCompletionRequest {
    /// The model to use for generating completions.
    pub model: String,
    /// A list of messages comprising the conversation so far.
    pub messages: Vec<ChatCompletionRequestMessage>,
    /// Adjust the randomness of the generated text. Between 0.0 and 2.0. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
    ///
    /// We generally recommend altering this or top_p but not both.
    /// Defaults to 1.0.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Limit the next token selection to a subset of tokens with a cumulative probability above a threshold P. The value should be between 0.0 and 1.0.
    ///
    /// Top-p sampling, also known as nucleus sampling, is another text generation method that selects the next token from a subset of tokens that together have a cumulative probability of at least p. This method provides a balance between diversity and quality by considering both the probabilities of tokens and the number of tokens to sample from. A higher value for top_p (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text.
    ///
    /// We generally recommend altering this or temperature but not both.
    /// Defaults to 1.0.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
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
    pub max_tokens: Option<i32>,
    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
    /// Defaults to 0.0.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
    /// Defaults to 0.0.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    /// Modify the likelihood of specified tokens appearing in the completion.
    ///
    /// Accepts a json object that maps tokens (specified by their token ID in the tokenizer) to an associated bias value from -100 to 100. Mathematically, the bias is added to the logits generated by the model prior to sampling. The exact effect will vary per model, but values between -1 and 1 should decrease or increase likelihood of selection; values like -100 or 100 should result in a ban or exclusive selection of the relevant token.
    /// Defaults to None.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<HashMap<String, f64>>,
    /// A unique identifier representing your end-user.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,

    //* OpenAI specific parameters
    /// A list of functions the model may generate JSON inputs for.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub functions: Option<Vec<ChatCompletionRequestFunction>>,
    /// Controls how the model responds to function calls. "none" means the model does not call a function, and responds to the end-user. "auto" means the model can pick between an end-user or calling a function. Specifying a particular function via `{"name":\ "my_function"}` forces the model to call that function. "none" is the default when no functions are present. "auto" is the default if functions are present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ChatCompletionRequestMessage {
    /// The role of the messages author. One of `system`, `user`, `assistant`, or `function`.
    pub role: ChatCompletionRole,

    /// The contents of the message. `content` is required for all messages, and may be empty for assistant messages with function calls.
    pub content: String,

    /// Only avaiable for OpenAI API. The name of the author of this message. `name` is required if role is `function`, and it should be the name of the function whose response is in the `content`. May contain a-z, A-Z, 0-9, and underscores, with a maximum length of 64 characters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// Only available for OpenAI API. The name and arguments of a function that should be called, as generated by the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<ChatMessageFunctionCall>,
}

#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq)]
pub enum ChatCompletionRequestSampling {
    /// What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
    Temperature(f32),
    /// An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.
    TopP(f32),
}

/// The role of the messages author.
#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ChatCompletionRole {
    System,
    User,
    Assistant,
    Function,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ChatCompletionRequestFunction {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    parameters: ChatCompletionRequestFunctionParameters,
}

/// The parameters the functions accepts, described as a JSON Schema object. See the [guide](https://platform.openai.com/docs/guides/gpt/function-calling) for examples, and the [JSON Schema reference](https://json-schema.org/understanding-json-schema/) for documentation about the format.
///
/// To describe a function that accepts no parameters, provide the value `{"type": "object", "properties": {}}`.
#[derive(Debug, Deserialize, Serialize)]
pub struct ChatCompletionRequestFunctionParameters {
    #[serde(rename = "type")]
    pub schema_type: JSONSchemaType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<HashMap<String, Box<JSONSchemaDefine>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum JSONSchemaType {
    Object,
    Number,
    String,
    Array,
    Null,
    Boolean,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct JSONSchemaDefine {
    #[serde(rename = "type")]
    pub schema_type: Option<JSONSchemaType>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enum_values: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<HashMap<String, Box<JSONSchemaDefine>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub items: Option<Box<JSONSchemaDefine>>,
}

/// Represents a chat completion response returned by model, based on the provided input.
#[derive(Debug, Deserialize, Serialize)]
pub struct ChatCompletionResponse {
    /// A unique identifier for the chat completion.
    pub id: String,
    /// The object type, which is always `chat.completion`.
    pub object: String,
    /// The Unix timestamp (in seconds) of when the chat completion was created.
    pub created: u64,
    /// The model used for the chat completion.
    pub model: String,
    /// A list of chat completion choices. Can be more than one if `n_choice` is greater than 1.
    pub choices: Vec<ChatCompletionResponseChoice>,
    /// Usage statistics for the completion request.
    pub usage: Usage,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ChatCompletionResponseChoice {
    /// The index of the choice in the list of choices.
    pub index: u32,
    /// A chat completion message generated by the model.
    pub message: ChatCompletionResponseMessage,
    /// The reason the model stopped generating tokens. This will be `stop` if the model hit a natural stop point or a provided stop sequence, `length` if the maximum number of tokens specified in the request was reached, or `function_call` if the model called a function.
    pub finish_reason: FinishReason,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ChatCompletionResponseMessage {
    /// The role of the author of this message.
    pub role: ChatCompletionRole,
    /// The contents of the message.
    pub content: String,
    /// The name and arguments of a function that should be called, as generated by the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<ChatMessageFunctionCall>,
}

/// The reason the model stopped generating tokens.
#[derive(Debug, Serialize, Deserialize)]
#[allow(non_camel_case_types)]
pub enum FinishReason {
    /// `stop` if the model hit a natural stop point or a provided stop sequence.
    stop,
    /// `length` if the maximum number of tokens specified in the request was reached.
    length,
    /// `function_call` if the model called a function.
    function_call,
}

/// The name and arguments of a function that should be called, as generated by the model.
#[derive(Debug, Deserialize, Serialize)]
pub struct ChatMessageFunctionCall {
    /// The name of the function to call.
    pub name: String,

    /// The arguments to call the function with, as generated by the model in JSON format. Note that the model does not always generate valid JSON, and may hallucinate parameters not defined by your function schema. Validate the arguments in your code before calling your function.
    pub arguments: String,
}

// #[derive(Debug, Deserialize, Serialize)]
// pub struct ChatCompletionChunk {
//     /// A unique identifier for the chat completion.
//     id: String,
//     /// The object type, which is always `chat.completion.chunk`.
//     object: String,
//     /// The Unix timestamp (in seconds) of when the chat completion was created.
//     created: u32,
//     /// The model used for the chat completion.
//     model: String,
//     /// A list of chat completion choices. Can be more than one if `n_choice` is greater than 1.
//     choices: Vec<ChatCompletionChunkChoice>,
// }

// #[derive(Debug, Deserialize, Serialize)]
// pub struct ChatCompletionChunkChoice {
//     /// The index of the choice in the list of choices.
//     index: u32,
//     /// A chat completion delta generated by streamed model responses.
//     delta: ChatCompletionChunkChoiceDelta,
//     /// The reason the model stopped generating tokens. This will be `stop` if the model hit a natural stop point or a provided stop sequence, `length` if the maximum number of tokens specified in the request was reached, or `function_call` if the model called a function.
//     finish_reason: String,
// }

// #[derive(Debug, Deserialize, Serialize)]
// pub struct ChatCompletionChunkChoiceDelta {
//     /// The role of the author of this message.
//     role: String,
//     /// The contents of the chunk message.
//     content: String,
//     /// The name and arguments of a function that should be called, as generated by the model.
//     function_call: ChatCompletionRequestFunctionCall,
// }
