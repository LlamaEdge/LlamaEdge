use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
#[allow(non_camel_case_types)]
pub enum LlamaCppLogitBiasType {
    input_ids,
    tokens,
}

#[derive(Debug, Default, Deserialize, Serialize)]
pub struct Usage {
    /// Number of tokens in the prompt.
    pub prompt_tokens: u64,
    /// Number of tokens in the generated completion.
    pub completion_tokens: u64,
    /// Total number of tokens used in the request (prompt + completion).
    pub total_tokens: u64,
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
