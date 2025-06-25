//! Define types for building chat completion requests, including messages, tools, and tool choices.
//!
//! **Example 1** Create a normal chat completion request.
//! ```
//! #[cfg(not(feature = "index"))]
//! {
//!   use endpoints::chat::*;
//!
//!   let mut messages = Vec::new();
//!
//!   // create a system message
//!   let system_message = ChatCompletionRequestMessage::System(
//!       ChatCompletionSystemMessage::new("Hello, world!", None),
//!   );
//!   messages.push(system_message);
//!
//!   // create a user message
//!   let user_message_content = ChatCompletionUserMessageContent::Parts(vec![
//!       ContentPart::Text(TextContentPart::new("what is in the picture?")),
//!       ContentPart::Image(ImageContentPart::new(Image {
//!           url: "https://example.com/image.png".to_string(),
//!           detail: None,
//!       })),
//!   ]);
//!   let user_message =
//!       ChatCompletionRequestMessage::new_user_message(user_message_content, None);
//!   messages.push(user_message);
//!
//!   // create a chat completion request
//!   let request = ChatCompletionRequestBuilder::new(&messages)
//!       .with_model("model-id")
//!       .with_tool_choice(ToolChoice::None)
//!       .build();
//!
//!   // serialize the request to JSON string
//!   let json = serde_json::to_string(&request).unwrap();
//!   assert_eq!(
//!       json,
//!       r#"{"model":"model-id","messages":[{"role":"system","content":"Hello, world!"},{"role":"user","content":[{"type":"text","text":"what is in the picture?"},{"type":"image_url","image_url":{"url":"https://example.com/image.png"}}]}],"temperature":0.8,"top_p":0.9,"n":1,"stream":false,"max_tokens":-1,"max_completion_tokens":-1,"presence_penalty":0.0,"frequency_penalty":0.0,"tool_choice":"none"}"#
//!   );
//! }
//! ```
//!

use crate::common::{FinishReason, Usage};
use indexmap::IndexMap;
use serde::{
    de::{self, IgnoredAny, MapAccess, Visitor},
    Deserialize, Deserializer, Serialize,
};
use serde_json::Value;
use std::{collections::HashMap, fmt};

/// Request builder for creating a new chat completion request.
pub struct ChatCompletionRequestBuilder {
    req: ChatCompletionRequest,
}
impl ChatCompletionRequestBuilder {
    /// Creates a new builder with the given messages.
    ///
    /// # Arguments
    ///
    /// * `messages` - A list of messages comprising the conversation so far.
    pub fn new(messages: &[ChatCompletionRequestMessage]) -> Self {
        Self {
            req: ChatCompletionRequest {
                messages: messages.to_vec(),
                ..Default::default()
            },
        }
    }

    /// Sets the model name to use for generating completions.
    ///
    /// # Arguments
    ///
    /// * `model` - The name of the model to use.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.req.model = Some(model.into());
        self
    }

    /// Sets the sampling method to use.
    ///
    /// # Arguments
    ///
    /// * `sampling` - The sampling method to use.
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
    pub fn with_n_choices(mut self, n: u64) -> Self {
        let n_choice = if n < 1 { 1 } else { n };
        self.req.n_choice = Some(n_choice);
        self
    }

    /// Enables streaming reponse.
    ///
    /// # Arguments
    ///
    /// * `flag` - Whether to enable streaming response.
    pub fn enable_stream(mut self, flag: bool) -> Self {
        self.req.stream = Some(flag);
        self
    }

    /// Includes usage in streaming response.
    pub fn include_usage(mut self) -> Self {
        self.req.stream_options = Some(StreamOptions {
            include_usage: Some(true),
        });
        self
    }

    /// Sets the stop tokens.
    ///
    /// # Arguments
    ///
    /// * `stop` - A list of tokens at which to stop generation.
    pub fn with_stop(mut self, stop: Vec<String>) -> Self {
        self.req.stop = Some(stop);
        self
    }

    /// **Deprecated** Use `max_completion_tokens` instead.
    ///
    /// Sets the maximum number of tokens to generate in the chat completion. The total length of input tokens and generated tokens is limited by the model's context length.
    ///
    /// # Argument
    ///
    /// * `max_tokens` - The maximum number of tokens to generate in the chat completion. `-1` means infinity. `-2` means until context filled. Defaults to `-1`.
    #[deprecated(
        since = "0.24.0",
        note = "Please use `with_max_completion_tokens` instead."
    )]
    #[allow(deprecated)]
    pub fn with_max_tokens(mut self, max_tokens: i32) -> Self {
        self.req.max_tokens = Some(max_tokens);
        self
    }

    /// Sets the maximum number of tokens that can be generated for a completion.
    ///
    /// # Argument
    ///
    /// * `max_completion_tokens` - The maximum number of tokens that can be generated for a completion.
    pub fn with_max_completion_tokens(mut self, max_completion_tokens: i32) -> Self {
        self.req.max_completion_tokens = Some(max_completion_tokens);
        self
    }

    /// Sets the presence penalty. Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
    ///
    /// # Arguments
    ///
    /// * `penalty` - The presence penalty.
    pub fn with_presence_penalty(mut self, penalty: f64) -> Self {
        self.req.presence_penalty = Some(penalty);
        self
    }

    /// Sets the frequency penalty. Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
    ///
    /// # Arguments
    ///
    /// * `penalty` - The frequency penalty.
    pub fn with_frequency_penalty(mut self, penalty: f64) -> Self {
        self.req.frequency_penalty = Some(penalty);
        self
    }

    /// Sets the logit bias.
    ///
    /// # Arguments
    ///
    /// * `map` - A map of tokens to their associated bias values.
    pub fn with_logits_bias(mut self, map: HashMap<String, f64>) -> Self {
        self.req.logit_bias = Some(map);
        self
    }

    /// Sets the user.
    ///
    /// # Arguments
    ///
    /// * `user` - A unique identifier representing your end-user.
    pub fn with_user(mut self, user: impl Into<String>) -> Self {
        self.req.user = Some(user.into());
        self
    }

    /// Sets the response format.
    ///
    /// # Arguments
    ///
    /// * `response_format` - The response format to use.
    pub fn with_reponse_format(mut self, response_format: ChatResponseFormat) -> Self {
        self.req.response_format = Some(response_format);
        self
    }

    /// Sets tools.
    ///
    /// # Arguments
    ///
    /// * `tools` - A list of tools the model may call.
    pub fn with_tools(mut self, tools: Vec<Tool>) -> Self {
        self.req.tools = Some(tools);
        self
    }

    /// Sets tool choice.
    ///
    /// # Arguments
    ///
    /// * `tool_choice` - The tool choice to use.
    pub fn with_tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.req.tool_choice = Some(tool_choice);
        self
    }

    /// Sets MCP tools.
    ///
    /// # Arguments
    ///
    /// * `mcp_tools` - A list of MCP tools the model may call.
    pub fn with_mcp_tools(mut self, mcp_tools: Vec<McpTool>) -> Self {
        self.req.mcp_tools = Some(mcp_tools);
        self
    }

    #[cfg(all(feature = "rag", feature = "index"))]
    pub fn with_kw_search_mcp_tool(mut self, kw_search_mcp_tool: McpTool) -> Self {
        self.req.kw_search_mcp_tool = Some(kw_search_mcp_tool);
        self
    }

    /// Sets the number of user messages to use for context retrieval.
    ///
    /// # Arguments
    ///
    /// * `context_window` - The number of user messages to use for context retrieval.
    #[cfg(feature = "rag")]
    pub fn with_rag_context_window(mut self, context_window: u64) -> Self {
        self.req.context_window = Some(context_window);
        self
    }

    /// Settings for vector search with Qdrant.
    ///
    /// # Arguments
    ///
    /// * `vdb_server_url` - The URL of the VectorDB server.
    ///
    /// * `vdb_collection_name` - The names of the collections in VectorDB.
    #[cfg(feature = "rag")]
    pub fn with_vector_search_settings(
        mut self,
        vdb_server_url: impl Into<String>,
        vdb_collection_name: impl Into<Vec<String>>,
        vdb_api_key: Option<String>,
    ) -> Self {
        self.req.vdb_server_url = Some(vdb_server_url.into());
        self.req.vdb_collection_name = Some(vdb_collection_name.into());
        self.req.vdb_api_key = vdb_api_key;
        self
    }

    /// Settings for keyword search with kw-search-server.
    ///
    /// # Arguments
    ///
    /// * `kw_search_url` - The URL of the keyword search server.
    /// * `kw_search_index` - The name of the index to use for keyword search.
    /// * `kw_api_key` - The API key for the keyword search server.
    #[cfg(all(feature = "rag", feature = "index"))]
    pub fn with_kw_search_settings(
        mut self,
        kw_search_url: impl Into<String>,
        kw_search_index: impl Into<String>,
        kw_api_key: Option<String>,
    ) -> Self {
        self.req.kw_search_url = Some(kw_search_url.into());
        self.req.kw_search_index = Some(kw_search_index.into());
        self.req.kw_api_key = kw_api_key;

        self
    }

    /// Settings for keyword search with Elasticsearch.
    ///
    /// # Arguments
    ///
    /// * `es_search_url` - The URL of the Elasticsearch server.
    /// * `es_search_index` - The name of the index to use for the keyword search.
    /// * `es_search_fields` - The fields to use for the keyword search.
    /// * `es_api_key` - The API key for the Elasticsearch server.
    #[cfg(all(feature = "rag", feature = "index"))]
    pub fn with_es_search_settings(
        mut self,
        es_search_url: impl Into<String>,
        es_search_index: impl Into<String>,
        es_search_fields: impl Into<Vec<String>>,
        es_api_key: Option<String>,
    ) -> Self {
        self.req.es_search_url = Some(es_search_url.into());
        self.req.es_search_index = Some(es_search_index.into());
        self.req.es_search_fields = Some(es_search_fields.into());
        self.req.es_api_key = es_api_key;

        self
    }

    /// Settings for keyword search with TiDB.
    ///
    /// # Arguments
    ///
    /// * `tidb_search_host` - The host of the TiDB server.
    /// * `tidb_search_port` - The port of the TiDB server.
    /// * `tidb_search_username` - The username of the TiDB server.
    /// * `tidb_search_password` - The password of the TiDB server.
    /// * `tidb_search_database` - The database of the TiDB server.
    /// * `tidb_search_table` - The table of the TiDB server.
    #[cfg(all(feature = "rag", feature = "index"))]
    pub fn with_tidb_search_settings(
        mut self,
        tidb_search_host: impl Into<String>,
        tidb_search_port: impl Into<u16>,
        tidb_search_username: impl Into<String>,
        tidb_search_password: impl Into<String>,
        tidb_search_database: impl Into<String>,
        tidb_search_table: impl Into<String>,
    ) -> Self {
        self.req.tidb_search_host = Some(tidb_search_host.into());
        self.req.tidb_search_port = Some(tidb_search_port.into());
        self.req.tidb_search_username = Some(tidb_search_username.into());
        self.req.tidb_search_password = Some(tidb_search_password.into());
        self.req.tidb_search_database = Some(tidb_search_database.into());
        self.req.tidb_search_table = Some(tidb_search_table.into());

        self
    }

    /// Settings for filtering vector search results.
    ///
    /// # Arguments
    ///
    /// * `limit` - The limit for the search.
    /// * `score_threshold` - The score threshold for the search.
    /// * `weighted_alpha` - The weighted alpha for the search.
    #[cfg(all(feature = "rag", not(feature = "index")))]
    pub fn with_vector_search_filter(
        mut self,
        limit: impl Into<u64>,
        score_threshold: impl Into<f32>,
    ) -> Self {
        self.req.limit = Some(limit.into());
        self.req.score_threshold = Some(score_threshold.into());

        self
    }

    /// Settings for filtering vector and keyword search results.
    ///
    /// # Arguments
    ///
    /// * `limit` - The limit for the search.
    /// * `score_threshold` - The score threshold for the search.
    /// * `weighted_alpha` - The weighted alpha for the search.
    #[cfg(all(feature = "rag", feature = "index"))]
    pub fn with_search_filter(
        mut self,
        limit: impl Into<u64>,
        score_threshold: impl Into<f32>,
        weighted_alpha: impl Into<f64>,
    ) -> Self {
        self.req.limit = Some(limit.into());
        self.req.score_threshold = Some(score_threshold.into());
        self.req.weighted_alpha = Some(weighted_alpha.into());

        self
    }

    /// Builds the chat completion request.
    pub fn build(self) -> ChatCompletionRequest {
        self.req
    }
}

/// Represents a chat completion request.
#[derive(Debug, Serialize)]
pub struct ChatCompletionRequest {
    /// The model to use for generating completions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// A list of messages comprising the conversation so far.
    pub messages: Vec<ChatCompletionRequestMessage>,
    /// Adjust the randomness of the generated text. Between 0.0 and 2.0. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
    ///
    /// We generally recommend altering this or top_p but not both.
    /// Defaults to `0.8`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// An alternative to sampling with temperature. Limit the next token selection to a subset of tokens with a cumulative probability above a threshold `p`. The value should be between 0.0 and 1.0.
    ///
    /// Top-p sampling, also known as nucleus sampling, is another text generation method that selects the next token from a subset of tokens that together have a cumulative probability of at least `p`. This method provides a balance between diversity and quality by considering both the probabilities of tokens and the number of tokens to sample from. A higher value for top_p (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text.
    ///
    /// We generally recommend altering this or temperature but not both.
    /// Defaults to `0.9`. To disable top-p sampling, set it to `1.0`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// How many chat completion choices to generate for each input message.
    /// Defaults to `1`.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "n")]
    pub n_choice: Option<u64>,
    /// Whether to stream the results as they are generated. Useful for chatbots.
    /// Defaults to false.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Options for streaming response. Only set this when you set `stream: true`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    /// A list of tokens at which to stop generation. If None, no stop tokens are used. Up to 4 sequences where the API will stop generating further tokens.
    /// Defaults to None
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    /// **Deprecated since 0.24.0.** Use `max_completion_tokens` instead.
    ///
    /// The maximum number of tokens to generate.
    /// `-1` means infinity. `-2` means until context filled. Defaults to `-1`.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[deprecated(since = "0.24.0", note = "Please use `max_completion_tokens` instead.")]
    pub max_tokens: Option<i32>,
    /// An upper bound for the number of tokens that can be generated for a completion. `-1` means infinity. `-2` means until context filled. Defaults to `-1`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<i32>,
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

    /// Format that the model must output
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ChatResponseFormat>,
    /// A list of tools the model may call.
    ///
    /// Currently, only functions are supported as a tool. Use this to provide a list of functions the model may generate JSON inputs for.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    /// Controls which (if any) function is called by the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,

    // * Fields for MCP Tools
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mcp_tools: Option<Vec<McpTool>>,
    #[cfg(all(feature = "rag", feature = "index"))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kw_search_mcp_tool: Option<McpTool>,

    /// Number of user messages to use for context retrieval.
    /// The parameter is only used in RAG.
    #[cfg(feature = "rag")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u64>,

    // * Fields for vector search
    /// The URL of the VectorDB server.
    #[cfg(feature = "rag")]
    #[serde(rename = "vdb_server_url", skip_serializing_if = "Option::is_none")]
    pub vdb_server_url: Option<String>,
    /// The names of the collections in VectorDB.
    #[cfg(feature = "rag")]
    #[serde(
        rename = "vdb_collection_name",
        skip_serializing_if = "Option::is_none"
    )]
    pub vdb_collection_name: Option<Vec<String>>,
    /// The API key for the VectorDB server.
    #[cfg(feature = "rag")]
    #[serde(rename = "vdb_api_key", skip_serializing_if = "Option::is_none")]
    pub vdb_api_key: Option<String>,

    // * Fields for keyword search with kw-search-server
    /// The URL of the kw-search-server. This parameter is only used in RAG.
    #[cfg(all(feature = "rag", feature = "index"))]
    #[serde(rename = "kw_search_url", skip_serializing_if = "Option::is_none")]
    pub kw_search_url: Option<String>,
    /// The name of the index to use for the keyword search. This parameter is only used in RAG and reserved for kw-search-server.
    #[cfg(all(feature = "rag", feature = "index"))]
    #[serde(rename = "kw_search_index", skip_serializing_if = "Option::is_none")]
    pub kw_search_index: Option<String>,
    /// The API key for the keyword search server. This parameter is only used in RAG and reserved for kw-search-server.
    #[cfg(all(feature = "rag", feature = "index"))]
    #[serde(rename = "kw_api_key", skip_serializing_if = "Option::is_none")]
    pub kw_api_key: Option<String>,

    // * Fields for keyword search with Elasticsearch
    /// The URL of the Elasticsearch server. This parameter is only used in RAG.
    #[cfg(all(feature = "rag", feature = "index"))]
    #[serde(rename = "kw_search_url", skip_serializing_if = "Option::is_none")]
    pub es_search_url: Option<String>,
    /// The name of the index to use for the keyword search. This parameter is only used in RAG and reserved for Elasticsearch.
    #[cfg(all(feature = "rag", feature = "index"))]
    #[serde(rename = "kw_search_index", skip_serializing_if = "Option::is_none")]
    pub es_search_index: Option<String>,
    /// The fields to use for the keyword search. This parameter is only used in RAG and reserved for Elasticsearch.
    #[cfg(all(feature = "rag", feature = "index"))]
    #[serde(rename = "kw_search_fields", skip_serializing_if = "Option::is_none")]
    pub es_search_fields: Option<Vec<String>>,
    /// The API key for Elasticsearch server. This parameter is only used in RAG and reserved for Elasticsearch.
    #[cfg(all(feature = "rag", feature = "index"))]
    #[serde(rename = "kw_api_key", skip_serializing_if = "Option::is_none")]
    pub es_api_key: Option<String>,

    // * Fields for keyword search with TiDB
    /// The host of the TiDB server. This parameter is only used in RAG and reserved for TiDB.
    #[cfg(all(feature = "rag", feature = "index"))]
    #[serde(rename = "tidb_search_host", skip_serializing_if = "Option::is_none")]
    pub tidb_search_host: Option<String>,
    /// The port of the TiDB server. This parameter is only used in RAG and reserved for TiDB.
    #[cfg(all(feature = "rag", feature = "index"))]
    #[serde(rename = "tidb_search_port", skip_serializing_if = "Option::is_none")]
    pub tidb_search_port: Option<u16>,
    /// The username of the TiDB server. This parameter is only used in RAG and reserved for TiDB.
    #[cfg(all(feature = "rag", feature = "index"))]
    #[serde(
        rename = "tidb_search_username",
        skip_serializing_if = "Option::is_none"
    )]
    pub tidb_search_username: Option<String>,
    /// The password of the TiDB server. This parameter is only used in RAG and reserved for TiDB.
    #[cfg(all(feature = "rag", feature = "index"))]
    #[serde(
        rename = "tidb_search_password",
        skip_serializing_if = "Option::is_none"
    )]
    pub tidb_search_password: Option<String>,
    /// The database of the TiDB server. This parameter is only used in RAG and reserved for TiDB.
    #[cfg(all(feature = "rag", feature = "index"))]
    #[serde(
        rename = "tidb_search_database",
        skip_serializing_if = "Option::is_none"
    )]
    pub tidb_search_database: Option<String>,
    /// The table name of the TiDB server. This parameter is only used in RAG and reserved for TiDB.
    #[cfg(all(feature = "rag", feature = "index"))]
    #[serde(rename = "tidb_search_table", skip_serializing_if = "Option::is_none")]
    pub tidb_search_table: Option<String>,

    // * Fields for filtering search results
    /// Max number of retrieved results. This parameter is only used in RAG.
    #[cfg(feature = "rag")]
    #[serde(rename = "limit", skip_serializing_if = "Option::is_none")]
    pub limit: Option<u64>,
    /// The score threshold for the retrieved results. This parameter is only used in RAG.
    #[cfg(feature = "rag")]
    #[serde(rename = "score_threshold", skip_serializing_if = "Option::is_none")]
    pub score_threshold: Option<f32>,
    /// The weighted alpha for the keyword search result (`alpha`) and the vector search result (`1-alpha`). This parameter is only used in RAG.
    #[cfg(all(feature = "rag", feature = "index"))]
    #[serde(rename = "weighted_alpha", skip_serializing_if = "Option::is_none")]
    pub weighted_alpha: Option<f64>,
}
#[allow(deprecated)]
impl<'de> Deserialize<'de> for ChatCompletionRequest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct ChatCompletionRequestVisitor;

        impl<'de> Visitor<'de> for ChatCompletionRequestVisitor {
            type Value = ChatCompletionRequest;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct ChatCompletionRequest")
            }

            fn visit_map<V>(self, mut map: V) -> Result<ChatCompletionRequest, V::Error>
            where
                V: MapAccess<'de>,
            {
                // Initialize all fields as None or empty
                let mut model = None;
                let mut messages = None;
                let mut temperature = None;
                let mut top_p = None;
                let mut n_choice = None;
                let mut stream = None;
                let mut stream_options = None;
                let mut stop = None;
                let mut max_tokens = None;
                let mut max_completion_tokens = None;
                let mut presence_penalty = None;
                let mut frequency_penalty = None;
                let mut logit_bias = None;
                let mut user = None;
                let mut response_format = None;
                let mut tools = None;
                let mut tool_choice = None;

                // fields for MCP Tools
                let mut mcp_tools = None;
                #[cfg(all(feature = "rag", feature = "index"))]
                let mut kw_search_mcp_tool = None;

                #[cfg(feature = "rag")]
                let mut context_window = None;

                // fields for vector search
                #[cfg(feature = "rag")]
                let mut vdb_server_url = None;
                #[cfg(feature = "rag")]
                let mut vdb_collection_name = None;
                #[cfg(feature = "rag")]
                let mut vdb_api_key = None;

                // fields for keyword search with kw-search-server
                #[cfg(all(feature = "rag", feature = "index"))]
                let mut kw_search_url = None;
                #[cfg(all(feature = "rag", feature = "index"))]
                let mut kw_search_index: Option<String> = None;
                #[cfg(all(feature = "rag", feature = "index"))]
                let mut kw_api_key: Option<String> = None;

                // fields for keyword search with Elasticsearch
                #[cfg(all(feature = "rag", feature = "index"))]
                let mut es_search_url = None;
                #[cfg(all(feature = "rag", feature = "index"))]
                let mut es_search_index = None;
                #[cfg(all(feature = "rag", feature = "index"))]
                let mut es_search_fields = None;
                #[cfg(all(feature = "rag", feature = "index"))]
                let mut es_api_key = None;

                // fields for keyword search with TiDB
                #[cfg(all(feature = "rag", feature = "index"))]
                let mut tidb_search_host = None;
                #[cfg(all(feature = "rag", feature = "index"))]
                let mut tidb_search_port = None;
                #[cfg(all(feature = "rag", feature = "index"))]
                let mut tidb_search_username = None;
                #[cfg(all(feature = "rag", feature = "index"))]
                let mut tidb_search_password = None;
                #[cfg(all(feature = "rag", feature = "index"))]
                let mut tidb_search_database = None;
                #[cfg(all(feature = "rag", feature = "index"))]
                let mut tidb_search_table = None;

                // fields for vector and index search results
                #[cfg(feature = "rag")]
                let mut limit = None;
                #[cfg(feature = "rag")]
                let mut score_threshold = None;
                #[cfg(all(feature = "rag", feature = "index"))]
                let mut weighted_alpha: Option<f64> = None;

                while let Some(key) = map.next_key::<String>()? {
                    #[cfg(feature = "logging")]
                    debug!(target: "stdout", "key: {key}");

                    match key.as_str() {
                        "model" => model = map.next_value()?,
                        "messages" => messages = map.next_value()?,
                        "temperature" => temperature = map.next_value()?,
                        "top_p" => top_p = map.next_value()?,
                        "n" => n_choice = map.next_value()?,
                        "stream" => stream = map.next_value()?,
                        "stream_options" => stream_options = map.next_value()?,
                        "stop" => stop = map.next_value()?,
                        "max_tokens" => max_tokens = map.next_value()?,
                        "max_completion_tokens" => max_completion_tokens = map.next_value()?,
                        "presence_penalty" => presence_penalty = map.next_value()?,
                        "frequency_penalty" => frequency_penalty = map.next_value()?,
                        "logit_bias" => logit_bias = map.next_value()?,
                        "user" => user = map.next_value()?,
                        "response_format" => response_format = map.next_value()?,
                        "tools" => tools = map.next_value()?,
                        "tool_choice" => tool_choice = map.next_value()?,
                        "mcp_tools" => mcp_tools = map.next_value()?,
                        #[cfg(all(feature = "rag", feature = "index"))]
                        "kw_search_mcp_tool" => kw_search_mcp_tool = map.next_value()?,

                        #[cfg(feature = "rag")]
                        "context_window" => context_window = map.next_value()?,
                        #[cfg(feature = "rag")]
                        "vdb_server_url" => vdb_server_url = map.next_value()?,
                        #[cfg(feature = "rag")]
                        "vdb_collection_name" => vdb_collection_name = map.next_value()?,
                        #[cfg(feature = "rag")]
                        "vdb_api_key" => vdb_api_key = map.next_value()?,

                        #[cfg(all(feature = "rag", feature = "index"))]
                        "kw_search_url" => kw_search_url = map.next_value()?,
                        #[cfg(all(feature = "rag", feature = "index"))]
                        "kw_search_index" => kw_search_index = map.next_value()?,
                        #[cfg(all(feature = "rag", feature = "index"))]
                        "kw_api_key" => kw_api_key = map.next_value()?,

                        #[cfg(all(feature = "rag", feature = "index"))]
                        "es_search_url" => es_search_url = map.next_value()?,
                        #[cfg(all(feature = "rag", feature = "index"))]
                        "es_search_index" => es_search_index = map.next_value()?,
                        #[cfg(all(feature = "rag", feature = "index"))]
                        "es_search_fields" => es_search_fields = map.next_value()?,
                        #[cfg(all(feature = "rag", feature = "index"))]
                        "es_api_key" => es_api_key = map.next_value()?,

                        #[cfg(all(feature = "rag", feature = "index"))]
                        "tidb_search_host" => tidb_search_host = map.next_value()?,
                        #[cfg(all(feature = "rag", feature = "index"))]
                        "tidb_search_port" => tidb_search_port = map.next_value()?,
                        #[cfg(all(feature = "rag", feature = "index"))]
                        "tidb_search_username" => tidb_search_username = map.next_value()?,
                        #[cfg(all(feature = "rag", feature = "index"))]
                        "tidb_search_password" => tidb_search_password = map.next_value()?,
                        #[cfg(all(feature = "rag", feature = "index"))]
                        "tidb_search_database" => tidb_search_database = map.next_value()?,
                        #[cfg(all(feature = "rag", feature = "index"))]
                        "tidb_search_table" => tidb_search_table = map.next_value()?,

                        #[cfg(feature = "rag")]
                        "limit" => limit = map.next_value()?,
                        #[cfg(feature = "rag")]
                        "score_threshold" => score_threshold = map.next_value()?,
                        #[cfg(all(feature = "rag", feature = "index"))]
                        "weighted_alpha" => weighted_alpha = map.next_value()?,
                        _ => {
                            // Ignore unknown fields
                            let _ = map.next_value::<IgnoredAny>()?;

                            #[cfg(feature = "logging")]
                            warn!(target: "stdout", "Not supported field: {key}");
                        }
                    }
                }

                // Ensure all required fields are initialized
                let messages = messages.ok_or_else(|| de::Error::missing_field("messages"))?;

                // Set default value for `max_tokens` if not provided
                if max_tokens.is_none() {
                    max_tokens = Some(-1);
                }

                // Set default value for `max_completion_tokens` if not provided
                if max_completion_tokens.is_none() {
                    max_completion_tokens = Some(-1);
                }

                // Check tools and tool_choice
                if tools.is_some() || mcp_tools.is_some() {
                    if tool_choice.is_none() {
                        tool_choice = Some(ToolChoice::None);
                    }
                } else if tool_choice.is_none() {
                    tool_choice = Some(ToolChoice::None);
                }

                #[cfg(all(feature = "rag", feature = "index"))]
                if kw_search_mcp_tool.is_some() && tool_choice.is_none() {
                    tool_choice = Some(ToolChoice::None);
                }

                if n_choice.is_none() {
                    n_choice = Some(1);
                }

                if stream.is_none() {
                    stream = Some(false);
                }

                #[cfg(all(feature = "rag", feature = "index"))]
                if let Some(name) = &kw_search_index {
                    if name.is_empty() {
                        #[cfg(feature = "logging")]
                        warn!(target: "stdout", "Found empty index name");

                        kw_search_index = None;
                    }
                }

                // Construct ChatCompletionRequest with all fields
                Ok(ChatCompletionRequest {
                    model,
                    messages,
                    temperature,
                    top_p,
                    n_choice,
                    stream,
                    stream_options,
                    stop,
                    max_tokens,
                    max_completion_tokens,
                    presence_penalty,
                    frequency_penalty,
                    logit_bias,
                    user,
                    response_format,
                    tools,
                    tool_choice,
                    mcp_tools,
                    #[cfg(all(feature = "rag", feature = "index"))]
                    kw_search_mcp_tool,
                    #[cfg(feature = "rag")]
                    context_window,
                    #[cfg(feature = "rag")]
                    vdb_server_url,
                    #[cfg(feature = "rag")]
                    vdb_collection_name,
                    #[cfg(feature = "rag")]
                    vdb_api_key,

                    #[cfg(all(feature = "rag", feature = "index"))]
                    kw_search_url,
                    #[cfg(all(feature = "rag", feature = "index"))]
                    kw_search_index,
                    #[cfg(all(feature = "rag", feature = "index"))]
                    kw_api_key,

                    #[cfg(all(feature = "rag", feature = "index"))]
                    es_search_url,
                    #[cfg(all(feature = "rag", feature = "index"))]
                    es_search_index,
                    #[cfg(all(feature = "rag", feature = "index"))]
                    es_search_fields,
                    #[cfg(all(feature = "rag", feature = "index"))]
                    es_api_key,

                    #[cfg(all(feature = "rag", feature = "index"))]
                    tidb_search_host,
                    #[cfg(all(feature = "rag", feature = "index"))]
                    tidb_search_port,
                    #[cfg(all(feature = "rag", feature = "index"))]
                    tidb_search_username,
                    #[cfg(all(feature = "rag", feature = "index"))]
                    tidb_search_password,
                    #[cfg(all(feature = "rag", feature = "index"))]
                    tidb_search_database,
                    #[cfg(all(feature = "rag", feature = "index"))]
                    tidb_search_table,

                    #[cfg(feature = "rag")]
                    limit,
                    #[cfg(feature = "rag")]
                    score_threshold,
                    #[cfg(all(feature = "rag", feature = "index"))]
                    weighted_alpha,
                })
            }
        }

        const FIELDS: &[&str] = &[
            "model",
            "messages",
            "temperature",
            "top_p",
            "n",
            "stream",
            "stream_options",
            "stop",
            "max_tokens",
            "max_completion_tokens",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "user",
            "response_format",
            "tools",
            "tool_choice",
            "mcp_tools",
            #[cfg(all(feature = "rag", feature = "index"))]
            "kw_search_mcp_tool",
            #[cfg(feature = "rag")]
            "context_window",
            #[cfg(feature = "rag")]
            "vdb_server_url",
            #[cfg(feature = "rag")]
            "vdb_collection_name",
            #[cfg(feature = "rag")]
            "vdb_api_key",
            #[cfg(all(feature = "rag", feature = "index"))]
            "kw_search_url",
            #[cfg(all(feature = "rag", feature = "index"))]
            "kw_search_index",
            #[cfg(all(feature = "rag", feature = "index"))]
            "kw_search_limit",
            #[cfg(all(feature = "rag", feature = "index"))]
            "kw_api_key",
            #[cfg(all(feature = "rag", feature = "index"))]
            "es_search_url",
            #[cfg(all(feature = "rag", feature = "index"))]
            "es_search_index",
            #[cfg(all(feature = "rag", feature = "index"))]
            "es_search_fields",
            #[cfg(all(feature = "rag", feature = "index"))]
            "es_api_key",
            #[cfg(all(feature = "rag", feature = "index"))]
            "tidb_search_host",
            #[cfg(all(feature = "rag", feature = "index"))]
            "tidb_search_port",
            #[cfg(all(feature = "rag", feature = "index"))]
            "tidb_search_username",
            #[cfg(all(feature = "rag", feature = "index"))]
            "tidb_search_password",
            #[cfg(all(feature = "rag", feature = "index"))]
            "tidb_search_database",
            #[cfg(all(feature = "rag", feature = "index"))]
            "tidb_search_table",
            #[cfg(feature = "rag")]
            "limit",
            #[cfg(feature = "rag")]
            "score_threshold",
            #[cfg(all(feature = "rag", feature = "index"))]
            "weighted_alpha",
        ];
        deserializer.deserialize_struct(
            "ChatCompletionRequest",
            FIELDS,
            ChatCompletionRequestVisitor,
        )
    }
}
#[allow(deprecated)]
impl Default for ChatCompletionRequest {
    fn default() -> Self {
        Self {
            model: None,
            messages: vec![],
            temperature: Some(0.8),
            top_p: Some(0.9),
            n_choice: Some(1),
            stream: Some(false),
            stream_options: None,
            stop: None,
            max_tokens: Some(-1),
            max_completion_tokens: Some(-1),
            presence_penalty: Some(0.0),
            frequency_penalty: Some(0.0),
            logit_bias: None,
            user: None,
            response_format: None,
            tools: None,
            tool_choice: None,
            mcp_tools: None,
            #[cfg(all(feature = "rag", feature = "index"))]
            kw_search_mcp_tool: None,
            #[cfg(feature = "rag")]
            context_window: None,

            #[cfg(feature = "rag")]
            vdb_server_url: None,
            #[cfg(feature = "rag")]
            vdb_collection_name: None,
            #[cfg(feature = "rag")]
            vdb_api_key: None,

            #[cfg(all(feature = "rag", feature = "index"))]
            kw_search_url: None,
            #[cfg(all(feature = "rag", feature = "index"))]
            kw_search_index: None,
            #[cfg(all(feature = "rag", feature = "index"))]
            kw_api_key: None,

            #[cfg(all(feature = "rag", feature = "index"))]
            es_search_url: None,
            #[cfg(all(feature = "rag", feature = "index"))]
            es_search_index: None,
            #[cfg(all(feature = "rag", feature = "index"))]
            es_search_fields: None,
            #[cfg(all(feature = "rag", feature = "index"))]
            es_api_key: None,

            #[cfg(all(feature = "rag", feature = "index"))]
            tidb_search_host: None,
            #[cfg(all(feature = "rag", feature = "index"))]
            tidb_search_port: None,
            #[cfg(all(feature = "rag", feature = "index"))]
            tidb_search_username: None,
            #[cfg(all(feature = "rag", feature = "index"))]
            tidb_search_password: None,
            #[cfg(all(feature = "rag", feature = "index"))]
            tidb_search_database: None,
            #[cfg(all(feature = "rag", feature = "index"))]
            tidb_search_table: None,

            #[cfg(feature = "rag")]
            limit: None,
            #[cfg(feature = "rag")]
            score_threshold: None,
            #[cfg(all(feature = "rag", feature = "index"))]
            weighted_alpha: None,
        }
    }
}
impl ChatCompletionRequest {
    /// Return the reference to the latest user message from the chat history.
    pub fn latest_user_message(&self) -> Option<&ChatCompletionUserMessage> {
        self.messages
            .iter()
            .rev()
            .find_map(|message| match message {
                ChatCompletionRequestMessage::User(user_message) => Some(user_message),
                _ => None,
            })
    }

    /// Return the mutable reference to the latest user message from the chat history.
    pub fn latest_user_message_mut(&mut self) -> Option<&mut ChatCompletionUserMessage> {
        self.messages
            .iter_mut()
            .rev()
            .find_map(|message| match message {
                ChatCompletionRequestMessage::User(user_message) => Some(user_message),
                _ => None,
            })
    }

    /// Return the type of the latest message from the chat history.
    pub fn latest_message_type(&self) -> Option<String> {
        self.messages.last().map(|message| match message {
            ChatCompletionRequestMessage::User(_) => "user".to_string(),
            ChatCompletionRequestMessage::Assistant(_) => "assistant".to_string(),
            ChatCompletionRequestMessage::System(_) => "system".to_string(),
            ChatCompletionRequestMessage::Tool(_) => "tool".to_string(),
        })
    }
}

#[test]
fn test_chat_serialize_chat_request() {
    #[cfg(not(feature = "index"))]
    {
        let mut messages = Vec::new();
        let system_message = ChatCompletionRequestMessage::System(
            ChatCompletionSystemMessage::new("Hello, world!", None),
        );
        messages.push(system_message);
        let user_message = ChatCompletionRequestMessage::User(ChatCompletionUserMessage::new(
            ChatCompletionUserMessageContent::Text("Hello, world!".to_string()),
            None,
        ));
        messages.push(user_message);
        let assistant_message = ChatCompletionRequestMessage::Assistant(
            ChatCompletionAssistantMessage::new(Some("Hello, world!".to_string()), None, None),
        );
        messages.push(assistant_message);
        let request = ChatCompletionRequestBuilder::new(&messages)
            .with_model("model-id")
            .with_sampling(ChatCompletionRequestSampling::Temperature(0.8))
            .with_n_choices(3)
            .enable_stream(true)
            .include_usage()
            .with_stop(vec!["stop1".to_string(), "stop2".to_string()])
            .with_presence_penalty(0.5)
            .with_frequency_penalty(0.5)
            .with_reponse_format(ChatResponseFormat::default())
            .with_tool_choice(ToolChoice::Auto)
            .build();
        let json = serde_json::to_string(&request).unwrap();
        assert_eq!(
            json,
            r#"{"model":"model-id","messages":[{"role":"system","content":"Hello, world!"},{"role":"user","content":"Hello, world!"},{"role":"assistant","content":"Hello, world!"}],"temperature":0.8,"top_p":1.0,"n":3,"stream":true,"stream_options":{"include_usage":true},"stop":["stop1","stop2"],"max_tokens":-1,"max_completion_tokens":-1,"presence_penalty":0.5,"frequency_penalty":0.5,"response_format":{"type":"text"},"tool_choice":"auto"}"#
        );
    }

    #[cfg(not(feature = "index"))]
    {
        let mut messages = Vec::new();
        let system_message = ChatCompletionRequestMessage::System(
            ChatCompletionSystemMessage::new("Hello, world!", None),
        );
        messages.push(system_message);

        let user_message_content = ChatCompletionUserMessageContent::Parts(vec![
            ContentPart::Text(TextContentPart::new("what is in the picture?")),
            ContentPart::Image(ImageContentPart::new(Image {
                url: "https://example.com/image.png".to_string(),
                detail: None,
            })),
        ]);
        let user_message =
            ChatCompletionRequestMessage::new_user_message(user_message_content, None);
        messages.push(user_message);

        let request = ChatCompletionRequestBuilder::new(&messages)
            .with_model("model-id")
            .with_tool_choice(ToolChoice::None)
            .build();
        let json = serde_json::to_string(&request).unwrap();
        assert_eq!(
            json,
            r#"{"model":"model-id","messages":[{"role":"system","content":"Hello, world!"},{"role":"user","content":[{"type":"text","text":"what is in the picture?"},{"type":"image_url","image_url":{"url":"https://example.com/image.png"}}]}],"temperature":0.8,"top_p":0.9,"n":1,"stream":false,"max_tokens":-1,"max_completion_tokens":-1,"presence_penalty":0.0,"frequency_penalty":0.0,"tool_choice":"none"}"#
        );
    }

    #[cfg(not(feature = "index"))]
    {
        let mut messages = Vec::new();
        let system_message = ChatCompletionRequestMessage::System(
            ChatCompletionSystemMessage::new("Hello, world!", None),
        );
        messages.push(system_message);
        let user_message = ChatCompletionRequestMessage::User(ChatCompletionUserMessage::new(
            ChatCompletionUserMessageContent::Text("Hello, world!".to_string()),
            None,
        ));
        messages.push(user_message);
        let assistant_message = ChatCompletionRequestMessage::Assistant(
            ChatCompletionAssistantMessage::new(Some("Hello, world!".to_string()), None, None),
        );
        messages.push(assistant_message);

        let json_str = r###"{
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "definitions": {
                        "TemperatureUnit": {
                            "enum": [
                                "celsius",
                                "fahrenheit"
                            ],
                            "type": "string"
                        }
                    },
                    "properties": {
                        "api_key": {
                            "description": "the OpenWeatherMap API key to use. If not provided, the server will use the OPENWEATHERMAP_API_KEY environment variable.",
                            "type": [
                                "string",
                                "null"
                            ]
                        },
                        "location": {
                            "description": "the city to get the weather for, e.g., 'Beijing', 'New York', 'Tokyo'",
                            "type": "string"
                        },
                        "unit": {
                            "allOf": [
                                {
                                    "$ref": "#/definitions/TemperatureUnit"
                                }
                            ],
                            "description": "the unit to use for the temperature, e.g., 'celsius', 'fahrenheit'"
                        }
                    },
                    "required": [
                        "location",
                        "unit"
                    ],
                    "title": "GetWeatherRequest",
                    "type": "object"
                }"###;
        let json_schema: JsonObject = serde_json::from_str(json_str).unwrap();

        let tool = Tool {
            ty: ToolType::Function,
            function: ToolFunction {
                name: "my_function".to_string(),
                description: None,
                parameters: Some(json_schema),
            },
        };

        let mcp_tool = McpTool {
            ty: ToolType::Mcp,
            server_label: "test".to_string(),
            server_url: "https://test.com".to_string(),
            transport: McpTransport::Sse,
            allowed_tools: Some(vec!["test".to_string()]),
            headers: Some(HashMap::from([(
                "Authorization".to_string(),
                "Bearer token".to_string(),
            )])),
        };

        let request = ChatCompletionRequestBuilder::new(&messages)
            .with_model("model-id")
            .with_sampling(ChatCompletionRequestSampling::Temperature(0.8))
            .with_n_choices(3)
            .enable_stream(true)
            .include_usage()
            .with_stop(vec!["stop1".to_string(), "stop2".to_string()])
            .with_max_completion_tokens(100)
            .with_presence_penalty(0.5)
            .with_frequency_penalty(0.5)
            .with_reponse_format(ChatResponseFormat::default())
            .with_tools(vec![tool])
            .with_mcp_tools(vec![mcp_tool])
            .with_tool_choice(ToolChoice::Tool(ToolChoiceTool {
                ty: ToolType::Function,
                function: ToolChoiceToolFunction {
                    name: "my_function".to_string(),
                },
            }))
            .build();
        let json = serde_json::to_string(&request).unwrap();
        assert_eq!(
            json,
            r###"{"model":"model-id","messages":[{"role":"system","content":"Hello, world!"},{"role":"user","content":"Hello, world!"},{"role":"assistant","content":"Hello, world!"}],"temperature":0.8,"top_p":1.0,"n":3,"stream":true,"stream_options":{"include_usage":true},"stop":["stop1","stop2"],"max_tokens":-1,"max_completion_tokens":100,"presence_penalty":0.5,"frequency_penalty":0.5,"response_format":{"type":"text"},"tools":[{"type":"function","function":{"name":"my_function","parameters":{"$schema":"http://json-schema.org/draft-07/schema#","definitions":{"TemperatureUnit":{"enum":["celsius","fahrenheit"],"type":"string"}},"properties":{"api_key":{"description":"the OpenWeatherMap API key to use. If not provided, the server will use the OPENWEATHERMAP_API_KEY environment variable.","type":["string","null"]},"location":{"description":"the city to get the weather for, e.g., 'Beijing', 'New York', 'Tokyo'","type":"string"},"unit":{"allOf":[{"$ref":"#/definitions/TemperatureUnit"}],"description":"the unit to use for the temperature, e.g., 'celsius', 'fahrenheit'"}},"required":["location","unit"],"title":"GetWeatherRequest","type":"object"}}}],"tool_choice":{"type":"function","function":{"name":"my_function"}},"mcp_tools":[{"type":"mcp","server_label":"test","server_url":"https://test.com","transport":"sse","allowed_tools":["test"],"headers":{"Authorization":"Bearer token"}}]}"###
        );
    }

    #[cfg(not(feature = "index"))]
    {
        let mut messages = Vec::new();
        let system_message = ChatCompletionRequestMessage::System(
            ChatCompletionSystemMessage::new("Hello, world!", None),
        );
        messages.push(system_message);
        let user_message = ChatCompletionRequestMessage::User(ChatCompletionUserMessage::new(
            ChatCompletionUserMessageContent::Text("Hello, world!".to_string()),
            None,
        ));
        messages.push(user_message);
        let assistant_message = ChatCompletionRequestMessage::Assistant(
            ChatCompletionAssistantMessage::new(Some("Hello, world!".to_string()), None, None),
        );
        messages.push(assistant_message);

        let tool = Tool {
            ty: ToolType::Function,
            function: ToolFunction {
                name: "my_function".to_string(),
                description: None,
                parameters: None,
            },
        };

        let request = ChatCompletionRequestBuilder::new(&messages)
            .with_model("model-id")
            .with_sampling(ChatCompletionRequestSampling::Temperature(0.8))
            .with_n_choices(3)
            .enable_stream(true)
            .include_usage()
            .with_stop(vec!["stop1".to_string(), "stop2".to_string()])
            .with_max_completion_tokens(100)
            .with_presence_penalty(0.5)
            .with_frequency_penalty(0.5)
            .with_reponse_format(ChatResponseFormat::default())
            .with_tools(vec![tool])
            .with_tool_choice(ToolChoice::Auto)
            .build();
        let json = serde_json::to_string(&request).unwrap();
        assert_eq!(
            json,
            r#"{"model":"model-id","messages":[{"role":"system","content":"Hello, world!"},{"role":"user","content":"Hello, world!"},{"role":"assistant","content":"Hello, world!"}],"temperature":0.8,"top_p":1.0,"n":3,"stream":true,"stream_options":{"include_usage":true},"stop":["stop1","stop2"],"max_tokens":-1,"max_completion_tokens":100,"presence_penalty":0.5,"frequency_penalty":0.5,"response_format":{"type":"text"},"tools":[{"type":"function","function":{"name":"my_function"}}],"tool_choice":"auto"}"#
        );
    }

    #[cfg(all(feature = "rag", not(feature = "index")))]
    {
        let mut messages = Vec::new();
        let system_message = ChatCompletionRequestMessage::System(
            ChatCompletionSystemMessage::new("Hello, world!", None),
        );
        messages.push(system_message);
        let user_message = ChatCompletionRequestMessage::User(ChatCompletionUserMessage::new(
            ChatCompletionUserMessageContent::Text("Hello, world!".to_string()),
            None,
        ));
        messages.push(user_message);
        let assistant_message = ChatCompletionRequestMessage::Assistant(
            ChatCompletionAssistantMessage::new(Some("Hello, world!".to_string()), None, None),
        );
        messages.push(assistant_message);

        let params_json = r###"{
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "properties": {
                        "a": {
                            "description": "the left hand side number",
                            "format": "int32",
                            "type": "integer"
                        },
                        "b": {
                            "description": "the right hand side number",
                            "format": "int32",
                            "type": "integer"
                        }
                    },
                    "required": [
                        "a",
                        "b"
                    ],
                    "title": "SumRequest",
                    "type": "object"
                }"###;
        let params: JsonObject = serde_json::from_str(params_json).unwrap();

        let tool = Tool {
            ty: ToolType::Function,
            function: ToolFunction {
                name: "my_function".to_string(),
                description: None,
                parameters: Some(params),
            },
        };

        let request = ChatCompletionRequestBuilder::new(&messages)
            .with_model("model-id")
            .with_sampling(ChatCompletionRequestSampling::Temperature(0.8))
            .with_n_choices(3)
            .enable_stream(true)
            .include_usage()
            .with_stop(vec!["stop1".to_string(), "stop2".to_string()])
            .with_max_completion_tokens(100)
            .with_presence_penalty(0.5)
            .with_frequency_penalty(0.5)
            .with_reponse_format(ChatResponseFormat::default())
            .with_tools(vec![tool])
            .with_tool_choice(ToolChoice::Auto)
            .with_rag_context_window(3)
            .with_vector_search_settings(
                "http://localhost:6333",
                &["collection1".to_string(), "collection2".to_string()],
                None,
            )
            .with_vector_search_filter(10u64, 0.5f32)
            .build();
        let json = serde_json::to_string(&request).unwrap();
        assert_eq!(
            json,
            r#"{"model":"model-id","messages":[{"role":"system","content":"Hello, world!"},{"role":"user","content":"Hello, world!"},{"role":"assistant","content":"Hello, world!"}],"temperature":0.8,"top_p":1.0,"n":3,"stream":true,"stream_options":{"include_usage":true},"stop":["stop1","stop2"],"max_tokens":-1,"max_completion_tokens":100,"presence_penalty":0.5,"frequency_penalty":0.5,"response_format":{"type":"text"},"tools":[{"type":"function","function":{"name":"my_function","parameters":{"$schema":"http://json-schema.org/draft-07/schema#","properties":{"a":{"description":"the left hand side number","format":"int32","type":"integer"},"b":{"description":"the right hand side number","format":"int32","type":"integer"}},"required":["a","b"],"title":"SumRequest","type":"object"}}}],"tool_choice":"auto","context_window":3,"vdb_server_url":"http://localhost:6333","vdb_collection_name":["collection1","collection2"],"limit":10,"score_threshold":0.5}"#
        );
    }

    #[cfg(all(feature = "rag", feature = "index"))]
    {
        let mut messages = Vec::new();
        let system_message = ChatCompletionRequestMessage::System(
            ChatCompletionSystemMessage::new("Hello, world!", None),
        );
        messages.push(system_message);
        let user_message = ChatCompletionRequestMessage::User(ChatCompletionUserMessage::new(
            ChatCompletionUserMessageContent::Text("Hello, world!".to_string()),
            None,
        ));
        messages.push(user_message);
        let assistant_message = ChatCompletionRequestMessage::Assistant(
            ChatCompletionAssistantMessage::new(Some("Hello, world!".to_string()), None, None),
        );
        messages.push(assistant_message);

        let params_json = r###"{
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "properties": {
                        "a": {
                            "description": "the left hand side number",
                            "format": "int32",
                            "type": "integer"
                        },
                        "b": {
                            "description": "the right hand side number",
                            "format": "int32",
                            "type": "integer"
                        }
                    },
                    "required": [
                        "a",
                        "b"
                    ],
                    "title": "SumRequest",
                    "type": "object"
                }"###;
        let params: JsonObject = serde_json::from_str(params_json).unwrap();

        let tool = Tool {
            ty: ToolType::Function,
            function: ToolFunction {
                name: "my_function".to_string(),
                description: None,
                parameters: Some(params),
            },
        };

        let request = ChatCompletionRequestBuilder::new(&messages)
            .with_model("model-id")
            .with_sampling(ChatCompletionRequestSampling::Temperature(0.8))
            .with_n_choices(3)
            .enable_stream(true)
            .include_usage()
            .with_stop(vec!["stop1".to_string(), "stop2".to_string()])
            .with_max_completion_tokens(100)
            .with_presence_penalty(0.5)
            .with_frequency_penalty(0.5)
            .with_reponse_format(ChatResponseFormat::default())
            .with_tools(vec![tool])
            .with_tool_choice(ToolChoice::Auto)
            .with_rag_context_window(3)
            .with_vector_search_settings(
                "http://localhost:6333",
                &["collection1".to_string(), "collection2".to_string()],
                None,
            )
            .with_kw_search_settings("http://localhost:9069", "index-name", None)
            .with_search_filter(10u64, 0.5f32, 0.5f64)
            .build();
        let json = serde_json::to_string(&request).unwrap();
        assert_eq!(
            json,
            r#"{"model":"model-id","messages":[{"role":"system","content":"Hello, world!"},{"role":"user","content":"Hello, world!"},{"role":"assistant","content":"Hello, world!"}],"temperature":0.8,"top_p":1.0,"n":3,"stream":true,"stream_options":{"include_usage":true},"stop":["stop1","stop2"],"max_tokens":-1,"max_completion_tokens":100,"presence_penalty":0.5,"frequency_penalty":0.5,"response_format":{"type":"text"},"tools":[{"type":"function","function":{"name":"my_function","parameters":{"$schema":"http://json-schema.org/draft-07/schema#","properties":{"a":{"description":"the left hand side number","format":"int32","type":"integer"},"b":{"description":"the right hand side number","format":"int32","type":"integer"}},"required":["a","b"],"title":"SumRequest","type":"object"}}}],"tool_choice":"auto","context_window":3,"vdb_server_url":"http://localhost:6333","vdb_collection_name":["collection1","collection2"],"kw_search_url":"http://localhost:9069","kw_search_index":"index-name","limit":10,"score_threshold":0.5,"weighted_alpha":0.5}"#
        );
    }
}

#[test]
fn test_chat_deserialize_chat_request() {
    {
        let json = r#"{"model":"model-id","messages":[{"role":"system","content":"Hello, world!"},{"role":"user","content":"Hello, world!"},{"role":"assistant","content":"Hello, world!"}],"temperature":0.8,"top_p":1.0,"n":3,"stream":true,"stop":["stop1","stop2"],"presence_penalty":0.5,"frequency_penalty":0.5,"response_format":{"type":"text"}}"#;
        let request: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.model, Some("model-id".to_string()));
        assert_eq!(request.messages.len(), 3);
        assert_eq!(
            request.messages[0],
            ChatCompletionRequestMessage::System(ChatCompletionSystemMessage::new(
                "Hello, world!",
                None
            ))
        );
        assert_eq!(
            request.messages[1],
            ChatCompletionRequestMessage::User(ChatCompletionUserMessage::new(
                ChatCompletionUserMessageContent::Text("Hello, world!".to_string()),
                None
            ))
        );
        assert_eq!(
            request.messages[2],
            ChatCompletionRequestMessage::Assistant(ChatCompletionAssistantMessage::new(
                Some("Hello, world!".to_string()),
                None,
                None
            ))
        );
        assert_eq!(request.temperature, Some(0.8));
        assert_eq!(request.top_p, Some(1.0));
        assert_eq!(request.n_choice, Some(3));
        assert_eq!(request.stream, Some(true));
        assert_eq!(
            request.stop,
            Some(vec!["stop1".to_string(), "stop2".to_string()])
        );
        assert_eq!(request.max_completion_tokens, Some(-1));
        assert_eq!(request.presence_penalty, Some(0.5));
        assert_eq!(request.frequency_penalty, Some(0.5));
        assert_eq!(request.tool_choice, Some(ToolChoice::None));
    }

    {
        let json = r#"{"model":"model-id","messages":[{"role":"system","content":"Hello, world!"},{"role":"user","content":"Hello, world!"},{"role":"assistant","content":"Hello, world!"}],"temperature":0.8,"top_p":1.0,"n":3,"stream":true,"stop":["stop1","stop2"],"max_completion_tokens":100,"presence_penalty":0.5,"frequency_penalty":0.5,"response_format":{"type":"text"},"tool_choice":"auto"}"#;
        let request: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.model, Some("model-id".to_string()));
        assert_eq!(request.messages.len(), 3);
        assert_eq!(request.temperature, Some(0.8));
        assert_eq!(request.top_p, Some(1.0));
        assert_eq!(request.n_choice, Some(3));
        assert_eq!(request.stream, Some(true));
        assert_eq!(
            request.stop,
            Some(vec!["stop1".to_string(), "stop2".to_string()])
        );
        assert_eq!(request.max_completion_tokens, Some(100));
        assert_eq!(request.presence_penalty, Some(0.5));
        assert_eq!(request.frequency_penalty, Some(0.5));
        assert_eq!(request.tool_choice, Some(ToolChoice::Auto));
    }

    {
        let json = r#"{"model":"model-id","messages":[{"role":"system","content":"Hello, world!"},{"role":"user","content":"Hello, world!"},{"role":"assistant","content":"Hello, world!"}],"temperature":0.8,"top_p":1.0,"n":3,"stream":true,"stop":["stop1","stop2"],"max_completion_tokens":100,"presence_penalty":0.5,"frequency_penalty":0.5,"response_format":{"type":"text"},"tool_choice":{"type":"function","function":{"name":"my_function"}}}"#;
        let request: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.model, Some("model-id".to_string()));
        assert_eq!(request.messages.len(), 3);
        assert_eq!(request.temperature, Some(0.8));
        assert_eq!(request.top_p, Some(1.0));
        assert_eq!(request.n_choice, Some(3));
        assert_eq!(request.stream, Some(true));
        assert_eq!(
            request.stop,
            Some(vec!["stop1".to_string(), "stop2".to_string()])
        );
        assert_eq!(request.max_completion_tokens, Some(100));
        assert_eq!(request.presence_penalty, Some(0.5));
        assert_eq!(request.frequency_penalty, Some(0.5));
        assert_eq!(
            request.tool_choice,
            Some(ToolChoice::Tool(ToolChoiceTool {
                ty: ToolType::Function,
                function: ToolChoiceToolFunction {
                    name: "my_function".to_string(),
                },
            }))
        );
    }

    {
        let json = r#"{"model":"model-id","messages":[{"role":"system","content":"Hello, world!"},{"role":"user","content":"Hello, world!"},{"role":"assistant","content":"Hello, world!"}],"temperature":0.8,"top_p":1.0,"n":3,"stream":true,"stream_options":{"include_usage":true},"stop":["stop1","stop2"],"max_completion_tokens":100,"presence_penalty":0.5,"frequency_penalty":0.5,"response_format":{"type":"text"},"tools":[{"type":"function","function":{"name":"my_function","parameters":{"type":"object","properties":{"location":{"type":"string","description":"The city and state, e.g. San Francisco, CA"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["location"]}}}]}"#;

        let request: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        let tool_choice = request.tool_choice.unwrap();
        assert_eq!(tool_choice, ToolChoice::None);
    }

    {
        let json = r#"{"model":"model-id","messages":[{"role":"system","content":"Hello, world!"},{"role":"user","content":"Hello, world!"},{"role":"assistant","content":"Hello, world!"}],"temperature":0.8,"top_p":1.0,"n":3,"stream":true,"stream_options":{"include_usage":true},"stop":["stop1","stop2"],"max_completion_tokens":100,"presence_penalty":0.5,"frequency_penalty":0.5,"response_format":{"type":"text"}}"#;

        let request: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        let tool_choice = request.tool_choice.unwrap();
        assert_eq!(tool_choice, ToolChoice::None);
    }

    {
        let json_str = r###"{
    "model": "Llama-3-Groq-8B",
    "messages": [
        {
            "role": "user",
            "content": "How is the weather of Beijing, China? In Celsius."
        }
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "sum",
                "description": "Calculate the sum of two numbers",
                "parameters": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "properties": {
                        "a": {
                            "description": "the left hand side number",
                            "format": "int32",
                            "type": "integer"
                        },
                        "b": {
                            "description": "the right hand side number",
                            "format": "int32",
                            "type": "integer"
                        }
                    },
                    "required": [
                        "a",
                        "b"
                    ],
                    "title": "SumRequest",
                    "type": "object"
                }
            }
        }
    ],
    "mcp_tools": [
        {
            "type": "mcp",
            "server_label": "test",
            "server_url": "https://test.com",
            "transport": "sse",
            "allowed_tools": ["test"],
            "headers": {
                "Authorization": "Bearer token"
            }
        }
    ],
    "tool_choice": "required",
    "stream": false
}"###;

        let request: ChatCompletionRequest = serde_json::from_str(json_str).unwrap();
        assert!(request.model.is_some());
        let mcp_tools = request.mcp_tools.unwrap();
        assert!(mcp_tools.len() == 1);
        let mcp_tool = &mcp_tools[0];
        assert_eq!(mcp_tool.ty, ToolType::Mcp);
        assert_eq!(mcp_tool.server_label, "test");
        assert_eq!(mcp_tool.server_url, "https://test.com");
        assert_eq!(mcp_tool.transport, McpTransport::Sse);
        assert_eq!(mcp_tool.allowed_tools, Some(vec!["test".to_string()]));
        assert_eq!(
            mcp_tool.headers,
            Some(HashMap::from([(
                "Authorization".to_string(),
                "Bearer token".to_string(),
            )]))
        );
        let tools = request.tools.unwrap();
        assert!(tools.len() == 1);
        let tool = &tools[0];
        assert_eq!(tool.ty, ToolType::Function);
        assert_eq!(tool.function.name, "sum");
        assert!(tool.function.parameters.is_some());
        let params = tool.function.parameters.as_ref().unwrap();
        assert!(params.contains_key("properties"));
        let properties = params.get("properties").unwrap();
        let properties = properties.as_object().unwrap();
        assert!(properties.len() == 2);
        assert!(properties.contains_key("a"));
        assert!(properties.contains_key("b"));
        let a = properties.get("a").unwrap();
        let a = a.as_object().unwrap();
        assert!(a.contains_key("description"));
        assert_eq!(
            a.get("description").unwrap().as_str().unwrap(),
            "the left hand side number"
        );
        assert!(a.contains_key("format"));
        assert_eq!(a.get("format").unwrap().as_str().unwrap(), "int32");
        assert!(a.contains_key("type"));
        assert_eq!(a.get("type").unwrap().as_str().unwrap(), "integer");
        let b = properties.get("b").unwrap();
        let b = b.as_object().unwrap();
        assert!(b.contains_key("description"));
        assert_eq!(
            b.get("description").unwrap().as_str().unwrap(),
            "the right hand side number"
        );
        assert!(b.contains_key("format"));
        assert_eq!(b.get("format").unwrap().as_str().unwrap(), "int32");
        assert!(b.contains_key("type"));
        assert_eq!(b.get("type").unwrap().as_str().unwrap(), "integer");
    }

    #[cfg(feature = "rag")]
    {
        let json = r#"{"model":"model-id","messages":[{"role":"system","content":"Hello, world!"},{"role":"user","content":"Hello, world!"},{"role":"assistant","content":"Hello, world!"}],"temperature":0.8,"top_p":1.0,"n":3,"stream":true,"stream_options":{"include_usage":true},"stop":["stop1","stop2"],"max_completion_tokens":100,"presence_penalty":0.5,"frequency_penalty":0.5,"response_format":{"type":"text"},"vdb_server_url":"http://localhost:6333","vdb_collection_name":["collection1","collection2"],"limit":10,"score_threshold":0.5}"#;

        let request: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        let tool_choice = request.tool_choice.unwrap();
        assert_eq!(tool_choice, ToolChoice::None);
        assert_eq!(
            request.vdb_server_url,
            Some("http://localhost:6333".to_string())
        );
        assert_eq!(
            request.vdb_collection_name,
            Some(vec!["collection1".to_string(), "collection2".to_string()])
        );
        assert_eq!(request.limit, Some(10));
        assert_eq!(request.score_threshold, Some(0.5));
    }

    #[cfg(all(feature = "rag", feature = "index"))]
    {
        let json = r#"{"model":"model-id","messages":[{"role":"system","content":"Hello, world!"},{"role":"user","content":"Hello, world!"},{"role":"assistant","content":"Hello, world!"}],"temperature":0.8,"top_p":1.0,"n":3,"stream":true,"stream_options":{"include_usage":true},"stop":["stop1","stop2"],"max_completion_tokens":100,"presence_penalty":0.5,"frequency_penalty":0.5,"response_format":{"type":"text"},"vdb_server_url":"http://localhost:6333","vdb_collection_name":["collection1","collection2"],"kw_search_url":"http://localhost:9069","kw_search_index":"index-name"}"#;

        let request: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        let tool_choice = request.tool_choice.unwrap();
        assert_eq!(tool_choice, ToolChoice::None);
        assert_eq!(
            request.vdb_server_url,
            Some("http://localhost:6333".to_string())
        );
        assert_eq!(
            request.vdb_collection_name,
            Some(vec!["collection1".to_string(), "collection2".to_string()])
        );
        assert_eq!(
            request.kw_search_url,
            Some("http://localhost:9069".to_string())
        );
        assert_eq!(request.kw_search_index, Some("index-name".to_string()));
    }
}

/// An object specifying the format that the model must output.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatResponseFormat {
    /// Must be one of `text`` or `json_object`. Defaults to `text`.
    #[serde(rename = "type")]
    pub ty: String,
}
impl Default for ChatResponseFormat {
    fn default() -> Self {
        Self {
            ty: "text".to_string(),
        }
    }
}

#[test]
fn test_chat_serialize_response_format() {
    let response_format = ChatResponseFormat {
        ty: "text".to_string(),
    };
    let json = serde_json::to_string(&response_format).unwrap();
    assert_eq!(json, r#"{"type":"text"}"#);

    let response_format = ChatResponseFormat {
        ty: "json_object".to_string(),
    };
    let json = serde_json::to_string(&response_format).unwrap();
    assert_eq!(json, r#"{"type":"json_object"}"#);
}

/// Options for streaming response. Only set this when you set stream: `true`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct StreamOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_usage: Option<bool>,
}

/// Controls which (if any) function is called by the model. Defaults to `None`.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub enum ToolChoice {
    /// The model will not call a function and instead generates a message.
    #[serde(rename = "none")]
    None,
    /// The model can pick between generating a message or calling a function.
    #[serde(rename = "auto")]
    Auto,
    /// The model must call one or more tools.
    #[serde(rename = "required")]
    Required,
    /// Specifies a tool the model should use. Use to force the model to call a specific function.
    #[serde(untagged)]
    Tool(ToolChoiceTool),
}
impl Default for ToolChoice {
    fn default() -> Self {
        Self::None
    }
}

#[test]
fn test_chat_serialize_tool_choice() {
    let tool_choice = ToolChoice::None;
    let json = serde_json::to_string(&tool_choice).unwrap();
    assert_eq!(json, r#""none""#);

    let tool_choice = ToolChoice::Auto;
    let json = serde_json::to_string(&tool_choice).unwrap();
    assert_eq!(json, r#""auto""#);

    let tool_choice = ToolChoice::Tool(ToolChoiceTool {
        ty: ToolType::Function,
        function: ToolChoiceToolFunction {
            name: "my_function".to_string(),
        },
    });
    let json = serde_json::to_string(&tool_choice).unwrap();
    assert_eq!(
        json,
        r#"{"type":"function","function":{"name":"my_function"}}"#
    );
}

#[test]
fn test_chat_deserialize_tool_choice() {
    let json = r#""none""#;
    let tool_choice: ToolChoice = serde_json::from_str(json).unwrap();
    assert_eq!(tool_choice, ToolChoice::None);

    let json = r#""auto""#;
    let tool_choice: ToolChoice = serde_json::from_str(json).unwrap();
    assert_eq!(tool_choice, ToolChoice::Auto);

    let json = r#"{"type":"function","function":{"name":"my_function"}}"#;
    let tool_choice: ToolChoice = serde_json::from_str(json).unwrap();
    assert_eq!(
        tool_choice,
        ToolChoice::Tool(ToolChoiceTool {
            ty: ToolType::Function,
            function: ToolChoiceToolFunction {
                name: "my_function".to_string(),
            },
        })
    );
}

/// A tool the model should use.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct ToolChoiceTool {
    /// The type of the tool. Currently, only `function` is supported.
    #[serde(rename = "type")]
    pub ty: ToolType,
    /// The function the model calls.
    pub function: ToolChoiceToolFunction,
}

/// Represents a tool the model should use.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct ToolChoiceToolFunction {
    /// The name of the function to call.
    pub name: String,
}

/// Represents a tool the model may generate JSON inputs for.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Tool {
    /// The type of the tool. Currently, only `function` is supported.
    #[serde(rename = "type")]
    pub ty: ToolType,
    /// Function the model may generate JSON inputs for.
    pub function: ToolFunction,
}
impl Tool {
    pub fn new(function: ToolFunction) -> Self {
        Self {
            ty: ToolType::Function,
            function,
        }
    }
}

#[test]
fn test_chat_serialize_tool() {
    {
        let tool = Tool {
            ty: ToolType::Function,
            function: ToolFunction {
                name: "my_function".to_string(),
                description: None,
                parameters: None,
            },
        };
        let json = serde_json::to_string(&tool).unwrap();
        assert_eq!(
            json,
            r#"{"type":"function","function":{"name":"my_function"}}"#
        );
    }

    {
        let input_schema_str = r###"{
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "properties": {
                        "a": {
                            "description": "the left hand side number",
                            "format": "int32",
                            "type": "integer"
                        },
                        "b": {
                            "description": "the right hand side number",
                            "format": "int32",
                            "type": "integer"
                        }
                    },
                    "required": [
                        "a",
                        "b"
                    ],
                    "title": "SumRequest",
                    "type": "object"
                }"###;
        let input_schema: JsonObject = serde_json::from_str(input_schema_str).unwrap();

        let tool = Tool {
            ty: ToolType::Function,
            function: ToolFunction {
                name: "sum".to_string(),
                description: Some("Calculate the sum of two numbers".to_string()),
                parameters: Some(input_schema),
            },
        };
        let json = serde_json::to_string(&tool).unwrap();
        assert_eq!(
            json,
            r###"{"type":"function","function":{"name":"sum","description":"Calculate the sum of two numbers","parameters":{"$schema":"http://json-schema.org/draft-07/schema#","properties":{"a":{"description":"the left hand side number","format":"int32","type":"integer"},"b":{"description":"the right hand side number","format":"int32","type":"integer"}},"required":["a","b"],"title":"SumRequest","type":"object"}}}"###
        );
    }

    {
        let tool1_json_str = r###"{
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "definitions": {
                        "TemperatureUnit": {
                            "enum": [
                                "celsius",
                                "fahrenheit"
                            ],
                            "type": "string"
                        }
                    },
                    "properties": {
                        "api_key": {
                            "description": "the OpenWeatherMap API key to use. If not provided, the server will use the OPENWEATHERMAP_API_KEY environment variable.",
                            "type": [
                                "string",
                                "null"
                            ]
                        },
                        "location": {
                            "description": "the city to get the weather for, e.g., 'Beijing', 'New York', 'Tokyo'",
                            "type": "string"
                        },
                        "unit": {
                            "allOf": [
                                {
                                    "$ref": "#/definitions/TemperatureUnit"
                                }
                            ],
                            "description": "the unit to use for the temperature, e.g., 'celsius', 'fahrenheit'"
                        }
                    },
                    "required": [
                        "location",
                        "unit"
                    ],
                    "title": "GetWeatherRequest",
                    "type": "object"
                }
            }
        }"###;
        let tool_1: Tool = serde_json::from_str(tool1_json_str).unwrap();

        let tool2_json_str = r###"{
            "type": "function",
            "function": {
                "name": "sum",
                "description": "Calculate the sum of two numbers",
                "parameters": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "properties": {
                        "a": {
                            "description": "the left hand side number",
                            "format": "int32",
                            "type": "integer"
                        },
                        "b": {
                            "description": "the right hand side number",
                            "format": "int32",
                            "type": "integer"
                        }
                    },
                    "required": [
                        "a",
                        "b"
                    ],
                    "title": "SumRequest",
                    "type": "object"
                }
            }
        }"###;
        let tool_2: Tool = serde_json::from_str(tool2_json_str).unwrap();

        let tools = vec![tool_1, tool_2];
        let json_str = serde_json::to_string(&tools).unwrap();
        assert_eq!(
            json_str,
            r###"[{"type":"function","function":{"name":"get_current_weather","description":"Get the current weather in a given location","parameters":{"$schema":"http://json-schema.org/draft-07/schema#","definitions":{"TemperatureUnit":{"enum":["celsius","fahrenheit"],"type":"string"}},"properties":{"api_key":{"description":"the OpenWeatherMap API key to use. If not provided, the server will use the OPENWEATHERMAP_API_KEY environment variable.","type":["string","null"]},"location":{"description":"the city to get the weather for, e.g., 'Beijing', 'New York', 'Tokyo'","type":"string"},"unit":{"allOf":[{"$ref":"#/definitions/TemperatureUnit"}],"description":"the unit to use for the temperature, e.g., 'celsius', 'fahrenheit'"}},"required":["location","unit"],"title":"GetWeatherRequest","type":"object"}}},{"type":"function","function":{"name":"sum","description":"Calculate the sum of two numbers","parameters":{"$schema":"http://json-schema.org/draft-07/schema#","properties":{"a":{"description":"the left hand side number","format":"int32","type":"integer"},"b":{"description":"the right hand side number","format":"int32","type":"integer"}},"required":["a","b"],"title":"SumRequest","type":"object"}}}]"###
        );
    }
}

#[test]
fn test_chat_deserialize_tool() {
    use std::any::{Any, TypeId};

    let json = r###"{
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "definitions": {
                "TemperatureUnit": {
                    "enum": [
                        "celsius",
                        "fahrenheit"
                    ],
                    "type": "string"
                }
            },
            "properties": {
                "api_key": {
                    "description": "the OpenWeatherMap API key to use. If not provided, the server will use the OPENWEATHERMAP_API_KEY environment variable.",
                    "type": [
                        "string",
                        "null"
                    ]
                },
                "location": {
                    "description": "the city to get the weather for, e.g., 'Beijing', 'New York', 'Tokyo'",
                    "type": "string"
                },
                "unit": {
                    "allOf": [
                        {
                            "$ref": "#/definitions/TemperatureUnit"
                        }
                    ],
                    "description": "the unit to use for the temperature, e.g., 'celsius', 'fahrenheit'"
                }
            },
            "required": [
                "location",
                "unit"
            ],
            "title": "GetWeatherRequest",
            "type": "object"
        }
    }
}"###;
    let tool: Tool = serde_json::from_str(json).unwrap();
    assert_eq!(tool.ty, ToolType::Function);
    assert_eq!(tool.function.name, "get_current_weather");
    assert!(tool.function.description.is_some());
    assert!(tool.function.parameters.is_some());
    let params = tool.function.parameters.as_ref().unwrap();
    assert_eq!(params.type_id(), TypeId::of::<JsonObject>());
    assert!(params.contains_key("$schema"));
    assert!(params.contains_key("definitions"));
    let definitions = params.get("definitions").unwrap();
    let definitions = definitions.as_object().unwrap();
    assert_eq!(definitions.len(), 1);
    assert!(definitions.contains_key("TemperatureUnit"));
    let temperature_unit = definitions.get("TemperatureUnit").unwrap();
    let temperature_unit = temperature_unit.as_object().unwrap();
    assert_eq!(temperature_unit.len(), 2);
    assert!(temperature_unit.contains_key("enum"));
    let enum_values = temperature_unit.get("enum").unwrap();
    let enum_values = enum_values.as_array().unwrap();
    assert_eq!(enum_values.len(), 2);
    assert_eq!(enum_values[0].as_str().unwrap(), "celsius");
    assert_eq!(enum_values[1].as_str().unwrap(), "fahrenheit");
    assert!(params.contains_key("properties"));
    let properties = params.get("properties").unwrap();
    let properties = properties.as_object().unwrap();
    assert_eq!(properties.len(), 3);
    assert!(properties.contains_key("unit"));
    assert!(properties.contains_key("location"));
    assert!(properties.contains_key("api_key"));
    let unit = properties.get("unit").unwrap();
    let unit = unit.as_object().unwrap();
    assert!(unit.contains_key("allOf"));
    let all_of = unit.get("allOf").unwrap();
    let all_of = all_of.as_array().unwrap();
    assert_eq!(all_of.len(), 1);
    let all_of_0 = all_of[0].as_object().unwrap();
    assert!(all_of_0.contains_key("$ref"));
    assert_eq!(
        all_of_0.get("$ref").unwrap().as_str().unwrap(),
        "#/definitions/TemperatureUnit"
    );
    let location = properties.get("location").unwrap();
    let location = location.as_object().unwrap();
    assert_eq!(location.len(), 2);
    assert!(location.contains_key("description"));
    assert_eq!(
        location.get("description").unwrap().as_str().unwrap(),
        "the city to get the weather for, e.g., 'Beijing', 'New York', 'Tokyo'"
    );
    assert!(location.contains_key("type"));
    assert_eq!(location.get("type").unwrap().as_str().unwrap(), "string");
    assert!(params.contains_key("required"));
    let required = params.get("required").unwrap();
    let required = required.as_array().unwrap();
    assert_eq!(required.len(), 2);
    assert_eq!(required[0].as_str().unwrap(), "location");
    assert_eq!(required[1].as_str().unwrap(), "unit");
}

/// Function the model may generate JSON inputs for.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolFunction {
    /// The name of the function to be called. Must be a-z, A-Z, 0-9, or contain underscores and dashes, with a maximum length of 64.
    pub name: String,
    /// A description of what the function does, used by the model to choose when and how to call the function.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// The parameters the functions accepts, described as a JSON Schema object.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<JsonObject>,
}

pub type JsonObject<F = Value> = serde_json::Map<String, F>;

#[test]
fn test_chat_serialize_tool_function() {
    let parameters_str = r###"{
        "$schema": "http://json-schema.org/draft-07/schema#",
        "definitions": {
          "TemperatureUnit": {
            "enum": [
              "celsius",
              "fahrenheit"
            ],
            "type": "string"
          }
        },
        "properties": {
          "api_key": {
            "description": "the OpenWeatherMap API key to use. If not provided, the server will use the OPENWEATHERMAP_API_KEY environment variable.",
            "type": [
              "string",
              "null"
            ]
          },
          "location": {
            "description": "the city to get the weather for, e.g., 'Beijing', 'New York', 'Tokyo'",
            "type": "string"
          },
          "unit": {
            "allOf": [
              {
                "$ref": "#/definitions/TemperatureUnit"
              }
            ],
            "description": "the unit to use for the temperature, e.g., 'celsius', 'fahrenheit'"
          }
        },
        "required": [
          "location",
          "unit"
        ],
        "title": "GetWeatherRequest",
        "type": "object"
      }"###;
    let parameters: JsonObject = serde_json::from_str(parameters_str).unwrap();

    let func = ToolFunction {
        name: "get_current_weather".to_string(),
        description: Some("Get the current weather in a given location".to_string()),
        parameters: Some(parameters),
    };

    let json_str = serde_json::to_string(&func).unwrap();
    assert_eq!(
        json_str,
        r###"{"name":"get_current_weather","description":"Get the current weather in a given location","parameters":{"$schema":"http://json-schema.org/draft-07/schema#","definitions":{"TemperatureUnit":{"enum":["celsius","fahrenheit"],"type":"string"}},"properties":{"api_key":{"description":"the OpenWeatherMap API key to use. If not provided, the server will use the OPENWEATHERMAP_API_KEY environment variable.","type":["string","null"]},"location":{"description":"the city to get the weather for, e.g., 'Beijing', 'New York', 'Tokyo'","type":"string"},"unit":{"allOf":[{"$ref":"#/definitions/TemperatureUnit"}],"description":"the unit to use for the temperature, e.g., 'celsius', 'fahrenheit'"}},"required":["location","unit"],"title":"GetWeatherRequest","type":"object"}}"###
    );
}

/// The parameters the functions accepts, described as a JSON Schema object.
///
/// See the [guide](https://platform.openai.com/docs/guides/gpt/function-calling) for examples, and the [JSON Schema reference](https://json-schema.org/understanding-json-schema/) for documentation about the format.
///
/// To describe a function that accepts no parameters, provide the value
/// `{"type": "object", "properties": {}}`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolFunctionParameters {
    #[serde(rename = "type")]
    pub schema_type: JSONSchemaType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<IndexMap<String, Box<JSONSchemaDefine>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required: Option<Vec<String>>,
}

#[test]
fn test_chat_serialize_tool_function_params() {
    {
        let params = ToolFunctionParameters {
            schema_type: JSONSchemaType::Object,
            properties: Some(
                vec![
                    (
                        "location".to_string(),
                        Box::new(JSONSchemaDefine {
                            schema_type: Some(JSONSchemaType::String),
                            description: Some(
                                "The city and state, e.g. San Francisco, CA".to_string(),
                            ),
                            enum_values: None,
                            properties: None,
                            required: None,
                            items: None,
                            default: None,
                            maximum: None,
                            minimum: None,
                            title: None,
                            examples: None,
                        }),
                    ),
                    (
                        "unit".to_string(),
                        Box::new(JSONSchemaDefine {
                            schema_type: Some(JSONSchemaType::String),
                            description: None,
                            enum_values: Some(vec![
                                "celsius".to_string(),
                                "fahrenheit".to_string(),
                            ]),
                            properties: None,
                            required: None,
                            items: None,
                            default: None,
                            maximum: None,
                            minimum: None,
                            title: None,
                            examples: None,
                        }),
                    ),
                ]
                .into_iter()
                .collect(),
            ),
            required: Some(vec!["location".to_string()]),
        };

        let json = serde_json::to_string(&params).unwrap();
        assert_eq!(
            json,
            r#"{"type":"object","properties":{"location":{"type":"string","description":"The city and state, e.g. San Francisco, CA"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["location"]}"#
        );
    }
}

#[test]
fn test_chat_deserialize_tool_function_params() {
    {
        let json = r###"
    {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "The city and state, e.g. San Francisco, CA"
          },
          "unit": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"]
          }
        },
        "required": ["location"]
    }"###;
        let params: ToolFunctionParameters = serde_json::from_str(json).unwrap();
        assert_eq!(params.schema_type, JSONSchemaType::Object);
        let properties = params.properties.as_ref().unwrap();
        assert_eq!(properties.len(), 2);
        assert!(properties.contains_key("unit"));
        assert!(properties.contains_key("location"));
        let unit = properties.get("unit").unwrap();
        assert_eq!(unit.schema_type, Some(JSONSchemaType::String));
        assert_eq!(
            unit.enum_values,
            Some(vec!["celsius".to_string(), "fahrenheit".to_string()])
        );
        let location = properties.get("location").unwrap();
        assert_eq!(location.schema_type, Some(JSONSchemaType::String));
        assert_eq!(
            location.description,
            Some("The city and state, e.g. San Francisco, CA".to_string())
        );
        let required = params.required.as_ref().unwrap();
        assert_eq!(required.len(), 1);
        assert_eq!(required[0], "location");
    }

    {
        let json = r###"{
            "properties": {
                "include_spam_trash": {
                    "default": false,
                    "description": "Include messages from SPAM and TRASH in the results.",
                    "title": "Include Spam Trash",
                    "type": "boolean"
                },
                "add_label_ids": {
                    "default": [],
                    "description": "A list of IDs of labels to add to this thread.",
                    "items": {
                        "type": "string"
                    },
                    "title": "Add Label Ids",
                    "type": "array"
                },
                "max_results": {
                    "default": 10,
                    "description": "Maximum number of messages to return.",
                    "examples": [
                        10,
                        50,
                        100
                    ],
                    "maximum": 500,
                    "minimum": 1,
                    "title": "Max Results",
                    "type": "integer"
                },
                "query": {
                    "default": null,
                    "description": "Only return threads matching the specified query.",
                    "examples": [
                        "is:unread",
                        "from:john.doe@example.com"
                    ],
                    "title": "Query",
                    "type": "string"
                }
            },
            "title": "FetchEmailsRequest",
            "type": "object"
        }"###;

        let params: ToolFunctionParameters = serde_json::from_str(json).unwrap();
        assert_eq!(params.schema_type, JSONSchemaType::Object);
        let properties = params.properties.as_ref().unwrap();
        assert_eq!(properties.len(), 4);
        // println!("{:?}", properties);
        assert!(properties.contains_key("include_spam_trash"));
        assert!(properties.contains_key("add_label_ids"));
        assert!(properties.contains_key("max_results"));
        assert!(properties.contains_key("query"));

        let include_spam_trash = properties.get("include_spam_trash").unwrap();
        assert_eq!(
            include_spam_trash.schema_type,
            Some(JSONSchemaType::Boolean)
        );
        assert_eq!(
            include_spam_trash.description,
            Some("Include messages from SPAM and TRASH in the results.".to_string())
        );
        assert_eq!(
            include_spam_trash.title,
            Some("Include Spam Trash".to_string())
        );
        assert_eq!(
            include_spam_trash.default,
            Some(serde_json::Value::Bool(false))
        );

        let add_label_ids = properties.get("add_label_ids").unwrap();
        assert_eq!(add_label_ids.schema_type, Some(JSONSchemaType::Array));
        assert_eq!(
            add_label_ids.description,
            Some("A list of IDs of labels to add to this thread.".to_string())
        );
        assert_eq!(add_label_ids.title, Some("Add Label Ids".to_string()));
        assert_eq!(
            add_label_ids.default,
            Some(serde_json::Value::Array(vec![]))
        );
        let items = add_label_ids.items.as_ref().unwrap();
        assert_eq!(items.schema_type, Some(JSONSchemaType::String));

        let max_results = properties.get("max_results").unwrap();
        assert_eq!(max_results.schema_type, Some(JSONSchemaType::Integer));
        assert_eq!(
            max_results.description,
            Some("Maximum number of messages to return.".to_string())
        );
        assert_eq!(
            max_results.examples,
            Some(vec![
                Value::Number(serde_json::Number::from(10)),
                Value::Number(serde_json::Number::from(50)),
                Value::Number(serde_json::Number::from(100))
            ])
        );
        assert_eq!(
            max_results.maximum,
            Some(Value::Number(serde_json::Number::from(500)))
        );
        assert_eq!(
            max_results.minimum,
            Some(Value::Number(serde_json::Number::from(1)))
        );
        assert_eq!(max_results.title, Some("Max Results".to_string()));
        assert_eq!(
            max_results.default,
            Some(serde_json::Value::Number(10.into()))
        );

        let query = properties.get("query").unwrap();
        assert_eq!(query.schema_type, Some(JSONSchemaType::String));
        assert_eq!(
            query.description,
            Some("Only return threads matching the specified query.".to_string())
        );
        assert_eq!(
            query.examples,
            Some(vec![
                Value::String("is:unread".to_string()),
                Value::String("from:john.doe@example.com".to_string())
            ])
        );
        assert_eq!(query.title, Some("Query".to_string()));
        assert_eq!(query.default, None);
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct McpTool {
    /// The type of the tool.
    #[serde(rename = "type")]
    pub ty: ToolType,
    /// The label of the server..
    #[serde(rename = "server_label")]
    pub server_label: String,
    /// The URL of the server.
    #[serde(rename = "server_url")]
    pub server_url: String,
    /// The transport type to use for the server.
    #[serde(rename = "transport")]
    pub transport: McpTransport,
    /// The tools allowed to be called by the model.
    #[serde(rename = "allowed_tools", skip_serializing_if = "Option::is_none")]
    pub allowed_tools: Option<Vec<String>>,
    /// The headers to send to the server.
    #[serde(rename = "headers", skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
}
impl McpTool {
    pub fn new(server_label: String, server_url: String, transport: McpTransport) -> Self {
        Self {
            ty: ToolType::Mcp,
            server_label,
            server_url,
            transport,
            allowed_tools: None,
            headers: None,
        }
    }
}

#[test]
fn test_chat_serialize_mcp_tool() {
    let tool = McpTool::new(
        "test".to_string(),
        "https://test.com".to_string(),
        McpTransport::Sse,
    );
    let json = serde_json::to_string(&tool).unwrap();
    assert_eq!(
        json,
        r#"{"type":"mcp","server_label":"test","server_url":"https://test.com","transport":"sse"}"#
    );

    let tool = McpTool {
        ty: ToolType::Mcp,
        server_label: "test".to_string(),
        server_url: "https://test.com".to_string(),
        transport: McpTransport::Sse,
        allowed_tools: Some(vec!["test".to_string()]),
        headers: Some(HashMap::new()),
    };
    let json = serde_json::to_string(&tool).unwrap();
    assert_eq!(
        json,
        r#"{"type":"mcp","server_label":"test","server_url":"https://test.com","transport":"sse","allowed_tools":["test"],"headers":{}}"#
    );

    let tool = McpTool {
        ty: ToolType::Mcp,
        server_label: "test".to_string(),
        server_url: "https://test.com".to_string(),
        transport: McpTransport::StreamHttp,
        allowed_tools: Some(vec!["test".to_string()]),
        headers: Some(HashMap::from([(
            "Authorization".to_string(),
            "Bearer token".to_string(),
        )])),
    };
    let json = serde_json::to_string(&tool).unwrap();
    assert_eq!(
        json,
        r#"{"type":"mcp","server_label":"test","server_url":"https://test.com","transport":"stream-http","allowed_tools":["test"],"headers":{"Authorization":"Bearer token"}}"#
    );
}

#[test]
fn test_chat_deserialize_mcp_tool() {
    let json =
        r#"{"type":"mcp","server_label":"test","server_url":"https://test.com","transport":"sse"}"#;
    let tool: McpTool = serde_json::from_str(json).unwrap();
    assert_eq!(tool.ty, ToolType::Mcp);
    assert_eq!(tool.server_label, "test");
    assert_eq!(tool.server_url, "https://test.com");
    assert_eq!(tool.transport, McpTransport::Sse);
    assert_eq!(tool.allowed_tools, None);
    assert_eq!(tool.headers, None);

    let json = r#"{"type":"mcp","server_label":"test","server_url":"https://test.com","transport":"stream-http","allowed_tools":["test"],"headers":{"Authorization":"Bearer token"}}"#;
    let tool: McpTool = serde_json::from_str(json).unwrap();
    assert_eq!(tool.ty, ToolType::Mcp);
    assert_eq!(tool.server_label, "test");
    assert_eq!(tool.server_url, "https://test.com");
    assert_eq!(tool.transport, McpTransport::StreamHttp);
    assert_eq!(tool.allowed_tools, Some(vec!["test".to_string()]));
    assert_eq!(
        tool.headers,
        Some(HashMap::from([(
            "Authorization".to_string(),
            "Bearer token".to_string()
        )]))
    );
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub enum ToolType {
    #[serde(rename = "function")]
    Function,
    #[serde(rename = "mcp")]
    Mcp,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
pub enum McpTransport {
    #[serde(rename = "sse")]
    Sse,
    #[serde(rename = "stream-http")]
    StreamHttp,
    #[serde(rename = "stdio")]
    Stdio,
}
impl fmt::Display for McpTransport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            McpTransport::Sse => write!(f, "sse"),
            McpTransport::StreamHttp => write!(f, "stream-http"),
            McpTransport::Stdio => write!(f, "stdio"),
        }
    }
}

/// Message for comprising the conversation.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum ChatCompletionRequestMessage {
    System(ChatCompletionSystemMessage),
    User(ChatCompletionUserMessage),
    Assistant(ChatCompletionAssistantMessage),
    Tool(ChatCompletionToolMessage),
}
impl ChatCompletionRequestMessage {
    /// Creates a new system message.
    ///
    /// # Arguments
    ///
    /// * `content` - The contents of the system message.
    ///
    /// * `name` - An optional name for the participant. Provides the model information to differentiate between participants of the same role.
    pub fn new_system_message(content: impl Into<String>, name: Option<String>) -> Self {
        ChatCompletionRequestMessage::System(ChatCompletionSystemMessage::new(content, name))
    }

    /// Creates a new user message.
    ///
    /// # Arguments
    ///
    /// * `content` - The contents of the user message.
    ///
    /// * `name` - An optional name for the participant. Provides the model information to differentiate between participants of the same role.
    pub fn new_user_message(
        content: ChatCompletionUserMessageContent,
        name: Option<String>,
    ) -> Self {
        ChatCompletionRequestMessage::User(ChatCompletionUserMessage::new(content, name))
    }

    /// Creates a new assistant message.
    ///
    /// # Arguments
    ///
    /// * `content` - The contents of the assistant message. Required unless `tool_calls` is specified.
    ///
    /// * `name` - An optional name for the participant. Provides the model information to differentiate between participants of the same role.
    ///
    /// * `tool_calls` - The tool calls generated by the model.
    pub fn new_assistant_message(
        content: Option<String>,
        name: Option<String>,
        tool_calls: Option<Vec<ToolCall>>,
    ) -> Self {
        ChatCompletionRequestMessage::Assistant(ChatCompletionAssistantMessage::new(
            content, name, tool_calls,
        ))
    }

    /// Creates a new tool message.
    pub fn new_tool_message(content: impl Into<String>, tool_call_id: Option<String>) -> Self {
        ChatCompletionRequestMessage::Tool(ChatCompletionToolMessage::new(content, tool_call_id))
    }

    /// The role of the messages author.
    pub fn role(&self) -> ChatCompletionRole {
        match self {
            ChatCompletionRequestMessage::System(_) => ChatCompletionRole::System,
            ChatCompletionRequestMessage::User(_) => ChatCompletionRole::User,
            ChatCompletionRequestMessage::Assistant(_) => ChatCompletionRole::Assistant,
            ChatCompletionRequestMessage::Tool(_) => ChatCompletionRole::Tool,
        }
    }

    /// The name of the participant. Provides the model information to differentiate between participants of the same role.
    pub fn name(&self) -> Option<&String> {
        match self {
            ChatCompletionRequestMessage::System(message) => message.name(),
            ChatCompletionRequestMessage::User(message) => message.name(),
            ChatCompletionRequestMessage::Assistant(message) => message.name(),
            ChatCompletionRequestMessage::Tool(_) => None,
        }
    }
}

#[test]
fn test_chat_serialize_request_message() {
    let message = ChatCompletionRequestMessage::System(ChatCompletionSystemMessage::new(
        "Hello, world!",
        None,
    ));
    let json = serde_json::to_string(&message).unwrap();
    assert_eq!(json, r#"{"role":"system","content":"Hello, world!"}"#);

    let message = ChatCompletionRequestMessage::User(ChatCompletionUserMessage::new(
        ChatCompletionUserMessageContent::Text("Hello, world!".to_string()),
        None,
    ));
    let json = serde_json::to_string(&message).unwrap();
    assert_eq!(json, r#"{"role":"user","content":"Hello, world!"}"#);

    let message = ChatCompletionRequestMessage::Assistant(ChatCompletionAssistantMessage::new(
        Some("Hello, world!".to_string()),
        None,
        None,
    ));
    let json = serde_json::to_string(&message).unwrap();
    assert_eq!(json, r#"{"role":"assistant","content":"Hello, world!"}"#);

    let message = ChatCompletionRequestMessage::Tool(ChatCompletionToolMessage::new(
        "Hello, world!",
        Some("tool-call-id".into()),
    ));
    let json = serde_json::to_string(&message).unwrap();
    assert_eq!(
        json,
        r#"{"role":"tool","content":"Hello, world!","tool_call_id":"tool-call-id"}"#
    );
}

#[test]
fn test_chat_deserialize_request_message() {
    let json = r#"{"content":"Hello, world!","role":"assistant"}"#;
    let message: ChatCompletionRequestMessage = serde_json::from_str(json).unwrap();
    assert_eq!(message.role(), ChatCompletionRole::Assistant);

    let json = r#"{"content":"Hello, world!","role":"system"}"#;
    let message: ChatCompletionRequestMessage = serde_json::from_str(json).unwrap();
    assert_eq!(message.role(), ChatCompletionRole::System);

    let json = r#"{"content":"Hello, world!","role":"user"}"#;
    let message: ChatCompletionRequestMessage = serde_json::from_str(json).unwrap();
    assert_eq!(message.role(), ChatCompletionRole::User);

    let json = r#"{"role":"tool","content":"Hello, world!","tool_call_id":"tool-call-id"}"#;
    let message: ChatCompletionRequestMessage = serde_json::from_str(json).unwrap();
    assert_eq!(message.role(), ChatCompletionRole::Tool);
}

/// Defines the content of a system message.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct ChatCompletionSystemMessage {
    /// The contents of the system message.
    content: String,
    /// An optional name for the participant. Provides the model information to differentiate between participants of the same role.
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
}
impl ChatCompletionSystemMessage {
    /// Creates a new system message.
    ///
    /// # Arguments
    ///
    /// * `content` - The contents of the system message.
    ///
    /// * `name` - An optional name for the participant. Provides the model information to differentiate between participants of the same role.
    pub fn new(content: impl Into<String>, name: Option<String>) -> Self {
        Self {
            content: content.into(),
            name,
        }
    }

    pub fn role(&self) -> ChatCompletionRole {
        ChatCompletionRole::System
    }

    pub fn content(&self) -> &str {
        &self.content
    }

    pub fn name(&self) -> Option<&String> {
        self.name.as_ref()
    }
}

/// Defines the content of a user message.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct ChatCompletionUserMessage {
    /// The contents of the user message.
    content: ChatCompletionUserMessageContent,
    /// An optional name for the participant. Provides the model information to differentiate between participants of the same role.
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
}
impl ChatCompletionUserMessage {
    /// Creates a new user message.
    ///
    /// # Arguments
    ///
    /// * `content` - The contents of the user message.
    ///
    /// * `name` - An optional name for the participant. Provides the model information to differentiate between participants of the same role.
    pub fn new(content: ChatCompletionUserMessageContent, name: Option<String>) -> Self {
        Self { content, name }
    }

    pub fn role(&self) -> ChatCompletionRole {
        ChatCompletionRole::User
    }

    pub fn content(&self) -> &ChatCompletionUserMessageContent {
        &self.content
    }

    pub fn name(&self) -> Option<&String> {
        self.name.as_ref()
    }
}

#[test]
fn test_chat_serialize_user_message() {
    let message = ChatCompletionUserMessage::new(
        ChatCompletionUserMessageContent::Text("Hello, world!".to_string()),
        None,
    );
    let json = serde_json::to_string(&message).unwrap();
    assert_eq!(json, r#"{"content":"Hello, world!"}"#);

    let message = ChatCompletionUserMessage::new(
        ChatCompletionUserMessageContent::Parts(vec![
            ContentPart::Text(TextContentPart::new("Hello, world!")),
            ContentPart::Image(ImageContentPart::new(Image {
                url: "https://example.com/image.png".to_string(),
                detail: Some("auto".to_string()),
            })),
        ]),
        None,
    );
    let json = serde_json::to_string(&message).unwrap();
    assert_eq!(
        json,
        r#"{"content":[{"type":"text","text":"Hello, world!"},{"type":"image_url","image_url":{"url":"https://example.com/image.png","detail":"auto"}}]}"#
    );
}

#[test]
fn test_chat_deserialize_user_message() {
    let json = r#"{"content":"Hello, world!","role":"user"}"#;
    let message: ChatCompletionUserMessage = serde_json::from_str(json).unwrap();
    assert_eq!(message.content().ty(), "text");

    let json = r#"{"content":[{"type":"text","text":"Hello, world!"},{"type":"image_url","image_url":{"url":"https://example.com/image.png","detail":"auto"}}],"role":"user"}"#;
    let message: ChatCompletionUserMessage = serde_json::from_str(json).unwrap();
    assert_eq!(message.content().ty(), "parts");
}

/// Defines the content of an assistant message.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct ChatCompletionAssistantMessage {
    /// The contents of the assistant message. Required unless `tool_calls` is specified.
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    /// An optional name for the participant. Provides the model information to differentiate between participants of the same role.
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    /// The tool calls generated by the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCall>>,
}
impl ChatCompletionAssistantMessage {
    /// Creates a new assistant message.
    ///
    /// # Arguments
    ///
    /// * `content` - The contents of the assistant message. Required unless `tool_calls` is specified.
    ///
    /// * `name` - An optional name for the participant. Provides the model information to differentiate between participants of the same role.
    ///
    /// * `tool_calls` - The tool calls generated by the model.
    pub fn new(
        content: Option<String>,
        name: Option<String>,
        tool_calls: Option<Vec<ToolCall>>,
    ) -> Self {
        match tool_calls.is_some() {
            true => Self {
                content,
                name,
                tool_calls,
            },
            false => Self {
                content,
                name,
                tool_calls: None,
            },
        }
    }

    /// The role of the messages author, in this case `assistant`.
    pub fn role(&self) -> ChatCompletionRole {
        ChatCompletionRole::Assistant
    }

    /// The contents of the assistant message. If `tool_calls` is specified, then `content` is None.
    pub fn content(&self) -> Option<&String> {
        self.content.as_ref()
    }

    /// An optional name for the participant.
    pub fn name(&self) -> Option<&String> {
        self.name.as_ref()
    }

    /// The tool calls generated by the model.
    pub fn tool_calls(&self) -> Option<&Vec<ToolCall>> {
        self.tool_calls.as_ref()
    }
}

#[test]
fn test_chat_serialize_assistant_message() {
    let message =
        ChatCompletionAssistantMessage::new(Some("Hello, world!".to_string()), None, None);
    let json = serde_json::to_string(&message).unwrap();
    assert_eq!(json, r#"{"content":"Hello, world!"}"#);
}

#[test]
fn test_chat_deserialize_assistant_message() {
    let json = r#"{"content":"Hello, world!","role":"assistant"}"#;
    let message: ChatCompletionAssistantMessage = serde_json::from_str(json).unwrap();
    assert_eq!(message.role(), ChatCompletionRole::Assistant);
    assert_eq!(message.content().unwrap().as_str(), "Hello, world!");
}

/// Defines the content of a tool message.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct ChatCompletionToolMessage {
    /// The contents of the tool message.
    content: String,
    /// Tool call that this message is responding to.
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}
impl ChatCompletionToolMessage {
    /// Creates a new tool message.
    ///
    /// # Arguments
    ///
    /// * `content` - The contents of the tool message.
    ///
    /// * `tool_call_id` - Tool call that this message is responding to.
    pub fn new(content: impl Into<String>, tool_call_id: Option<String>) -> Self {
        Self {
            content: content.into(),
            tool_call_id,
        }
    }

    /// The role of the messages author, in this case `tool`.
    pub fn role(&self) -> ChatCompletionRole {
        ChatCompletionRole::Tool
    }

    /// The contents of the tool message.
    pub fn content(&self) -> &str {
        &self.content
    }

    /// Tool call that this message is responding to.
    pub fn tool_call_id(&self) -> Option<String> {
        self.tool_call_id.clone()
    }
}

/// Represents a tool call generated by the model.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct ToolCall {
    /// The ID of the tool call.
    pub id: String,
    /// The type of the tool. Currently, only function is supported.
    #[serde(rename = "type")]
    pub ty: String,
    /// The function that the model called.
    pub function: Function,
}
impl From<ToolCallForChunk> for ToolCall {
    fn from(value: ToolCallForChunk) -> Self {
        Self {
            id: value.id,
            ty: value.ty,
            function: Function {
                name: value.function.name,
                arguments: value.function.arguments,
            },
        }
    }
}

#[test]
fn test_deserialize_tool_call() {
    let json = r#"{"id":"tool-call-id","type":"function","function":{"name":"my_function","arguments":"{\"location\":\"San Francisco, CA\"}"}}"#;
    let tool_call: ToolCall = serde_json::from_str(json).unwrap();
    assert_eq!(tool_call.id, "tool-call-id");
    assert_eq!(tool_call.ty, "function");
    assert_eq!(
        tool_call.function,
        Function {
            name: "my_function".to_string(),
            arguments: r#"{"location":"San Francisco, CA"}"#.to_string()
        }
    );
}

/// Represents a tool call generated by the model.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct ToolCallForChunk {
    pub index: usize,
    /// The ID of the tool call.
    pub id: String,
    /// The type of the tool. Currently, only function is supported.
    #[serde(rename = "type")]
    pub ty: String,
    /// The function that the model called.
    pub function: Function,
}

#[test]
fn test_deserialize_tool_call_for_chunk() {
    let json = r#"{"index":0, "id":"tool-call-id","type":"function","function":{"name":"my_function","arguments":"{\"location\":\"San Francisco, CA\"}"}}"#;
    let tool_call: ToolCallForChunk = serde_json::from_str(json).unwrap();
    assert_eq!(tool_call.index, 0);
    assert_eq!(tool_call.id, "tool-call-id");
    assert_eq!(tool_call.ty, "function");
    assert_eq!(
        tool_call.function,
        Function {
            name: "my_function".to_string(),
            arguments: r#"{"location":"San Francisco, CA"}"#.to_string()
        }
    );
}

/// The function that the model called.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct Function {
    /// The name of the function that the model called.
    pub name: String,
    /// The arguments that the model called the function with.
    pub arguments: String,
}

/// Defines the types of a user message content.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum ChatCompletionUserMessageContent {
    /// The text contents of the message.
    Text(String),
    /// An array of content parts with a defined type, each can be of type `text` or `image_url` when passing in images.
    /// It is required that there must be one content part of type `text` at least. Multiple images are allowed by adding multiple image_url content parts.
    Parts(Vec<ContentPart>),
}
impl ChatCompletionUserMessageContent {
    pub fn ty(&self) -> &str {
        match self {
            ChatCompletionUserMessageContent::Text(_) => "text",
            ChatCompletionUserMessageContent::Parts(_) => "parts",
        }
    }
}

#[test]
fn test_chat_serialize_user_message_content() {
    let content = ChatCompletionUserMessageContent::Text("Hello, world!".to_string());
    let json = serde_json::to_string(&content).unwrap();
    assert_eq!(json, r#""Hello, world!""#);

    let content = ChatCompletionUserMessageContent::Parts(vec![
        ContentPart::Text(TextContentPart::new("Hello, world!")),
        ContentPart::Image(ImageContentPart::new(Image {
            url: "https://example.com/image.png".to_string(),
            detail: Some("auto".to_string()),
        })),
    ]);
    let json = serde_json::to_string(&content).unwrap();
    assert_eq!(
        json,
        r#"[{"type":"text","text":"Hello, world!"},{"type":"image_url","image_url":{"url":"https://example.com/image.png","detail":"auto"}}]"#
    );
}

#[test]
fn test_chat_deserialize_user_message_content() {
    let json = r#"[{"type":"text","text":"Hello, world!"},{"type":"image_url","image_url":{"url":"https://example.com/image.png","detail":"auto"}}]"#;
    let content: ChatCompletionUserMessageContent = serde_json::from_str(json).unwrap();
    assert_eq!(content.ty(), "parts");
    if let ChatCompletionUserMessageContent::Parts(parts) = content {
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].ty(), "text");
        assert_eq!(parts[1].ty(), "image_url");
    }
}

/// Define the content part of a user message.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "lowercase")]
// #[serde(untagged)]
pub enum ContentPart {
    #[serde(rename = "text")]
    Text(TextContentPart),
    #[serde(rename = "image_url")]
    Image(ImageContentPart),
    #[serde(rename = "input_audio")]
    Audio(AudioContentPart),
}
impl ContentPart {
    pub fn ty(&self) -> &str {
        match self {
            ContentPart::Text(_) => "text",
            ContentPart::Image(_) => "image_url",
            ContentPart::Audio(_) => "input_audio",
        }
    }
}

#[test]
fn test_chat_serialize_content_part() {
    let text_content_part = TextContentPart::new("Hello, world!");
    let content_part = ContentPart::Text(text_content_part);
    let json = serde_json::to_string(&content_part).unwrap();
    assert_eq!(json, r#"{"type":"text","text":"Hello, world!"}"#);

    let image_content_part = ImageContentPart::new(Image {
        url: "https://example.com/image.png".to_string(),
        detail: Some("auto".to_string()),
    });
    let content_part = ContentPart::Image(image_content_part);
    let json = serde_json::to_string(&content_part).unwrap();
    assert_eq!(
        json,
        r#"{"type":"image_url","image_url":{"url":"https://example.com/image.png","detail":"auto"}}"#
    );

    let audio_content_part = AudioContentPart::new(Audio {
        data: "dummy-base64-encodings".to_string(),
        format: AudioFormat::Wav,
    });
    let content_part = ContentPart::Audio(audio_content_part);
    let json = serde_json::to_string(&content_part).unwrap();
    assert_eq!(
        json,
        r#"{"type":"input_audio","input_audio":{"data":"dummy-base64-encodings","format":"wav"}}"#
    );
}

#[test]
fn test_chat_deserialize_content_part() {
    let json = r#"{"type":"text","text":"Hello, world!"}"#;
    let content_part: ContentPart = serde_json::from_str(json).unwrap();
    assert_eq!(content_part.ty(), "text");

    let json = r#"{"type":"image_url","image_url":{"url":"https://example.com/image.png","detail":"auto"}}"#;
    let content_part: ContentPart = serde_json::from_str(json).unwrap();
    assert_eq!(content_part.ty(), "image_url");

    let json =
        r#"{"type":"input_audio","input_audio":{"data":"dummy-base64-encodings","format":"wav"}}"#;
    let content_part: ContentPart = serde_json::from_str(json).unwrap();
    assert_eq!(content_part.ty(), "input_audio");
}

/// Represents the text part of a user message content.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct TextContentPart {
    /// The text content.
    text: String,
}
impl TextContentPart {
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }

    /// The text content.
    pub fn text(&self) -> &str {
        &self.text
    }
}

#[test]
fn test_chat_serialize_text_content_part() {
    let text_content_part = TextContentPart::new("Hello, world!");
    let json = serde_json::to_string(&text_content_part).unwrap();
    assert_eq!(json, r#"{"text":"Hello, world!"}"#);
}

#[test]
fn test_chat_deserialize_text_content_part() {
    let json = r#"{"type":"text","text":"Hello, world!"}"#;
    let text_content_part: TextContentPart = serde_json::from_str(json).unwrap();
    assert_eq!(text_content_part.text, "Hello, world!");
}

/// Represents the image part of a user message content.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct ImageContentPart {
    #[serde(rename = "image_url")]
    image: Image,
}
impl ImageContentPart {
    pub fn new(image: Image) -> Self {
        Self { image }
    }

    /// The image URL.
    pub fn image(&self) -> &Image {
        &self.image
    }
}

#[test]
fn test_chat_serialize_image_content_part() {
    let image_content_part = ImageContentPart::new(Image {
        url: "https://example.com/image.png".to_string(),
        detail: Some("auto".to_string()),
    });
    let json = serde_json::to_string(&image_content_part).unwrap();
    assert_eq!(
        json,
        r#"{"image_url":{"url":"https://example.com/image.png","detail":"auto"}}"#
    );

    let image_content_part = ImageContentPart::new(Image {
        url: "https://example.com/image.png".to_string(),
        detail: None,
    });
    let json = serde_json::to_string(&image_content_part).unwrap();
    assert_eq!(
        json,
        r#"{"image_url":{"url":"https://example.com/image.png"}}"#
    );

    let image_content_part = ImageContentPart::new(Image {
        url: "base64".to_string(),
        detail: Some("auto".to_string()),
    });
    let json = serde_json::to_string(&image_content_part).unwrap();
    assert_eq!(json, r#"{"image_url":{"url":"base64","detail":"auto"}}"#);

    let image_content_part = ImageContentPart::new(Image {
        url: "base64".to_string(),
        detail: None,
    });
    let json = serde_json::to_string(&image_content_part).unwrap();
    assert_eq!(json, r#"{"image_url":{"url":"base64"}}"#);
}

#[test]
fn test_chat_deserialize_image_content_part() {
    let json = r#"{"type":"image_url","image_url":{"url":"https://example.com/image.png","detail":"auto"}}"#;
    let image_content_part: ImageContentPart = serde_json::from_str(json).unwrap();
    // assert_eq!(image_content_part.ty, "image_url");
    assert_eq!(
        image_content_part.image.url,
        "https://example.com/image.png"
    );
    assert_eq!(image_content_part.image.detail, Some("auto".to_string()));
}

/// JPEG baseline & progressive (12 bpc/arithmetic not supported, same as stock IJG lib)
/// PNG 1/2/4/8/16-bit-per-channel
///
/// TGA (not sure what subset, if a subset)
/// BMP non-1bpp, non-RLE
/// PSD (composited view only, no extra channels, 8/16 bit-per-channel)
///
/// GIF (*comp always reports as 4-channel)
/// HDR (radiance rgbE format)
/// PIC (Softimage PIC)
/// PNM (PPM and PGM binary only)
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
pub struct Image {
    /// Either a URL of the image or the base64 encoded image data.
    pub url: String,
    /// Specifies the detail level of the image. Defaults to auto.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}
impl Image {
    pub fn is_url(&self) -> bool {
        url::Url::parse(&self.url).is_ok()
    }
}

#[test]
fn test_chat_serialize_image() {
    let image = Image {
        url: "https://example.com/image.png".to_string(),
        detail: Some("auto".to_string()),
    };
    let json = serde_json::to_string(&image).unwrap();
    assert_eq!(
        json,
        r#"{"url":"https://example.com/image.png","detail":"auto"}"#
    );

    let image = Image {
        url: "https://example.com/image.png".to_string(),
        detail: None,
    };
    let json = serde_json::to_string(&image).unwrap();
    assert_eq!(json, r#"{"url":"https://example.com/image.png"}"#);

    let image = Image {
        url: "base64".to_string(),
        detail: Some("auto".to_string()),
    };
    let json = serde_json::to_string(&image).unwrap();
    assert_eq!(json, r#"{"url":"base64","detail":"auto"}"#);

    let image = Image {
        url: "base64".to_string(),
        detail: None,
    };
    let json = serde_json::to_string(&image).unwrap();
    assert_eq!(json, r#"{"url":"base64"}"#);
}

#[test]
fn test_chat_deserialize_image() {
    let json = r#"{"url":"https://example.com/image.png","detail":"auto"}"#;
    let image: Image = serde_json::from_str(json).unwrap();
    assert_eq!(image.url, "https://example.com/image.png");
    assert_eq!(image.detail, Some("auto".to_string()));

    let json = r#"{"url":"https://example.com/image.png"}"#;
    let image: Image = serde_json::from_str(json).unwrap();
    assert_eq!(image.url, "https://example.com/image.png");
    assert_eq!(image.detail, None);

    let json = r#"{"url":"base64","detail":"auto"}"#;
    let image: Image = serde_json::from_str(json).unwrap();
    assert_eq!(image.url, "base64");
    assert_eq!(image.detail, Some("auto".to_string()));

    let json = r#"{"url":"base64"}"#;
    let image: Image = serde_json::from_str(json).unwrap();
    assert_eq!(image.url, "base64");
    assert_eq!(image.detail, None);
}

/// Represents the audio part of a user message content.
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
pub struct AudioContentPart {
    #[serde(rename = "input_audio")]
    audio: Audio,
}
impl AudioContentPart {
    pub fn new(audio: Audio) -> Self {
        Self { audio }
    }

    /// The audio data.
    pub fn audio(&self) -> &Audio {
        &self.audio
    }
}

#[test]
fn test_chat_serialize_audio_content_part() {
    let audio_content_part = AudioContentPart {
        audio: Audio {
            data: "dummy-base64-encodings".to_string(),
            format: AudioFormat::Wav,
        },
    };
    let json = serde_json::to_string(&audio_content_part).unwrap();
    assert_eq!(
        json,
        r#"{"input_audio":{"data":"dummy-base64-encodings","format":"wav"}}"#
    );

    let audio_content_part = AudioContentPart {
        audio: Audio {
            data: "dummy-base64-encodings".to_string(),
            format: AudioFormat::Mp3,
        },
    };
    let json = serde_json::to_string(&audio_content_part).unwrap();
    assert_eq!(
        json,
        r#"{"input_audio":{"data":"dummy-base64-encodings","format":"mp3"}}"#
    );
}

#[test]
fn test_chat_deserialize_audio_content_part() {
    let json = r#"{"input_audio":{"data":"dummy-base64-encodings","format":"wav"}}"#;
    let audio_content_part: AudioContentPart = serde_json::from_str(json).unwrap();
    assert_eq!(audio_content_part.audio.data, "dummy-base64-encodings");
    assert_eq!(audio_content_part.audio.format, AudioFormat::Wav);

    let json = r#"{"input_audio":{"data":"dummy-base64-encodings","format":"mp3"}}"#;
    let audio_content_part: AudioContentPart = serde_json::from_str(json).unwrap();
    assert_eq!(audio_content_part.audio.data, "dummy-base64-encodings");
    assert_eq!(audio_content_part.audio.format, AudioFormat::Mp3);
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
pub struct Audio {
    /// Base64 encoded audio data.
    pub data: String,
    /// The format of the encoded audio data.
    pub format: AudioFormat,
}

/// The format of the encoded audio data.
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum AudioFormat {
    Wav,
    Mp3,
}

#[test]
fn test_chat_serialize_audio() {
    let audio = Audio {
        data: "dummy-base64-encodings".to_string(),
        format: AudioFormat::Wav,
    };
    let json = serde_json::to_string(&audio).unwrap();
    assert_eq!(json, r#"{"data":"dummy-base64-encodings","format":"wav"}"#);

    let audio = Audio {
        data: "dummy-base64-encodings".to_string(),
        format: AudioFormat::Mp3,
    };
    let json = serde_json::to_string(&audio).unwrap();
    assert_eq!(json, r#"{"data":"dummy-base64-encodings","format":"mp3"}"#);
}

#[test]
fn test_chat_deserialize_audio() {
    let json = r#"{"data":"dummy-base64-encodings","format":"wav"}"#;
    let audio: Audio = serde_json::from_str(json).unwrap();
    assert_eq!(audio.data, "dummy-base64-encodings");
    assert_eq!(audio.format, AudioFormat::Wav);

    let json = r#"{"data":"dummy-base64-encodings","format":"mp3"}"#;
    let audio: Audio = serde_json::from_str(json).unwrap();
    assert_eq!(audio.data, "dummy-base64-encodings");
    assert_eq!(audio.format, AudioFormat::Mp3);
}

/// Sampling methods used for chat completion requests.
#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq)]
pub enum ChatCompletionRequestSampling {
    /// What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
    Temperature(f64),
    /// An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.
    TopP(f64),
}

/// The role of the messages author.
#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ChatCompletionRole {
    System,
    User,
    Assistant,
    /// **Deprecated since 0.10.0.** Use [ChatCompletionRole::Tool] instead.
    Function,
    Tool,
}
impl std::fmt::Display for ChatCompletionRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChatCompletionRole::System => write!(f, "system"),
            ChatCompletionRole::User => write!(f, "user"),
            ChatCompletionRole::Assistant => write!(f, "assistant"),
            ChatCompletionRole::Function => write!(f, "function"),
            ChatCompletionRole::Tool => write!(f, "tool"),
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ChatCompletionRequestFunction {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    parameters: ChatCompletionRequestFunctionParameters,
}

/// The parameters the functions accepts, described as a JSON Schema object.
///
/// See the [guide](https://platform.openai.com/docs/guides/gpt/function-calling) for examples, and the [JSON Schema reference](https://json-schema.org/understanding-json-schema/) for documentation about the format.
///
/// To describe a function that accepts no parameters, provide the value
/// `{"type": "object", "properties": {}}`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatCompletionRequestFunctionParameters {
    #[serde(rename = "type")]
    pub schema_type: JSONSchemaType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<HashMap<String, Box<JSONSchemaDefine>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum JSONSchemaType {
    Object,
    Number,
    Integer,
    String,
    Array,
    Null,
    Boolean,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JSONSchemaDefine {
    #[serde(rename = "type")]
    pub schema_type: Option<JSONSchemaType>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(rename = "enum", skip_serializing_if = "Option::is_none")]
    pub enum_values: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<HashMap<String, Box<JSONSchemaDefine>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub items: Option<Box<JSONSchemaDefine>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub maximum: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub minimum: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub examples: Option<Vec<Value>>,
}

/// Represents a chat completion response returned by model, based on the provided input.
#[derive(Debug, Deserialize, Serialize)]
pub struct ChatCompletionObject {
    /// A unique identifier for the chat completion.
    pub id: String,
    /// The object type, which is always `chat.completion`.
    pub object: String,
    /// The Unix timestamp (in seconds) of when the chat completion was created.
    pub created: u64,
    /// The model used for the chat completion.
    pub model: String,
    /// A list of chat completion choices. Can be more than one if `n_choice` is greater than 1.
    pub choices: Vec<ChatCompletionObjectChoice>,
    /// Usage statistics for the completion request.
    pub usage: Usage,
}

#[test]
fn test_deserialize_chat_completion_object() {
    let json = r#"{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1699896916,
  "model": "gpt-3.5-turbo-0125",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_abc123",
            "type": "function",
            "function": {
              "name": "get_current_weather",
              "arguments": "{\n\"location\": \"Boston, MA\"\n}"
            }
          }
        ]
      },
      "logprobs": null,
      "finish_reason": "tool_calls"
    }
  ],
  "usage": {
    "prompt_tokens": 82,
    "completion_tokens": 17,
    "total_tokens": 99
  }
}"#;

    let chatcmp_object: ChatCompletionObject = serde_json::from_str(json).unwrap();
    assert_eq!(chatcmp_object.id, "chatcmpl-abc123");
    assert_eq!(chatcmp_object.object, "chat.completion");
    assert_eq!(chatcmp_object.created, 1699896916);
    assert_eq!(chatcmp_object.model, "gpt-3.5-turbo-0125");
    assert_eq!(chatcmp_object.choices.len(), 1);
    assert_eq!(chatcmp_object.choices[0].index, 0);
    assert_eq!(
        chatcmp_object.choices[0].finish_reason,
        FinishReason::tool_calls
    );
    assert_eq!(chatcmp_object.choices[0].message.tool_calls.len(), 1);
    assert_eq!(
        chatcmp_object.choices[0].message.tool_calls[0].id,
        "call_abc123"
    );
    assert_eq!(
        chatcmp_object.choices[0].message.tool_calls[0].ty,
        "function"
    );
    assert_eq!(
        chatcmp_object.choices[0].message.tool_calls[0]
            .function
            .name,
        "get_current_weather"
    );
    assert_eq!(
        chatcmp_object.choices[0].message.tool_calls[0]
            .function
            .arguments,
        "{\n\"location\": \"Boston, MA\"\n}"
    );
    assert_eq!(chatcmp_object.usage.prompt_tokens, 82);
    assert_eq!(chatcmp_object.usage.completion_tokens, 17);
    assert_eq!(chatcmp_object.usage.total_tokens, 99);
}

/// Represents a chat completion choice returned by model.
#[derive(Debug, Deserialize, Serialize)]
pub struct ChatCompletionObjectChoice {
    /// The index of the choice in the list of choices.
    pub index: u32,
    /// A chat completion message generated by the model.
    pub message: ChatCompletionObjectMessage,
    /// The reason the model stopped generating tokens. This will be `stop` if the model hit a natural stop point or a provided stop sequence, `length` if the maximum number of tokens specified in the request was reached, or `function_call` if the model called a function.
    pub finish_reason: FinishReason,
    /// Log probability information for the choice.
    pub logprobs: Option<LogProbs>,
}

#[test]
fn test_serialize_chat_completion_object_choice() {
    let tool = ToolCall {
        id: "call_abc123".to_string(),
        ty: "function".to_string(),
        function: Function {
            name: "get_current_weather".to_string(),
            arguments: "{\"location\": \"Boston, MA\"}".to_string(),
        },
    };
    let message = ChatCompletionObjectMessage {
        content: None,
        tool_calls: vec![tool],
        role: ChatCompletionRole::Assistant,
        function_call: None,
    };
    let choice = ChatCompletionObjectChoice {
        index: 0,
        message,
        finish_reason: FinishReason::tool_calls,
        logprobs: None,
    };
    let json = serde_json::to_string(&choice).unwrap();
    assert_eq!(
        json,
        r#"{"index":0,"message":{"content":null,"tool_calls":[{"id":"call_abc123","type":"function","function":{"name":"get_current_weather","arguments":"{\"location\": \"Boston, MA\"}"}}],"role":"assistant"},"finish_reason":"tool_calls","logprobs":null}"#
    );
}

/// Log probability information for the choice.
#[derive(Debug, Deserialize, Serialize)]
pub struct LogProbs;

/// Represents a chat completion message generated by the model.
#[derive(Debug, Serialize)]
pub struct ChatCompletionObjectMessage {
    /// The contents of the message.
    pub content: Option<String>,
    /// The tool calls generated by the model, such as function calls.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    /// The role of the author of this message.
    pub role: ChatCompletionRole,
    /// Deprecated. The name and arguments of a function that should be called, as generated by the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<ChatMessageFunctionCall>,
}
impl<'de> Deserialize<'de> for ChatCompletionObjectMessage {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct ChatCompletionObjectMessageVisitor;

        impl<'de> Visitor<'de> for ChatCompletionObjectMessageVisitor {
            type Value = ChatCompletionObjectMessage;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct ChatCompletionObjectMessage")
            }

            fn visit_map<V>(self, mut map: V) -> Result<ChatCompletionObjectMessage, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut content = None;
                let mut tool_calls = None;
                let mut role = None;
                let mut function_call = None;

                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "content" => content = map.next_value()?,
                        "tool_calls" => tool_calls = map.next_value()?,
                        "role" => role = map.next_value()?,
                        "function_call" => function_call = map.next_value()?,
                        _ => {
                            // Ignore unknown fields
                            let _ = map.next_value::<IgnoredAny>()?;

                            #[cfg(feature = "logging")]
                            warn!(target: "stdout", "Not supported field: {key}");
                        }
                    }
                }

                let content = content;
                let tool_calls = tool_calls.unwrap_or_default();
                let role = role.ok_or_else(|| de::Error::missing_field("role"))?;
                let function_call = function_call;

                Ok(ChatCompletionObjectMessage {
                    content,
                    tool_calls,
                    role,
                    function_call,
                })
            }
        }

        const FIELDS: &[&str] = &["content", "tool_calls", "role", "function_call"];
        deserializer.deserialize_struct(
            "ChatCompletionObjectMessage",
            FIELDS,
            ChatCompletionObjectMessageVisitor,
        )
    }
}

#[test]
fn test_serialize_chat_completion_object_message() {
    let tool = ToolCall {
        id: "call_abc123".to_string(),
        ty: "function".to_string(),
        function: Function {
            name: "get_current_weather".to_string(),
            arguments: "{\"location\": \"Boston, MA\"}".to_string(),
        },
    };
    let message = ChatCompletionObjectMessage {
        content: None,
        tool_calls: vec![tool],
        role: ChatCompletionRole::Assistant,
        function_call: None,
    };
    let json = serde_json::to_string(&message).unwrap();
    assert_eq!(
        json,
        r#"{"content":null,"tool_calls":[{"id":"call_abc123","type":"function","function":{"name":"get_current_weather","arguments":"{\"location\": \"Boston, MA\"}"}}],"role":"assistant"}"#
    );
}

#[test]
fn test_deserialize_chat_completion_object_message() {
    {
        let json = r#"{"content":null,"tool_calls":[{"id":"call_abc123","type":"function","function":{"name":"get_current_weather","arguments":"{\"location\": \"Boston, MA\"}"}}],"role":"assistant"}"#;
        let message: ChatCompletionObjectMessage = serde_json::from_str(json).unwrap();
        assert_eq!(message.content, None);
        assert_eq!(message.tool_calls.len(), 1);
        assert_eq!(message.role, ChatCompletionRole::Assistant);
    }

    {
        let json = r#"{"content":null,"role":"assistant"}"#;
        let message: ChatCompletionObjectMessage = serde_json::from_str(json).unwrap();
        assert_eq!(message.content, None);
        assert!(message.tool_calls.is_empty());
        assert_eq!(message.role, ChatCompletionRole::Assistant);
    }
}

/// The name and arguments of a function that should be called, as generated by the model.
#[derive(Debug, Deserialize, Serialize)]
pub struct ChatMessageFunctionCall {
    /// The name of the function to call.
    pub name: String,

    /// The arguments to call the function with, as generated by the model in JSON format. Note that the model does not always generate valid JSON, and may hallucinate parameters not defined by your function schema. Validate the arguments in your code before calling your function.
    pub arguments: String,
}

/// Represents a streamed chunk of a chat completion response returned by model, based on the provided input.
#[derive(Debug, Deserialize, Serialize)]
pub struct ChatCompletionChunk {
    /// A unique identifier for the chat completion.
    pub id: String,
    /// A list of chat completion choices. Can be more than one if `n_choice` is greater than 1.
    pub choices: Vec<ChatCompletionChunkChoice>,
    /// The Unix timestamp (in seconds) of when the chat completion was created.
    pub created: u64,
    /// The model used for the chat completion.
    pub model: String,
    /// This fingerprint represents the backend configuration that the model runs with. Can be used in conjunction with the `seed` request parameter to understand when backend changes have been made that might impact determinism.
    pub system_fingerprint: String,
    /// The object type, which is always `chat.completion.chunk`.
    pub object: String,
    /// Usage statistics for the completion request.
    ///
    /// An optional field that will only be present when you set stream_options: {"include_usage": true} in your request. When present, it contains a null value except for the last chunk which contains the token usage statistics for the entire request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[test]
fn test_serialize_chat_completion_chunk() {
    let chunk = ChatCompletionChunk {
        id: "chatcmpl-1d0ff773-e8ab-4254-a222-96e97e3c295a".to_string(),
        choices: vec![ChatCompletionChunkChoice {
            index: 0,
            delta: ChatCompletionChunkChoiceDelta {
                content: Some(".".to_owned()),
                tool_calls: vec![],
                role: ChatCompletionRole::Assistant,
            },
            logprobs: None,
            finish_reason: None,
        }],
        created: 1722433423,
        model: "default".to_string(),
        system_fingerprint: "fp_44709d6fcb".to_string(),
        object: "chat.completion.chunk".to_string(),
        usage: None,
    };

    let json = serde_json::to_string(&chunk).unwrap();
    assert_eq!(
        json,
        r#"{"id":"chatcmpl-1d0ff773-e8ab-4254-a222-96e97e3c295a","choices":[{"index":0,"delta":{"content":".","role":"assistant"},"logprobs":null,"finish_reason":null}],"created":1722433423,"model":"default","system_fingerprint":"fp_44709d6fcb","object":"chat.completion.chunk"}"#
    );
}

#[test]
fn test_deserialize_chat_completion_chunk() {
    {
        let json = r#"{"id":"chatcmpl-1d0ff773-e8ab-4254-a222-96e97e3c295a","choices":[{"index":0,"delta":{"content":".","role":"assistant"},"logprobs":null,"finish_reason":null}],"created":1722433423,"model":"default","system_fingerprint":"fp_44709d6fcb","object":"chat.completion.chunk"}"#;

        let chunk: ChatCompletionChunk = serde_json::from_str(json).unwrap();
        assert_eq!(chunk.id, "chatcmpl-1d0ff773-e8ab-4254-a222-96e97e3c295a");
        assert_eq!(chunk.choices.len(), 1);
        assert_eq!(chunk.choices[0].index, 0);
        assert_eq!(chunk.choices[0].delta.content, Some(".".to_owned()));
        assert!(chunk.choices[0].delta.tool_calls.is_empty());
        assert_eq!(chunk.choices[0].delta.role, ChatCompletionRole::Assistant);
        assert_eq!(chunk.created, 1722433423);
        assert_eq!(chunk.model, "default");
        assert_eq!(chunk.system_fingerprint, "fp_44709d6fcb");
        assert_eq!(chunk.object, "chat.completion.chunk");
    }

    {
        let json_str = r#"{"id":"chatcmpl-5b20a5a9-80e0-4cc4-9d33-7ab504dac9ca","choices":[{"index":0,"delta":{"content":null,"tool_calls":[{"index":0,"id":"call_abc123","type":"function","function":{"name":"get_current_weather","arguments":"{\"location\":\"Beijing\",\"unit\":\"celsius\"}"}}],"role":"assistant"},"logprobs":null,"finish_reason":null}],"created":1744028716,"model":"Llama-3-Groq-8B","system_fingerprint":"fp_44709d6fcb","object":"chat.completion.chunk"}"#;

        let chunk: ChatCompletionChunk = serde_json::from_str(json_str).unwrap();
        assert_eq!(chunk.id, "chatcmpl-5b20a5a9-80e0-4cc4-9d33-7ab504dac9ca");
        assert_eq!(chunk.choices.len(), 1);
        assert_eq!(chunk.choices[0].index, 0);
        assert_eq!(chunk.choices[0].delta.content, None);
        assert_eq!(chunk.choices[0].delta.tool_calls.len(), 1);
    }
}

/// Represents a chat completion choice in a streamed chunk of a chat completion response.
#[derive(Debug, Deserialize, Serialize)]
pub struct ChatCompletionChunkChoice {
    /// The index of the choice in the list of choices.
    pub index: u32,
    /// A chat completion delta generated by streamed model responses.
    pub delta: ChatCompletionChunkChoiceDelta,
    /// Log probability information for the choice.
    pub logprobs: Option<LogProbs>,
    /// The reason the model stopped generating tokens. This will be `stop` if the model hit a natural stop point or a provided stop sequence, `length` if the maximum number of tokens specified in the request was reached, or `function_call` if the model called a function.
    pub finish_reason: Option<FinishReason>,
}

/// Represents a chat completion delta generated by streamed model responses.
#[derive(Debug, Serialize)]
pub struct ChatCompletionChunkChoiceDelta {
    /// The contents of the chunk message.
    pub content: Option<String>,
    /// The name and arguments of a function that should be called, as generated by the model.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCallForChunk>,
    /// The role of the author of this message.
    pub role: ChatCompletionRole,
}
impl<'de> Deserialize<'de> for ChatCompletionChunkChoiceDelta {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct ChatCompletionChunkChoiceDeltaVisitor;

        impl<'de> Visitor<'de> for ChatCompletionChunkChoiceDeltaVisitor {
            type Value = ChatCompletionChunkChoiceDelta;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct ChatCompletionChunkChoiceDelta")
            }

            fn visit_map<V>(self, mut map: V) -> Result<ChatCompletionChunkChoiceDelta, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut content = None;
                let mut tool_calls = None;
                let mut role = None;

                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "content" => content = map.next_value()?,
                        "tool_calls" => tool_calls = map.next_value()?,
                        "role" => role = map.next_value()?,
                        _ => {
                            // Ignore unknown fields
                            let _ = map.next_value::<IgnoredAny>()?;

                            #[cfg(feature = "logging")]
                            warn!(target: "stdout", "Not supported field: {key}");
                        }
                    }
                }

                let content = content;
                let tool_calls = tool_calls.unwrap_or_default();
                let role = role.ok_or_else(|| de::Error::missing_field("role"))?;
                Ok(ChatCompletionChunkChoiceDelta {
                    content,
                    tool_calls,
                    role,
                })
            }
        }

        const FIELDS: &[&str] = &["content", "tool_calls", "role"];
        deserializer.deserialize_struct(
            "ChatCompletionChunkChoiceDelta",
            FIELDS,
            ChatCompletionChunkChoiceDeltaVisitor,
        )
    }
}
