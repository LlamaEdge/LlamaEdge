use crate::common::{FinishReason, Usage};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Request builder for creating a new chat completion request.
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
                model: Some(model.into()),
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
    pub fn with_max_tokens(mut self, max_tokens: u64) -> Self {
        let max_tokens = if max_tokens < 1 { 16 } else { max_tokens };
        self.req.max_tokens = Some(max_tokens);
        self
    }

    /// Sets the presence penalty. Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
    pub fn with_presence_penalty(mut self, penalty: f64) -> Self {
        self.req.presence_penalty = Some(penalty);
        self
    }

    /// Sets the frequency penalty. Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
    pub fn with_frequency_penalty(mut self, penalty: f64) -> Self {
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

/// Create a new chat completion request.
#[derive(Debug, Deserialize, Serialize, Default)]
pub struct ChatCompletionRequest {
    /// The model to use for generating completions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// A list of messages comprising the conversation so far.
    pub messages: Vec<ChatCompletionRequestMessage>,
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

    //* OpenAI specific parameters
    /// A list of functions the model may generate JSON inputs for.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub functions: Option<Vec<ChatCompletionRequestFunction>>,
    /// Controls how the model responds to function calls. "none" means the model does not call a function, and responds to the end-user. "auto" means the model can pick between an end-user or calling a function. Specifying a particular function via `{"name":\ "my_function"}` forces the model to call that function. "none" is the default when no functions are present. "auto" is the default if functions are present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<String>,
}

#[test]
fn test_chat_serialize_chat_request() {
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
        let request = ChatCompletionRequestBuilder::new("model-id", messages)
            .with_sampling(ChatCompletionRequestSampling::Temperature(0.8))
            .with_n_choices(3)
            .with_stream(true)
            .with_stop(vec!["stop1".to_string(), "stop2".to_string()])
            .with_max_tokens(100)
            .with_presence_penalty(0.5)
            .with_frequency_penalty(0.5)
            .build();
        let json = serde_json::to_string(&request).unwrap();
        assert_eq!(
            json,
            r#"{"model":"model-id","messages":[{"role":"system","content":"Hello, world!"},{"role":"user","content":"Hello, world!"},{"role":"assistant","content":"Hello, world!"}],"temperature":0.8,"top_p":1.0,"n_choice":3,"stream":true,"stop":["stop1","stop2"],"max_tokens":100,"presence_penalty":0.5,"frequency_penalty":0.5}"#
        );
    }

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

        let request = ChatCompletionRequestBuilder::new("model-id", messages).build();
        let json = serde_json::to_string(&request).unwrap();
        assert_eq!(
            json,
            r#"{"model":"model-id","messages":[{"role":"system","content":"Hello, world!"},{"role":"user","content":[{"type":"text","text":"what is in the picture?"},{"type":"image_url","image_url":{"url":"https://example.com/image.png"}}]}]}"#
        );
    }
}

/// Message for comprising the conversation.
#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum ChatCompletionRequestMessage {
    System(ChatCompletionSystemMessage),
    User(ChatCompletionUserMessage),
    Assistant(ChatCompletionAssistantMessage),
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

    /// The role of the messages author.
    pub fn role(&self) -> ChatCompletionRole {
        match self {
            ChatCompletionRequestMessage::System(message) => message.role(),
            ChatCompletionRequestMessage::User(message) => message.role(),
            ChatCompletionRequestMessage::Assistant(message) => message.role(),
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
}

#[derive(Debug, Deserialize, Serialize)]
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

#[derive(Debug, Deserialize, Serialize)]
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
    let json = r#"{"content":"Hello, world!"}"#;
    let message: ChatCompletionUserMessage = serde_json::from_str(json).unwrap();
    assert_eq!(message.content().ty(), "text");

    let json = r#"{"content":[{"type":"text","text":"Hello, world!"},{"type":"image_url","image_url":{"url":"https://example.com/image.png","detail":"auto"}}]}"#;
    let message: ChatCompletionUserMessage = serde_json::from_str(json).unwrap();
    assert_eq!(message.content().ty(), "parts");
}

#[derive(Debug, Deserialize, Serialize)]
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
                content: None,
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

#[derive(Debug, Deserialize, Serialize)]
pub struct ToolCall {
    /// The ID of the tool call.
    id: String,
    /// The type of the tool. Currently, only function is supported.
    #[serde(rename = "type")]
    ty: String,
    /// The function that the model called.
    function: Fuction,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Fuction {
    /// The name of the function that the model called.
    name: String,
    /// The arguments that the model called the function with.
    arguments: String,
}

#[derive(Debug, Deserialize, Serialize)]
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

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
// #[serde(untagged)]
pub enum ContentPart {
    #[serde(rename = "text")]
    Text(TextContentPart),
    #[serde(rename = "image_url")]
    Image(ImageContentPart),
}
impl ContentPart {
    pub fn ty(&self) -> &str {
        match self {
            ContentPart::Text(_) => "text",
            ContentPart::Image(_) => "image_url",
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
}

#[test]
fn test_chat_deserialize_content_part() {
    let json = r#"{"type":"text","text":"Hello, world!"}"#;
    let content_part: ContentPart = serde_json::from_str(json).unwrap();
    assert_eq!(content_part.ty(), "text");

    let json = r#"{"type":"image_url","image_url":{"url":"https://example.com/image.png","detail":"auto"}}"#;
    let content_part: ContentPart = serde_json::from_str(json).unwrap();
    assert_eq!(content_part.ty(), "image_url");
}

#[derive(Debug, Deserialize, Serialize)]
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

#[derive(Debug, Deserialize, Serialize)]
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

#[derive(Debug, Deserialize, Serialize)]
pub struct ChatCompletionObjectChoice {
    /// The index of the choice in the list of choices.
    pub index: u32,
    /// A chat completion message generated by the model.
    pub message: ChatCompletionObjectMessage,
    /// The reason the model stopped generating tokens. This will be `stop` if the model hit a natural stop point or a provided stop sequence, `length` if the maximum number of tokens specified in the request was reached, or `function_call` if the model called a function.
    pub finish_reason: FinishReason,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ChatCompletionObjectMessage {
    /// The role of the author of this message.
    pub role: ChatCompletionRole,
    /// The contents of the message.
    pub content: String,
    /// The name and arguments of a function that should be called, as generated by the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<ChatMessageFunctionCall>,
}

/// The name and arguments of a function that should be called, as generated by the model.
#[derive(Debug, Deserialize, Serialize)]
pub struct ChatMessageFunctionCall {
    /// The name of the function to call.
    pub name: String,

    /// The arguments to call the function with, as generated by the model in JSON format. Note that the model does not always generate valid JSON, and may hallucinate parameters not defined by your function schema. Validate the arguments in your code before calling your function.
    pub arguments: String,
}

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
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ChatCompletionChunkChoice {
    /// The index of the choice in the list of choices.
    pub index: u32,
    /// A chat completion delta generated by streamed model responses.
    pub delta: ChatCompletionChunkChoiceDelta,
    /// Log probability information for the choice.
    pub logprobs: Option<ChatCompletionChunkChoiceLogprobs>,
    /// The reason the model stopped generating tokens. This will be `stop` if the model hit a natural stop point or a provided stop sequence, `length` if the maximum number of tokens specified in the request was reached, or `function_call` if the model called a function.
    pub finish_reason: Option<FinishReason>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ChatCompletionChunkChoiceDelta {
    /// The role of the author of this message.
    pub role: Option<ChatCompletionRole>,
    /// The contents of the chunk message.
    pub content: Option<String>,
    /// The name and arguments of a function that should be called, as generated by the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<ChatMessageFunctionCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

// #[derive(Debug, Deserialize, Serialize)]
// pub struct ToolCall {
//     index: u32,
//     id: String,
//     /// The type of the tool. Currently, only function is supported.
//     ty: String,
//     function: ToolCallFunction,
// }

// #[derive(Debug, Deserialize, Serialize)]
// pub struct ToolCallFunction {
//     name: String,
//     arguments: String,
// }

#[derive(Debug, Deserialize, Serialize)]
pub struct ChatCompletionChunkChoiceLogprobs;
