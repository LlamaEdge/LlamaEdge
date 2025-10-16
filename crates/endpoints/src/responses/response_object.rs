use crate::{chat::JsonObject, responses::items::*};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Represents a request body to generate a model response.
#[derive(Debug, Serialize)]
pub struct RequestOfModelResponse {
    /// Whether to run the model response in the background.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub background: Option<bool>,
    /// The conversation that this response belongs to. Items from this conversation are prepended to `input_items` for this response request. Input items and output items from this response are automatically added to this conversation after this response completes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation: Option<Conversation>,
    /// Specify additional output data to include in the model response. Currently supported values are:
    /// - `web_search_call.action.sources`: Include the sources of the web search tool call.
    /// - `code_interpreter_call.outputs`: Includes the outputs of python code execution in code interpreter tool call items.
    /// - `computer_call_output.output.image_url`: Include image urls from the computer call output.
    /// - `file_search_call.results`: Include the search results of the file search tool call.
    /// - `message.input_image.image_url`: Include image urls from the input message.
    /// - `message.output_text.logprobs`: Include logprobs with assistant messages.
    /// - `reasoning.encrypted_content`: Includes an encrypted version of reasoning tokens in reasoning item outputs. This enables reasoning items to be used in multi-turn conversations when using the Responses API statelessly (like when the store parameter is set to false, or when an organization is enrolled in the zero data retention program).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include: Option<Vec<String>>,
    /// Text, image, or file inputs to the model, used to generate a response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<Input>,
    /// A system (or developer) message inserted into the model's context.
    ///
    /// When using along with `previous_response_id`, the instructions from a previous response will not be carried over to the next response. This makes it simple to swap out system (or developer) messages in new responses.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    /// An upper bound for the number of tokens that can be generated for a response, including visible output tokens and reasoning tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<i32>,
    /// The maximum number of total calls to built-in tools that can be processed in a response. This maximum number applies across all built-in tool calls, not per individual tool. Any further attempts to call a tool by the model will be ignored.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tool_calls: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,
    /// Model ID used to generate the response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Whether to allow the model to run tool calls in parallel. Defaults to `true`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    /// The unique ID of the previous response to the model. Use this to create multi-turn conversations.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    /// A stable identifier used to help detect users of your application that may be violating OpenAI's usage policies. The IDs should be a string that uniquely identifies each user.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_identifier: Option<String>,
    /// Whether to store the generated model response for later retrieval via API.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    /// If set to true, the model response data will be streamed to the client as it is generated using server-sent events.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Adjust the randomness of the generated text. Between 0.0 and 2.0. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
    ///
    /// We generally recommend altering this or top_p but not both.
    /// Defaults to `0.8`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Controls which (if any) function is called by the model.
    pub tool_choice: ToolChoice,
    /// An array of tools the model may call while generating a response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    /// An alternative to sampling with temperature. Limit the next token selection to a subset of tokens with a cumulative probability above a threshold `p`. The value should be between 0.0 and 1.0.
    ///
    /// Top-p sampling, also known as nucleus sampling, is another text generation method that selects the next token from a subset of tokens that together have a cumulative probability of at least `p`. This method provides a balance between diversity and quality by considering both the probabilities of tokens and the number of tokens to sample from. A higher value for top_p (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text.
    ///
    /// We generally recommend altering this or temperature but not both.
    /// Defaults to `0.9`. To disable top-p sampling, set it to `1.0`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// The truncation strategy to use for the model response.
    ///
    /// - `auto`: If the input to this Response exceeds the model's context window size, the model will truncate the response to fit the context window by dropping items from the beginning of the conversation.
    /// - `disabled` (default): If the input size will exceed the context window size for a model, the request will fail with a 400 error.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<String>,
}
impl<'de> serde::Deserialize<'de> for RequestOfModelResponse {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::{self, MapAccess, Visitor};
        use std::fmt;

        struct RequestOfModelResponseVisitor;

        impl<'de> Visitor<'de> for RequestOfModelResponseVisitor {
            type Value = RequestOfModelResponse;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct RequestOfModelResponse")
            }

            fn visit_map<V>(self, mut map: V) -> Result<RequestOfModelResponse, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut background = None;
                let mut conversation = None;
                let mut include = None;
                let mut input = None;
                let mut instructions = None;
                let mut max_output_tokens = None;
                let mut max_tool_calls = None;
                let mut metadata = None;
                let mut model = None;
                let mut parallel_tool_calls = None;
                let mut previous_response_id = None;
                let mut safety_identifier = None;
                let mut store = None;
                let mut stream = None;
                let mut temperature = None;
                let mut tool_choice = None;
                let mut tools = None;
                let mut top_p = None;
                let mut truncation = None;

                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "background" => {
                            if background.is_some() {
                                return Err(de::Error::duplicate_field("background"));
                            }
                            background = Some(map.next_value()?);
                        }
                        "conversation" => {
                            if conversation.is_some() {
                                return Err(de::Error::duplicate_field("conversation"));
                            }
                            conversation = Some(map.next_value()?);
                        }
                        "include" => {
                            if include.is_some() {
                                return Err(de::Error::duplicate_field("include"));
                            }
                            include = Some(map.next_value()?);
                        }
                        "input" => {
                            if input.is_some() {
                                return Err(de::Error::duplicate_field("input"));
                            }
                            input = Some(map.next_value()?);
                        }
                        "instructions" => {
                            if instructions.is_some() {
                                return Err(de::Error::duplicate_field("instructions"));
                            }
                            instructions = Some(map.next_value()?);
                        }
                        "max_output_tokens" => {
                            if max_output_tokens.is_some() {
                                return Err(de::Error::duplicate_field("max_output_tokens"));
                            }
                            max_output_tokens = Some(map.next_value()?);
                        }
                        "max_tool_calls" => {
                            if max_tool_calls.is_some() {
                                return Err(de::Error::duplicate_field("max_tool_calls"));
                            }
                            max_tool_calls = Some(map.next_value()?);
                        }
                        "metadata" => {
                            if metadata.is_some() {
                                return Err(de::Error::duplicate_field("metadata"));
                            }
                            metadata = Some(map.next_value()?);
                        }
                        "model" => {
                            if model.is_some() {
                                return Err(de::Error::duplicate_field("model"));
                            }
                            model = Some(map.next_value()?);
                        }
                        "parallel_tool_calls" => {
                            if parallel_tool_calls.is_some() {
                                return Err(de::Error::duplicate_field("parallel_tool_calls"));
                            }
                            parallel_tool_calls = Some(map.next_value()?);
                        }
                        "previous_response_id" => {
                            if previous_response_id.is_some() {
                                return Err(de::Error::duplicate_field("previous_response_id"));
                            }
                            previous_response_id = Some(map.next_value()?);
                        }
                        "safety_identifier" => {
                            if safety_identifier.is_some() {
                                return Err(de::Error::duplicate_field("safety_identifier"));
                            }
                            safety_identifier = Some(map.next_value()?);
                        }
                        "store" => {
                            if store.is_some() {
                                return Err(de::Error::duplicate_field("store"));
                            }
                            store = Some(map.next_value()?);
                        }
                        "stream" => {
                            if stream.is_some() {
                                return Err(de::Error::duplicate_field("stream"));
                            }
                            stream = Some(map.next_value()?);
                        }
                        "temperature" => {
                            if temperature.is_some() {
                                return Err(de::Error::duplicate_field("temperature"));
                            }
                            temperature = Some(map.next_value()?);
                        }
                        "tool_choice" => {
                            if tool_choice.is_some() {
                                return Err(de::Error::duplicate_field("tool_choice"));
                            }
                            tool_choice = Some(map.next_value()?);
                        }
                        "tools" => {
                            if tools.is_some() {
                                return Err(de::Error::duplicate_field("tools"));
                            }
                            tools = Some(map.next_value()?);
                        }
                        "top_p" => {
                            if top_p.is_some() {
                                return Err(de::Error::duplicate_field("top_p"));
                            }
                            top_p = Some(map.next_value()?);
                        }
                        "truncation" => {
                            if truncation.is_some() {
                                return Err(de::Error::duplicate_field("truncation"));
                            }
                            truncation = Some(map.next_value()?);
                        }
                        _ => {
                            // 忽略未知字段
                            let _: serde::de::IgnoredAny = map.next_value()?;
                        }
                    }
                }

                let tool_choice = tool_choice.unwrap_or(ToolChoice::None);
                let parallel_tool_calls = if parallel_tool_calls.is_none() {
                    Some(true)
                } else {
                    parallel_tool_calls
                };

                Ok(RequestOfModelResponse {
                    background,
                    conversation,
                    include,
                    input,
                    instructions,
                    max_output_tokens,
                    max_tool_calls,
                    metadata,
                    model,
                    parallel_tool_calls,
                    previous_response_id,
                    safety_identifier,
                    store,
                    stream,
                    temperature,
                    tool_choice,
                    tools,
                    top_p,
                    truncation,
                })
            }
        }

        const FIELDS: &[&str] = &[
            "background",
            "conversation",
            "include",
            "input",
            "instructions",
            "max_output_tokens",
            "max_tool_calls",
            "metadata",
            "model",
            "parallel_tool_calls",
            "previous_response_id",
            "safety_identifier",
            "store",
            "stream",
            "temperature",
            "tool_choice",
            "tools",
            "top_p",
            "truncation",
        ];

        deserializer.deserialize_struct(
            "RequestOfModelResponse",
            FIELDS,
            RequestOfModelResponseVisitor,
        )
    }
}

/// Represents a response object
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseObject {
    /// Whether to run the model response in the background.
    pub background: bool,
    /// The conversation that this response belongs to. Input items and output items from this response are automatically added to this conversation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation: Option<Conversation>,
    /// Unix timestamp (in seconds) of when this Response was created.
    pub created_at: u64,
    /// An error object returned when the model fails to generate a Response.
    pub error: Option<ResponseObjectError>,
    /// Unique identifier for this Response.
    pub id: String,
    /// Details about why the response is incomplete.
    pub incomplete_details: Option<ResponseObjectIncompleteDetails>,
    /// A system (or developer) message inserted into the model's context.
    ///
    /// When using along with `previous_response_id`, the instructions from a previous response will not be carried over to the next response.
    pub instructions: Option<Input>,
    /// An upper bound for the number of tokens that can be generated for a response, including visible output tokens and reasoning tokens.
    pub max_output_tokens: Option<u32>,
    /// The maximum number of total calls to built-in tools that can be processed in a response. This maximum number applies across all built-in tool calls, not per individual tool. Any further attempts to call a tool by the model will be ignored.
    pub max_tool_calls: Option<u32>,
    pub metadata: HashMap<String, String>,
    /// Model ID used to generate the response.
    pub model: String,
    /// The object type of this resource - always set to `response`.
    pub object: String,
    /// An array of content items generated by the model.
    pub output: Vec<ResponseOutputItem>,
    /// Whether to allow the model to run tool calls in parallel.
    pub parallel_tool_calls: bool,
    /// The unique ID of the previous response to the model. Use this to create multi-turn conversations.
    pub previous_response_id: Option<String>,
    /// A stable identifier used to help detect users of your application that may be violating OpenAI's usage policies. The IDs should be a string that uniquely identifies each user.
    pub safety_identifier: Option<String>,
    /// The status of the response generation. One of `completed`, `failed`, `in_progress`, `cancelled`, `queued`, or `incomplete`.
    pub status: String,
    /// What sampling temperature to use.
    pub temperature: f64,
    /// Controls which (if any) function is called by the model.
    pub tool_choice: ToolChoice,
    /// An array of tools the model may call while generating a response.
    pub tools: Option<Vec<Tool>>,
    /// An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass.
    /// It's recommended that altering this or temperature but not both.
    pub top_p: f64,
    /// The truncation strategy to use for the model response.
    ///
    /// - `auto`: If the input to this Response exceeds the model's context window size, the model will truncate the response to fit the context window by dropping items from the beginning of the conversation.
    /// - `disabled` (default): If the input size will exceed the context window size for a model, the request will fail with a 400 error.
    pub truncation: Option<String>,
    /// Represents token usage details including input tokens, output tokens, a breakdown of output tokens, and the total tokens used.
    pub usage: Usage,
}

/// Represents a conversation, either by ID or as an object with an ID.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Conversation {
    /// The unique ID of the conversation.
    Id(String),
    /// The conversation that this response belongs to.
    ConversationObject {
        /// The unique ID of the conversation.
        id: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseObjectError {
    /// The error code for the response.
    pub code: String,
    /// A human-readable description of the error.
    pub message: String,
}

/// Represents details about why the response is incomplete.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseObjectIncompleteDetails {
    /// The reason why the response is incomplete.
    pub reason: String,
}

/// Represents text, image, or file inputs to the model, used to generate a response.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Input {
    /// A text input to the model, equivalent to a text input with the `developer` role.
    Text(String),
    /// A list of one or many input items to the model, containing different content types.
    InputItemList(Vec<InputItem>),
}

/// Represents an input item to the model, containing different content types.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum InputItem {
    /// A message input to the model with a role indicating instruction following hierarchy. Instructions given with the developer or system role take precedence over instructions given with the user role. Messages with the assistant role are presumed to have been generated by the model in previous interactions.
    InputMessage {
        /// Text, image, or audio input to the model, used to generate a response. Can also contain previous assistant responses.
        content: InputMessageContent,
        /// The role of the message input. One of `user`, `assistant`, `system`, or `developer`.
        role: String,
        /// The type of the message input. Always `message`.
        #[serde(rename = "type")]
        ty: String,
    },
    /// An item representing part of the context for the response to be generated by the model. Can contain text, images, and audio inputs, as well as previous assistant responses and tool call
    Item(ResponseItem),
    /// An internal identifier for an item to reference.
    ItemReference(InputItemReference),
}

/// Represents the content of a message input to the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum InputMessageContent {
    /// A text input to the model.
    Text(String),
    /// A list of one or many input items to the model, containing different content types.
    InputItemContentList(Vec<ResponseItemInputMessageContent>),
}

/// Represents a reference to an input item by its unique ID
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputItemReference {
    /// The unique ID of the input item to reference.
    pub id: String,
    /// The type of item to reference. Always `item_reference`.
    pub ty: String,
}

/// Controls which (if any) function is called by the model. Defaults to `None`.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub enum ToolChoice {
    /// Controls which (if any) tool is called by the model.
    /// `none` means the model will not call any tool and instead generates a message.
    /// `auto` means the model can pick between generating a message or calling one or more tools.
    /// `required` means the model must call one or more tools.
    #[serde(rename = "none")]
    None,
    #[serde(rename = "auto")]
    Auto,
    #[serde(rename = "required")]
    Required,
    #[serde(untagged)]
    AllowedTools {
        /// Constrains the tools available to the model to a pre-defined set.
        /// `auto` allows the model to pick from among the allowed tools and generate a message.
        /// `required` requires the model to call one or more of the allowed tools.
        mode: String,
        /// A list of tool definitions that the model should be allowed to call.
        tools: Vec<ToolChoiceMcpTool>,
    },
    #[serde(untagged)]
    McpTool(ToolChoiceMcpTool),
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct ToolChoiceMcpTool {
    /// The label of the MCP server to use.
    pub server_label: String,
    /// The name of the tool to call on the server.
    pub name: String,
    /// The type of the tool. Always `mcp`.
    #[serde(rename = "type")]
    pub ty: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct Tool {
    /// The name of the function to call.
    pub name: String,
    /// A JSON schema object describing the parameters of the function.
    pub parameters: JsonObject,
    /// Whether to enforce strict parameter validation. Default `true`.
    pub strict: bool,
    /// The type of the function tool. Always `function`.
    #[serde(rename = "type")]
    pub ty: String,
    /// A description of the function. Used by the model to determine whether or not to call the function.
    pub description: String,
}

/// Represents token usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    /// The number of input tokens.
    pub input_tokens: u64,
    /// A detailed breakdown of the input tokens.
    pub input_tokens_details: InputTokensDetails,
    /// The number of output tokens.
    pub output_tokens: u64,
    /// A detailed breakdown of the output tokens.
    pub output_tokens_details: OutputTokensDetails,
    /// The total number of tokens used.
    pub total_tokens: u64,
}

/// Represents input token details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputTokensDetails {
    /// The number of tokens that were retrieved from the cache.
    pub cached_tokens: u32,
}

/// Represents output token details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputTokensDetails {
    /// The number of reasoning tokens.
    pub reasoning_tokens: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json;

    #[test]
    fn test_tool_choice_none_serialization() {
        let tool_choice = ToolChoice::None;
        let json = serde_json::to_string(&tool_choice).expect("Failed to serialize");
        assert_eq!(json, r#""none""#);

        let deserialized: ToolChoice = serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(deserialized, tool_choice);
    }

    #[test]
    fn test_tool_choice_auto_serialization() {
        let tool_choice = ToolChoice::Auto;
        let json = serde_json::to_string(&tool_choice).expect("Failed to serialize");
        assert_eq!(json, r#""auto""#);

        let deserialized: ToolChoice = serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(deserialized, tool_choice);
    }

    #[test]
    fn test_tool_choice_required_serialization() {
        let tool_choice = ToolChoice::Required;
        let json = serde_json::to_string(&tool_choice).expect("Failed to serialize");
        assert_eq!(json, r#""required""#);

        let deserialized: ToolChoice = serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(deserialized, tool_choice);
    }

    #[test]
    fn test_tool_choice_allowed_tools_serialization() {
        let tool_choice = ToolChoice::AllowedTools {
            mode: "auto".to_string(),
            tools: vec![
                ToolChoiceMcpTool {
                    server_label: "weather_server".to_string(),
                    name: "get_weather".to_string(),
                    ty: "mcp".to_string(),
                },
                ToolChoiceMcpTool {
                    server_label: "calculator_server".to_string(),
                    name: "calculate".to_string(),
                    ty: "mcp".to_string(),
                },
            ],
        };

        let json = serde_json::to_string(&tool_choice).expect("Failed to serialize");
        let expected_json = r#"{"mode":"auto","tools":[{"server_label":"weather_server","name":"get_weather","type":"mcp"},{"server_label":"calculator_server","name":"calculate","type":"mcp"}]}"#;
        assert_eq!(json, expected_json);

        let deserialized: ToolChoice = serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(deserialized, tool_choice);
    }

    #[test]
    fn test_tool_choice_allowed_tools_required_mode_serialization() {
        let tool_choice = ToolChoice::AllowedTools {
            mode: "required".to_string(),
            tools: vec![ToolChoiceMcpTool {
                server_label: "file_server".to_string(),
                name: "read_file".to_string(),
                ty: "mcp".to_string(),
            }],
        };

        let json = serde_json::to_string(&tool_choice).expect("Failed to serialize");
        let expected_json = r#"{"mode":"required","tools":[{"server_label":"file_server","name":"read_file","type":"mcp"}]}"#;
        assert_eq!(json, expected_json);

        let deserialized: ToolChoice = serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(deserialized, tool_choice);
    }

    #[test]
    fn test_tool_choice_mcp_tool_serialization() {
        let tool_choice = ToolChoice::McpTool(ToolChoiceMcpTool {
            server_label: "database_server".to_string(),
            name: "query_database".to_string(),
            ty: "mcp".to_string(),
        });

        let json = serde_json::to_string(&tool_choice).expect("Failed to serialize");
        let expected_json =
            r#"{"server_label":"database_server","name":"query_database","type":"mcp"}"#;
        assert_eq!(json, expected_json);

        let deserialized: ToolChoice = serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(deserialized, tool_choice);
    }

    #[test]
    fn test_tool_choice_deserialization_from_object() {
        // Test deserialization of object values
        let json = r#"{
            "mode": "auto",
            "tools": [
                {
                    "server_label": "example_server",
                    "name": "example_tool",
                    "type": "mcp"
                }
            ]
        }"#;

        let tool_choice: ToolChoice = serde_json::from_str(json).expect("Failed to deserialize");

        if let ToolChoice::AllowedTools { mode, tools } = tool_choice {
            assert_eq!(mode, "auto");
            assert_eq!(tools.len(), 1);
            assert_eq!(tools[0].server_label, "example_server");
            assert_eq!(tools[0].name, "example_tool");
            assert_eq!(tools[0].ty, "mcp");
        } else {
            panic!("Expected AllowedTools variant");
        }
    }

    #[test]
    fn test_tool_choice_mcp_tool_deserialization() {
        let json = r#"{
            "server_label": "mcp_server",
            "name": "mcp_tool",
            "type": "mcp"
        }"#;

        let tool_choice: ToolChoice = serde_json::from_str(json).expect("Failed to deserialize");

        if let ToolChoice::McpTool(tool) = tool_choice {
            assert_eq!(tool.server_label, "mcp_server");
            assert_eq!(tool.name, "mcp_tool");
            assert_eq!(tool.ty, "mcp");
        } else {
            panic!("Expected McpTool variant");
        }
    }

    #[test]
    fn test_tool_choice_mode_serialization_format() {
        // Test that Mode variant serializes directly to string without wrapper
        let tool_choice = ToolChoice::Auto;
        let json = serde_json::to_string(&tool_choice).expect("Failed to serialize");

        // Should be just "auto", not {"Mode":"auto"}
        assert_eq!(json, r#""auto""#);

        let tool_choice = ToolChoice::None;
        let json = serde_json::to_string(&tool_choice).expect("Failed to serialize");
        assert_eq!(json, r#""none""#);

        let tool_choice = ToolChoice::Required;
        let json = serde_json::to_string(&tool_choice).expect("Failed to serialize");
        assert_eq!(json, r#""required""#);
    }

    #[test]
    fn test_response_object_deserialization() {
        let json = r#"{
  "id": "resp_67ca09c5efe0819096d0511c92b8c890096610f474011cc0",
  "background": false,
  "object": "response",
  "created_at": 1741294021,
  "status": "completed",
  "error": null,
  "incomplete_details": null,
  "instructions": null,
  "max_output_tokens": null,
  "model": "gpt-4.1-2025-04-14",
  "output": [
    {
      "type": "function_call",
      "id": "fc_67ca09c6bedc8190a7abfec07b1a1332096610f474011cc0",
      "call_id": "call_unLAR8MvFNptuiZK6K6HCy5k",
      "name": "get_current_weather",
      "arguments": "{\"location\":\"Boston, MA\",\"unit\":\"celsius\"}",
      "status": "completed"
    }
  ],
  "parallel_tool_calls": true,
  "previous_response_id": null,
  "reasoning": {
    "effort": null,
    "summary": null
  },
  "store": true,
  "temperature": 1.0,
  "text": {
    "format": {
      "type": "text"
    }
  },
  "tool_choice": "auto",
  "tools": [
    {
      "type": "function",
      "description": "Get the current weather in a given location",
      "name": "get_current_weather",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "The city and state, e.g. San Francisco, CA"
          },
          "unit": {
            "type": "string",
            "enum": [
              "celsius",
              "fahrenheit"
            ]
          }
        },
        "required": [
          "location",
          "unit"
        ]
      },
      "strict": true
    }
  ],
  "top_p": 1.0,
  "truncation": "disabled",
  "usage": {
    "input_tokens": 291,
    "input_tokens_details": {
      "cached_tokens": 0
    },
    "output_tokens": 23,
    "output_tokens_details": {
      "reasoning_tokens": 0
    },
    "total_tokens": 314
  },
  "user": null,
  "metadata": {}
}"#;

        // Try to deserialize the JSON
        let result: Result<ResponseObject, _> = serde_json::from_str(json);

        match result {
            Ok(response) => {
                // If deserialization succeeds, verify key fields
                assert_eq!(
                    response.id,
                    "resp_67ca09c5efe0819096d0511c92b8c890096610f474011cc0"
                );
                assert_eq!(response.object, "response");
                assert_eq!(response.created_at, 1741294021);
                assert_eq!(response.status, "completed");
                assert_eq!(response.model, "gpt-4.1-2025-04-14");
                assert_eq!(response.parallel_tool_calls, true);
                assert_eq!(response.temperature, 1.0);
                assert_eq!(response.top_p, 1.0);

                // Check usage
                assert_eq!(response.usage.input_tokens, 291);
                assert_eq!(response.usage.output_tokens, 23);
                assert_eq!(response.usage.total_tokens, 314);
                assert_eq!(response.usage.output_tokens_details.reasoning_tokens, 0);

                // Check output
                assert_eq!(response.output.len(), 1);
                if let ResponseOutputItem::FunctionCall {
                    arguments,
                    call_id,
                    id,
                    name,
                    ty: _,
                    status,
                } = &response.output[0]
                {
                    assert_eq!(id, "fc_67ca09c6bedc8190a7abfec07b1a1332096610f474011cc0");
                    assert_eq!(call_id, "call_unLAR8MvFNptuiZK6K6HCy5k");
                    assert_eq!(name, "get_current_weather");
                    assert_eq!(
                        arguments,
                        "{\"location\":\"Boston, MA\",\"unit\":\"celsius\"}"
                    );
                    assert_eq!(status, "completed");
                } else {
                    panic!("Expected FunctionToolCall variant");
                }

                // Check tools
                assert!(response.tools.is_some());
                let tools = response.tools.as_ref().unwrap();
                assert_eq!(tools.len(), 1);
                assert_eq!(tools[0].name, "get_current_weather");
                assert_eq!(
                    tools[0].description,
                    "Get the current weather in a given location"
                );
                assert_eq!(tools[0].strict, true);

                println!("✅ ResponseObject deserialization successful!");
            }
            Err(e) => {
                // If deserialization fails, print the error and identify missing fields
                println!("❌ ResponseObject deserialization failed: {}", e);
                println!("This indicates that some fields in the JSON are not present in the ResponseObject struct.");
                println!("Missing fields likely include: reasoning, store, text, truncation, user");

                // For now, we'll expect this to fail since the struct is missing some fields
                // In a real scenario, we'd need to add these fields to the struct
                panic!("Expected deserialization to succeed, but got error: {}", e);
            }
        }
    }

    #[test]
    fn test_response_object_basic_serialization() {
        let response = ResponseObject {
            background: false,
            conversation: None,
            created_at: 1693123456,
            error: None,
            id: "resp_test123".to_string(),
            incomplete_details: None,
            instructions: None,
            max_output_tokens: None,
            max_tool_calls: None,
            metadata: HashMap::new(),
            model: "gpt-4".to_string(),
            object: "response".to_string(),
            output: vec![ResponseOutputItem::OutputMessage {
                content: vec![ResponseOutputItemOutputMessageContent::OutputText {
                    annotations: vec![],
                    text: "Hello, world!".to_string(),
                    ty: "text".to_string(),
                    logprobs: None,
                }],
                id: "msg_123".to_string(),
                role: "assistant".to_string(),
                status: "completed".to_string(),
                ty: "message".to_string(),
            }],
            parallel_tool_calls: true,
            previous_response_id: None,
            safety_identifier: None,
            status: "completed".to_string(),
            temperature: 1.0,
            tool_choice: ToolChoice::Auto,
            tools: None,
            top_p: 1.0,
            truncation: None,
            usage: Usage {
                input_tokens: 10,
                input_tokens_details: InputTokensDetails { cached_tokens: 0 },
                output_tokens: 5,
                output_tokens_details: OutputTokensDetails {
                    reasoning_tokens: 0,
                },
                total_tokens: 15,
            },
        };

        let serialized = serde_json::to_string_pretty(&response).unwrap();
        println!("Serialized ResponseObject:\n{}", serialized);

        let deserialized: ResponseObject = serde_json::from_str(&serialized).unwrap();

        // Verify key fields match
        assert_eq!(response.id, deserialized.id);
        assert_eq!(response.model, deserialized.model);
        assert_eq!(response.object, deserialized.object);
        assert_eq!(response.status, deserialized.status);
        assert_eq!(response.created_at, deserialized.created_at);
        assert_eq!(response.temperature, deserialized.temperature);
        assert_eq!(response.top_p, deserialized.top_p);
        assert_eq!(
            response.parallel_tool_calls,
            deserialized.parallel_tool_calls
        );
        assert_eq!(response.tool_choice, deserialized.tool_choice);
        assert_eq!(response.usage.input_tokens, deserialized.usage.input_tokens);
        assert_eq!(
            response.usage.output_tokens,
            deserialized.usage.output_tokens
        );
        assert_eq!(response.usage.total_tokens, deserialized.usage.total_tokens);
        assert_eq!(response.output.len(), deserialized.output.len());
    }

    #[test]
    fn test_response_object_comprehensive_serialization() {
        use std::collections::HashMap;

        let mut metadata = HashMap::new();
        metadata.insert("user_id".to_string(), "user123".to_string());
        metadata.insert("session_id".to_string(), "session456".to_string());

        let response = ResponseObject {
            background: false,
            conversation: Some(Conversation::ConversationObject {
                id: "conv_123".to_string(),
            }),
            created_at: 1693123456,
            error: None,
            id: "resp_comprehensive_test".to_string(),
            incomplete_details: None,
            instructions: Some(Input::Text("You are a helpful assistant.".to_string())),
            max_output_tokens: Some(1000),
            max_tool_calls: Some(5),
            metadata,
            model: "gpt-4-turbo".to_string(),
            object: "response".to_string(),
            output: vec![
                ResponseOutputItem::OutputMessage {
                    content: vec![ResponseOutputItemOutputMessageContent::OutputText {
                        annotations: vec![],
                        text: "I can help you with that.".to_string(),
                        ty: "text".to_string(),
                        logprobs: None,
                    }],
                    id: "msg_456".to_string(),
                    role: "assistant".to_string(),
                    status: "completed".to_string(),
                    ty: "message".to_string(),
                },
                ResponseOutputItem::FunctionCall {
                    arguments: r#"{"location": "New York", "unit": "celsius"}"#.to_string(),
                    call_id: "call_789".to_string(),
                    id: "fc_abc".to_string(),
                    name: "get_weather".to_string(),
                    ty: "function_call".to_string(),
                    status: "completed".to_string(),
                },
            ],
            parallel_tool_calls: false,
            previous_response_id: Some("resp_previous".to_string()),
            safety_identifier: Some("safety_123".to_string()),
            status: "completed".to_string(),
            temperature: 0.7,
            tool_choice: ToolChoice::Required,
            tools: Some(vec![Tool {
                name: "get_weather".to_string(),
                parameters: {
                    let mut params = JsonObject::new();
                    params.insert("type".to_string(), serde_json::json!("object"));
                    params.insert(
                        "properties".to_string(),
                        serde_json::json!({
                            "location": {"type": "string"},
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                        }),
                    );
                    params.insert("required".to_string(), serde_json::json!(["location"]));
                    params
                },
                strict: true,
                ty: "function".to_string(),
                description: "Get weather information".to_string(),
            }]),
            top_p: 0.9,
            truncation: Some("auto".to_string()),
            usage: Usage {
                input_tokens: 50,
                input_tokens_details: InputTokensDetails { cached_tokens: 10 },
                output_tokens: 25,
                output_tokens_details: OutputTokensDetails {
                    reasoning_tokens: 5,
                },
                total_tokens: 75,
            },
        };

        let serialized = serde_json::to_string_pretty(&response).unwrap();
        let deserialized: ResponseObject = serde_json::from_str(&serialized).unwrap();

        // Verify all fields match
        assert_eq!(response.background, deserialized.background);
        assert_eq!(response.id, deserialized.id);
        assert_eq!(response.model, deserialized.model);
        assert_eq!(response.object, deserialized.object);
        assert_eq!(response.status, deserialized.status);
        assert_eq!(response.created_at, deserialized.created_at);
        assert_eq!(response.max_output_tokens, deserialized.max_output_tokens);
        assert_eq!(response.max_tool_calls, deserialized.max_tool_calls);
        assert_eq!(response.metadata, deserialized.metadata);
        assert_eq!(
            response.parallel_tool_calls,
            deserialized.parallel_tool_calls
        );
        assert_eq!(
            response.previous_response_id,
            deserialized.previous_response_id
        );
        assert_eq!(response.safety_identifier, deserialized.safety_identifier);
        assert_eq!(response.temperature, deserialized.temperature);
        assert_eq!(response.tool_choice, deserialized.tool_choice);
        assert_eq!(response.top_p, deserialized.top_p);
        assert_eq!(response.truncation, deserialized.truncation);
        assert_eq!(response.output.len(), deserialized.output.len());
        assert_eq!(response.usage.input_tokens, deserialized.usage.input_tokens);
        assert_eq!(
            response.usage.output_tokens,
            deserialized.usage.output_tokens
        );
        assert_eq!(response.usage.total_tokens, deserialized.usage.total_tokens);

        // Verify optional nested fields
        assert_eq!(
            response.usage.input_tokens_details.cached_tokens,
            deserialized.usage.input_tokens_details.cached_tokens
        );
        assert_eq!(
            response.usage.output_tokens_details.reasoning_tokens,
            deserialized.usage.output_tokens_details.reasoning_tokens
        );

        // Verify tools
        assert!(deserialized.tools.is_some());
        let tools = deserialized.tools.as_ref().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "get_weather");
        assert_eq!(tools[0].description, "Get weather information");
    }

    #[test]
    fn test_response_object_output_variants_serialization() {
        let response = ResponseObject {
            background: false,
            conversation: None,
            created_at: 1693123456,
            error: None,
            id: "resp_variants_test".to_string(),
            incomplete_details: None,
            instructions: None,
            max_output_tokens: None,
            max_tool_calls: None,
            metadata: HashMap::new(),
            model: "gpt-4".to_string(),
            object: "response".to_string(),
            output: vec![
                ResponseOutputItem::OutputMessage {
                    content: vec![
                        ResponseOutputItemOutputMessageContent::OutputText {
                            annotations: vec![],
                            text: "Here's the information you requested.".to_string(),
                            ty: "text".to_string(),
                            logprobs: None,
                        },
                        ResponseOutputItemOutputMessageContent::Refusal {
                            refusal: "I cannot provide that information.".to_string(),
                            ty: "refusal".to_string(),
                        },
                    ],
                    id: "msg_variants".to_string(),
                    role: "assistant".to_string(),
                    status: "completed".to_string(),
                    ty: "message".to_string(),
                },
                ResponseOutputItem::FunctionCall {
                    arguments: r#"{"query": "test query"}"#.to_string(),
                    call_id: "call_test123".to_string(),
                    id: "fc_test123".to_string(),
                    name: "search_function".to_string(),
                    ty: "function_call".to_string(),
                    status: "in_progress".to_string(),
                },
                ResponseOutputItem::ImageGeneration {
                    id: "img_test123".to_string(),
                    result: "base64_encoded_image_data".to_string(),
                    status: "completed".to_string(),
                    ty: "image_generation_call".to_string(),
                },
                ResponseOutputItem::McpToolCall {
                    arguments: r#"{"param": "value"}"#.to_string(),
                    id: "mcp_test123".to_string(),
                    name: "mcp_tool".to_string(),
                    server_label: "test_server".to_string(),
                    ty: "mcp_call".to_string(),
                    error: "".to_string(),
                    output: "tool output".to_string(),
                },
            ],
            parallel_tool_calls: true,
            previous_response_id: None,
            safety_identifier: None,
            status: "completed".to_string(),
            temperature: 1.0,
            tool_choice: ToolChoice::Auto,
            tools: None,
            top_p: 1.0,
            truncation: None,
            usage: Usage {
                input_tokens: 20,
                input_tokens_details: InputTokensDetails { cached_tokens: 0 },
                output_tokens: 15,
                output_tokens_details: OutputTokensDetails {
                    reasoning_tokens: 0,
                },
                total_tokens: 35,
            },
        };

        let serialized = serde_json::to_string_pretty(&response).unwrap();
        let deserialized: ResponseObject = serde_json::from_str(&serialized).unwrap();

        // Verify the output items were correctly serialized/deserialized
        assert_eq!(response.output.len(), deserialized.output.len());
        assert_eq!(response.output.len(), 4);

        // Verify each output item type
        match &deserialized.output[0] {
            ResponseOutputItem::OutputMessage { content, .. } => {
                assert_eq!(content.len(), 2);
                match &content[0] {
                    ResponseOutputItemOutputMessageContent::OutputText { text, .. } => {
                        assert_eq!(text, "Here's the information you requested.");
                    }
                    _ => panic!("Expected OutputText variant"),
                }
                match &content[1] {
                    ResponseOutputItemOutputMessageContent::Refusal { refusal, .. } => {
                        assert_eq!(refusal, "I cannot provide that information.");
                    }
                    _ => panic!("Expected Refusal variant"),
                }
            }
            _ => panic!("Expected OutputMessage variant"),
        }

        match &deserialized.output[1] {
            ResponseOutputItem::FunctionCall { name, status, .. } => {
                assert_eq!(name, "search_function");
                assert_eq!(status, "in_progress");
            }
            _ => panic!("Expected FunctionCall variant"),
        }

        match &deserialized.output[2] {
            ResponseOutputItem::ImageGeneration { result, .. } => {
                assert_eq!(result, "base64_encoded_image_data");
            }
            _ => panic!("Expected ImageGeneration variant"),
        }

        match &deserialized.output[3] {
            ResponseOutputItem::McpToolCall {
                server_label,
                output,
                ..
            } => {
                assert_eq!(server_label, "test_server");
                assert_eq!(output, "tool output");
            }
            _ => panic!("Expected McpToolCall variant"),
        }
    }

    #[test]
    fn test_response_object_error_scenarios_serialization() {
        // Test ResponseObject with error
        let response_with_error = ResponseObject {
            background: false,
            conversation: None,
            created_at: 1693123456,
            error: Some(ResponseObjectError {
                code: "rate_limit_exceeded".to_string(),
                message: "You have exceeded the rate limit.".to_string(),
            }),
            id: "resp_error_test".to_string(),
            incomplete_details: Some(ResponseObjectIncompleteDetails {
                reason: "max_tokens".to_string(),
            }),
            instructions: None,
            max_output_tokens: None,
            max_tool_calls: None,
            metadata: HashMap::new(),
            model: "gpt-4".to_string(),
            object: "response".to_string(),
            output: vec![],
            parallel_tool_calls: true,
            previous_response_id: None,
            safety_identifier: None,
            status: "failed".to_string(),
            temperature: 1.0,
            tool_choice: ToolChoice::None,
            tools: None,
            top_p: 1.0,
            truncation: None,
            usage: Usage {
                input_tokens: 100,
                input_tokens_details: InputTokensDetails { cached_tokens: 0 },
                output_tokens: 0,
                output_tokens_details: OutputTokensDetails {
                    reasoning_tokens: 0,
                },
                total_tokens: 100,
            },
        };

        let serialized = serde_json::to_string_pretty(&response_with_error).unwrap();
        let deserialized: ResponseObject = serde_json::from_str(&serialized).unwrap();

        // Verify error fields
        assert!(deserialized.error.is_some());
        let error = deserialized.error.as_ref().unwrap();
        assert_eq!(error.code, "rate_limit_exceeded");
        assert_eq!(error.message, "You have exceeded the rate limit.");

        // Verify incomplete details
        assert!(deserialized.incomplete_details.is_some());
        let incomplete = deserialized.incomplete_details.as_ref().unwrap();
        assert_eq!(incomplete.reason, "max_tokens");

        // Verify other fields
        assert_eq!(deserialized.status, "failed");
        assert_eq!(deserialized.tool_choice, ToolChoice::None);
        assert_eq!(deserialized.output.len(), 0);
        assert_eq!(deserialized.usage.output_tokens, 0);
    }

    #[test]
    fn test_conversation_variants_serialization() {
        // Test with Conversation::Id variant
        let response_with_id = ResponseObject {
            background: false,
            conversation: Some(Conversation::Id("conv_simple_id".to_string())),
            created_at: 1693123456,
            error: None,
            id: "resp_conv_id_test".to_string(),
            incomplete_details: None,
            instructions: None,
            max_output_tokens: None,
            max_tool_calls: None,
            metadata: HashMap::new(),
            model: "gpt-4".to_string(),
            object: "response".to_string(),
            output: vec![],
            parallel_tool_calls: true,
            previous_response_id: None,
            safety_identifier: None,
            status: "completed".to_string(),
            temperature: 1.0,
            tool_choice: ToolChoice::Auto,
            tools: None,
            top_p: 1.0,
            truncation: None,
            usage: Usage {
                input_tokens: 10,
                input_tokens_details: InputTokensDetails { cached_tokens: 0 },
                output_tokens: 5,
                output_tokens_details: OutputTokensDetails {
                    reasoning_tokens: 0,
                },
                total_tokens: 15,
            },
        };

        let serialized = serde_json::to_string_pretty(&response_with_id).unwrap();
        let deserialized: ResponseObject = serde_json::from_str(&serialized).unwrap();

        // Verify conversation deserialization
        assert!(deserialized.conversation.is_some());
        match deserialized.conversation.as_ref().unwrap() {
            Conversation::Id(id) => assert_eq!(id, "conv_simple_id"),
            _ => panic!("Expected Conversation::Id variant"),
        }

        // Test with Conversation::ConversationObject variant
        let response_with_object = ResponseObject {
            conversation: Some(Conversation::ConversationObject {
                id: "conv_object_id".to_string(),
            }),
            ..response_with_id.clone()
        };

        let serialized_obj = serde_json::to_string_pretty(&response_with_object).unwrap();
        let deserialized_obj: ResponseObject = serde_json::from_str(&serialized_obj).unwrap();

        match deserialized_obj.conversation.as_ref().unwrap() {
            Conversation::ConversationObject { id } => assert_eq!(id, "conv_object_id"),
            _ => panic!("Expected Conversation::ConversationObject variant"),
        }
    }

    #[test]
    fn test_response_object_real_json_deserialization() {
        let json = r#"{
  "id": "resp_67ccd3a9da748190baa7f1570fe91ac604becb25c45c1d41",
  "background": false,
  "object": "response",
  "created_at": 1741476777,
  "status": "completed",
  "error": null,
  "incomplete_details": null,
  "instructions": null,
  "max_output_tokens": null,
  "model": "gpt-4o-2024-08-06",
  "output": [
    {
      "type": "message",
      "id": "msg_67ccd3acc8d48190a77525dc6de64b4104becb25c45c1d41",
      "status": "completed",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "The image depicts a scenic landscape with a wooden boardwalk or pathway leading through lush, green grass under a blue sky with some clouds. The setting suggests a peaceful natural area, possibly a park or nature reserve. There are trees and shrubs in the background.",
          "annotations": [],
          "logprobs": []
        }
      ]
    }
  ],
  "parallel_tool_calls": true,
  "previous_response_id": null,
  "reasoning": {
    "effort": null,
    "summary": null
  },
  "store": true,
  "temperature": 1,
  "text": {
    "format": {
      "type": "text"
    }
  },
  "tool_choice": "auto",
  "tools": [],
  "top_p": 1,
  "truncation": "disabled",
  "usage": {
    "input_tokens": 328,
    "input_tokens_details": {
      "cached_tokens": 0
    },
    "output_tokens": 52,
    "output_tokens_details": {
      "reasoning_tokens": 0
    },
    "total_tokens": 380
  },
  "user": null,
  "metadata": {}
}"#;

        // Try to deserialize the JSON
        let result: Result<ResponseObject, _> = serde_json::from_str(json);

        match result {
            Ok(response) => {
                // Verify key fields from the JSON
                assert_eq!(
                    response.id,
                    "resp_67ccd3a9da748190baa7f1570fe91ac604becb25c45c1d41"
                );
                assert_eq!(response.object, "response");
                assert_eq!(response.created_at, 1741476777);
                assert_eq!(response.status, "completed");
                assert_eq!(response.model, "gpt-4o-2024-08-06");
                assert_eq!(response.parallel_tool_calls, true);
                assert_eq!(response.temperature, 1.0);
                assert_eq!(response.top_p, 1.0);
                assert_eq!(response.tool_choice, ToolChoice::Auto);
                assert!(response.tools.is_some());
                assert_eq!(response.tools.as_ref().unwrap().len(), 0); // Empty tools array
                assert_eq!(response.truncation, Some("disabled".to_string()));

                // Verify usage statistics
                assert_eq!(response.usage.input_tokens, 328);
                assert_eq!(response.usage.output_tokens, 52);
                assert_eq!(response.usage.total_tokens, 380);
                assert_eq!(response.usage.output_tokens_details.reasoning_tokens, 0);

                // Verify input_tokens_details
                assert_eq!(response.usage.input_tokens_details.cached_tokens, 0);

                // Verify output content
                assert_eq!(response.output.len(), 1);
                match &response.output[0] {
                    ResponseOutputItem::OutputMessage {
                        id,
                        status,
                        role,
                        content,
                        ty,
                    } => {
                        assert_eq!(id, "msg_67ccd3acc8d48190a77525dc6de64b4104becb25c45c1d41");
                        assert_eq!(status, "completed");
                        assert_eq!(role, "assistant");
                        assert_eq!(ty, "message");

                        assert_eq!(content.len(), 1);
                        match &content[0] {
                            ResponseOutputItemOutputMessageContent::OutputText {
                                text,
                                annotations,
                                ty,
                                logprobs,
                            } => {
                                assert_eq!(ty, "output_text");
                                assert_eq!(text, "The image depicts a scenic landscape with a wooden boardwalk or pathway leading through lush, green grass under a blue sky with some clouds. The setting suggests a peaceful natural area, possibly a park or nature reserve. There are trees and shrubs in the background.");
                                assert_eq!(annotations.len(), 0);
                                assert!(
                                    logprobs.is_some() && logprobs.as_ref().unwrap().is_empty()
                                );
                            }
                            _ => panic!("Expected OutputText variant"),
                        }
                    }
                    _ => panic!("Expected OutputMessage variant"),
                }

                // Verify optional fields are None as expected
                assert!(response.error.is_none());
                assert!(response.incomplete_details.is_none());
                assert!(response.instructions.is_none());
                assert!(response.max_output_tokens.is_none());
                assert!(response.previous_response_id.is_none());

                // Test serialization roundtrip
                let serialized = serde_json::to_string_pretty(&response).unwrap();
                let re_deserialized: ResponseObject = serde_json::from_str(&serialized).unwrap();

                // Verify roundtrip preserves key data
                assert_eq!(response.id, re_deserialized.id);
                assert_eq!(response.model, re_deserialized.model);
                assert_eq!(response.status, re_deserialized.status);
                assert_eq!(
                    response.usage.input_tokens,
                    re_deserialized.usage.input_tokens
                );
                assert_eq!(
                    response.usage.output_tokens,
                    re_deserialized.usage.output_tokens
                );
                assert_eq!(response.output.len(), re_deserialized.output.len());

                println!(
                    "✅ Real JSON ResponseObject deserialization and serialization successful!"
                );
            }
            Err(e) => {
                // If deserialization fails, print detailed error information
                println!("❌ ResponseObject deserialization failed: {}", e);
                println!("This indicates that some fields in the JSON are not handled by the ResponseObject struct.");

                // The JSON contains extra fields not in our struct: reasoning, store, text, user
                // This is expected behavior for now
                panic!("Expected deserialization to succeed after handling extra fields, but got error: {}", e);
            }
        }
    }

    #[test]
    fn test_create_model_response_with_image_input() {
        // Create a RequestOfModelResponse object that matches the JSON structure
        let request = RequestOfModelResponse {
            background: None,
            conversation: None,
            include: None,
            input: Some(Input::InputItemList(vec![
                InputItem::InputMessage {
                    role: "user".to_string(),
                    ty: "message".to_string(),
                    content: InputMessageContent::InputItemContentList(vec![
                        ResponseItemInputMessageContent::Text {
                            text: "what is in this image?".to_string(),
                            ty: "input_text".to_string(),
                        },
                        ResponseItemInputMessageContent::Image {
                            detail: "auto".to_string(),
                            ty: "input_image".to_string(),
                            file_id: "".to_string(),
                            image_url: "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg".to_string(),
                        },
                    ]),
                },
            ])),
            instructions: None,
            max_output_tokens: None,
            max_tool_calls: None,
            metadata: None,
            model: Some("gpt-4.1".to_string()),
            parallel_tool_calls: None,
            previous_response_id: None,
            safety_identifier: None,
            store: None,
            stream: None,
            temperature: Some(1.0),
            tool_choice: ToolChoice::Auto,
            tools: None,
            top_p: Some(1.0),
            truncation: None,
        };

        // Test serialization
        let serialized = serde_json::to_string_pretty(&request).unwrap();
        println!("Serialized RequestOfModelResponse:\n{}", serialized);

        // Test deserialization roundtrip
        let deserialized: RequestOfModelResponse = serde_json::from_str(&serialized).unwrap();

        // Verify key fields match
        assert_eq!(request.model, deserialized.model);
        assert_eq!(request.temperature, deserialized.temperature);
        assert_eq!(request.top_p, deserialized.top_p);
        assert_eq!(Some(true), deserialized.parallel_tool_calls);
        assert_eq!(request.tool_choice, deserialized.tool_choice);

        // Verify input structure
        assert!(deserialized.input.is_some());
        match deserialized.input.as_ref().unwrap() {
            Input::InputItemList(items) => {
                assert_eq!(items.len(), 1);

                match &items[0] {
                    InputItem::InputMessage { role, content, ty } => {
                        assert_eq!(role, "user");
                        assert_eq!(ty, "message");

                        match content {
                            InputMessageContent::InputItemContentList(content_items) => {
                                assert_eq!(content_items.len(), 2);

                                // Verify text content
                                match &content_items[0] {
                                    ResponseItemInputMessageContent::Text { text, ty } => {
                                        assert_eq!(text, "what is in this image?");
                                        assert_eq!(ty, "input_text");
                                    }
                                    _ => panic!("Expected Text variant for first content item"),
                                }

                                // Verify image content
                                match &content_items[1] {
                                    ResponseItemInputMessageContent::Image {
                                        image_url,
                                        ty,
                                        ..
                                    } => {
                                        assert_eq!(image_url, "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg");
                                        assert_eq!(ty, "input_image");
                                    }
                                    _ => panic!("Expected Image variant for second content item"),
                                }
                            }
                            _ => panic!("Expected InputItemContentList variant"),
                        }
                    }
                    _ => panic!("Expected InputMessage variant"),
                }
            }
            _ => panic!("Expected InputItemList variant"),
        }

        println!("✅ RequestOfModelResponse serialization and deserialization successful!");
    }

    #[test]
    fn test_create_model_response_with_text_input_and_tools() {
        // Create a RequestOfModelResponse that matches the provided JSON structure
        let request = RequestOfModelResponse {
            background: None,
            conversation: None,
            include: None,
            input: Some(Input::Text(
                "What is the weather like in Boston today?".to_string(),
            )),
            instructions: None,
            max_output_tokens: None,
            max_tool_calls: None,
            metadata: None,
            model: Some("gpt-4.1".to_string()),
            parallel_tool_calls: None,
            previous_response_id: None,
            safety_identifier: None,
            store: None,
            stream: None,
            temperature: Some(1.0),
            tool_choice: ToolChoice::Auto,
            tools: Some(vec![Tool {
                name: "get_current_weather".to_string(),
                description: "Get the current weather in a given location".to_string(),
                parameters: {
                    let mut params = JsonObject::new();
                    params.insert("type".to_string(), serde_json::json!("object"));
                    params.insert(
                        "properties".to_string(),
                        serde_json::json!({
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA"
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"]
                            }
                        }),
                    );
                    params.insert(
                        "required".to_string(),
                        serde_json::json!(["location", "unit"]),
                    );
                    params
                },
                strict: true,
                ty: "function".to_string(),
            }]),
            top_p: Some(1.0),
            truncation: None,
        };

        // Test serialization
        let serialized = serde_json::to_string_pretty(&request).unwrap();
        println!(
            "Serialized RequestOfModelResponse with text input and tools:\n{}",
            serialized
        );

        // Test deserialization roundtrip
        let deserialized: RequestOfModelResponse = serde_json::from_str(&serialized).unwrap();

        // Verify key fields match
        assert_eq!(request.model, deserialized.model);
        assert_eq!(request.temperature, deserialized.temperature);
        assert_eq!(request.top_p, deserialized.top_p);
        assert_eq!(Some(true), deserialized.parallel_tool_calls);
        assert_eq!(request.tool_choice, deserialized.tool_choice);

        // Verify input is text
        assert!(deserialized.input.is_some());
        match deserialized.input.as_ref().unwrap() {
            Input::Text(text) => {
                assert_eq!(text, "What is the weather like in Boston today?");
            }
            _ => panic!("Expected Text variant for input"),
        }

        // Verify tools
        assert!(deserialized.tools.is_some());
        let tools = deserialized.tools.as_ref().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "get_current_weather");
        assert_eq!(
            tools[0].description,
            "Get the current weather in a given location"
        );
        assert_eq!(tools[0].ty, "function");
        assert_eq!(tools[0].strict, true);

        // Verify tool parameters structure
        let params = &tools[0].parameters;
        assert!(params.contains_key("type"));
        assert!(params.contains_key("properties"));
        assert!(params.contains_key("required"));

        // Check that the properties contain location and unit
        if let Some(properties) = params.get("properties") {
            if let Some(props_obj) = properties.as_object() {
                assert!(props_obj.contains_key("location"));
                assert!(props_obj.contains_key("unit"));
            } else {
                panic!("Properties should be an object");
            }
        } else {
            panic!("Parameters should contain properties field");
        }

        println!("✅ RequestOfModelResponse with text input and tools - serialization and deserialization successful!");
    }

    #[test]
    fn test_response_object_function_call_json_deserialization() {
        // Test deserialization of the specific JSON provided by the user
        let json = r#"{
  "id": "resp_67ca09c5efe0819096d0511c92b8c890096610f474011cc0",
  "background": false,
  "object": "response",
  "created_at": 1741294021,
  "status": "completed",
  "error": null,
  "incomplete_details": null,
  "instructions": null,
  "max_output_tokens": null,
  "model": "gpt-4.1-2025-04-14",
  "output": [
    {
      "type": "function_call",
      "id": "fc_67ca09c6bedc8190a7abfec07b1a1332096610f474011cc0",
      "call_id": "call_unLAR8MvFNptuiZK6K6HCy5k",
      "name": "get_current_weather",
      "arguments": "{\"location\":\"Boston, MA\",\"unit\":\"celsius\"}",
      "status": "completed"
    }
  ],
  "parallel_tool_calls": true,
  "previous_response_id": null,
  "reasoning": {
    "effort": null,
    "summary": null
  },
  "store": true,
  "temperature": 1.0,
  "text": {
    "format": {
      "type": "text"
    }
  },
  "tool_choice": "auto",
  "tools": [
    {
      "type": "function",
      "description": "Get the current weather in a given location",
      "name": "get_current_weather",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "The city and state, e.g. San Francisco, CA"
          },
          "unit": {
            "type": "string",
            "enum": [
              "celsius",
              "fahrenheit"
            ]
          }
        },
        "required": [
          "location",
          "unit"
        ]
      },
      "strict": true
    }
  ],
  "top_p": 1.0,
  "truncation": "disabled",
  "usage": {
    "input_tokens": 291,
    "input_tokens_details": {
      "cached_tokens": 0
    },
    "output_tokens": 23,
    "output_tokens_details": {
      "reasoning_tokens": 0
    },
    "total_tokens": 314
  },
  "user": null,
  "metadata": {}
}"#;

        // Try to deserialize the JSON
        let result: Result<ResponseObject, _> = serde_json::from_str(json);

        match result {
            Ok(response) => {
                // Verify all the key fields from the provided JSON
                assert_eq!(
                    response.id,
                    "resp_67ca09c5efe0819096d0511c92b8c890096610f474011cc0"
                );
                assert_eq!(response.object, "response");
                assert_eq!(response.created_at, 1741294021);
                assert_eq!(response.status, "completed");
                assert_eq!(response.model, "gpt-4.1-2025-04-14");
                assert_eq!(response.parallel_tool_calls, true);
                assert_eq!(response.temperature, 1.0);
                assert_eq!(response.top_p, 1.0);
                assert_eq!(response.tool_choice, ToolChoice::Auto);
                assert_eq!(response.truncation, Some("disabled".to_string()));

                // Verify optional fields are None as expected
                assert!(response.error.is_none());
                assert!(response.incomplete_details.is_none());
                assert!(response.instructions.is_none());
                assert!(response.max_output_tokens.is_none());
                assert!(response.previous_response_id.is_none());

                // Verify usage statistics
                assert_eq!(response.usage.input_tokens, 291);
                assert_eq!(response.usage.output_tokens, 23);
                assert_eq!(response.usage.total_tokens, 314);
                assert_eq!(response.usage.output_tokens_details.reasoning_tokens, 0);

                // Verify output contains function call
                assert_eq!(response.output.len(), 1);
                if let ResponseOutputItem::FunctionCall {
                    id,
                    call_id,
                    name,
                    arguments,
                    status,
                    ty,
                } = &response.output[0]
                {
                    assert_eq!(id, "fc_67ca09c6bedc8190a7abfec07b1a1332096610f474011cc0");
                    assert_eq!(call_id, "call_unLAR8MvFNptuiZK6K6HCy5k");
                    assert_eq!(name, "get_current_weather");
                    assert_eq!(
                        arguments,
                        "{\"location\":\"Boston, MA\",\"unit\":\"celsius\"}"
                    );
                    assert_eq!(status, "completed");
                    assert_eq!(ty, "function_call");
                } else {
                    panic!("Expected FunctionCall variant");
                }

                // Verify tools
                assert!(response.tools.is_some());
                let tools = response.tools.as_ref().unwrap();
                assert_eq!(tools.len(), 1);
                assert_eq!(tools[0].name, "get_current_weather");
                assert_eq!(
                    tools[0].description,
                    "Get the current weather in a given location"
                );
                assert_eq!(tools[0].ty, "function");
                assert_eq!(tools[0].strict, true);

                // Verify tool parameters structure
                let params = &tools[0].parameters;
                assert!(params.contains_key("type"));
                assert!(params.contains_key("properties"));
                assert!(params.contains_key("required"));

                // Check properties and required fields
                if let Some(properties) = params.get("properties") {
                    if let Some(props_obj) = properties.as_object() {
                        assert!(props_obj.contains_key("location"));
                        assert!(props_obj.contains_key("unit"));

                        // Verify location property
                        if let Some(location) = props_obj.get("location") {
                            if let Some(loc_obj) = location.as_object() {
                                assert_eq!(
                                    loc_obj.get("type").unwrap().as_str().unwrap(),
                                    "string"
                                );
                                assert_eq!(
                                    loc_obj.get("description").unwrap().as_str().unwrap(),
                                    "The city and state, e.g. San Francisco, CA"
                                );
                            }
                        }

                        // Verify unit property
                        if let Some(unit) = props_obj.get("unit") {
                            if let Some(unit_obj) = unit.as_object() {
                                assert_eq!(
                                    unit_obj.get("type").unwrap().as_str().unwrap(),
                                    "string"
                                );
                                if let Some(enum_array) = unit_obj.get("enum") {
                                    if let Some(enum_values) = enum_array.as_array() {
                                        assert_eq!(enum_values.len(), 2);
                                        assert!(enum_values.contains(&serde_json::json!("celsius")));
                                        assert!(
                                            enum_values.contains(&serde_json::json!("fahrenheit"))
                                        );
                                    }
                                }
                            }
                        }
                    }
                }

                // Verify required fields
                if let Some(required) = params.get("required") {
                    if let Some(required_array) = required.as_array() {
                        assert_eq!(required_array.len(), 2);
                        assert!(required_array.contains(&serde_json::json!("location")));
                        assert!(required_array.contains(&serde_json::json!("unit")));
                    }
                }

                // Test serialization roundtrip
                let serialized = serde_json::to_string_pretty(&response).unwrap();
                println!("Serialized ResponseObject:\n{}", serialized);

                let re_deserialized: ResponseObject = serde_json::from_str(&serialized).unwrap();

                // Verify roundtrip preserves key data
                assert_eq!(response.id, re_deserialized.id);
                assert_eq!(response.model, re_deserialized.model);
                assert_eq!(response.status, re_deserialized.status);
                assert_eq!(response.created_at, re_deserialized.created_at);
                assert_eq!(
                    response.usage.input_tokens,
                    re_deserialized.usage.input_tokens
                );
                assert_eq!(
                    response.usage.output_tokens,
                    re_deserialized.usage.output_tokens
                );
                assert_eq!(response.output.len(), re_deserialized.output.len());
                assert_eq!(response.tools, re_deserialized.tools);

                println!(
                    "✅ Function call ResponseObject deserialization and serialization successful!"
                );
            }
            Err(e) => {
                // If deserialization fails, provide detailed error information
                println!("❌ ResponseObject deserialization failed: {}", e);
                println!("This indicates that some fields in the JSON are not present in the ResponseObject struct.");
                println!(
                    "Missing or incompatible fields likely include: reasoning, store, text, user"
                );

                // For debugging, let's try to identify what's missing
                // The struct might be missing some fields that are in the JSON
                panic!("Expected deserialization to succeed, but got error: {}", e);
            }
        }
    }
}
