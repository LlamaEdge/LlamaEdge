use serde::{Deserialize, Serialize};

pub(crate) fn log(msg: impl std::fmt::Display) {
    println!("{}", msg);
}

pub(crate) fn gen_chat_id() -> String {
    format!("chatcmpl-{}", uuid::Uuid::new_v4())
}

#[derive(
    Debug,
    Copy,
    Clone,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    clap::ValueEnum,
    Serialize,
    Deserialize,
    Default,
)]
#[serde(rename_all = "lowercase")]
pub(crate) enum LogLevel {
    /// Describes messages about the values of variables and the flow of
    /// control within a program.
    Trace,

    /// Describes messages likely to be of interest to someone debugging a
    /// program.
    Debug,

    /// Describes messages likely to be of interest to someone monitoring a
    /// program.
    #[default]
    Info,

    /// Describes messages indicating hazardous situations.
    Warn,

    /// Describes messages indicating serious errors.
    Error,

    /// Describes messages indicating fatal errors.
    Critical,
}
impl From<LogLevel> for log::LevelFilter {
    fn from(level: LogLevel) -> Self {
        match level {
            LogLevel::Trace => log::LevelFilter::Trace,
            LogLevel::Debug => log::LevelFilter::Debug,
            LogLevel::Info => log::LevelFilter::Info,
            LogLevel::Warn => log::LevelFilter::Warn,
            LogLevel::Error => log::LevelFilter::Error,
            LogLevel::Critical => log::LevelFilter::Error,
        }
    }
}
impl std::fmt::Display for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            LogLevel::Trace => write!(f, "trace"),
            LogLevel::Debug => write!(f, "debug"),
            LogLevel::Info => write!(f, "info"),
            LogLevel::Warn => write!(f, "warn"),
            LogLevel::Error => write!(f, "error"),
            LogLevel::Critical => write!(f, "critical"),
        }
    }
}
impl std::str::FromStr for LogLevel {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "trace" => Ok(LogLevel::Trace),
            "debug" => Ok(LogLevel::Debug),
            "info" => Ok(LogLevel::Info),
            "warn" => Ok(LogLevel::Warn),
            "error" => Ok(LogLevel::Error),
            "critical" => Ok(LogLevel::Critical),
            _ => Err(format!("Invalid log level: {}", s)),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct LogRecord {
    /// log level
    level: LogLevel,
    /// User id
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<String>,
    /// log message
    message: String,
    /// time stamp
    timestamp: String,
}

impl LogRecord {
    pub fn new(level: LogLevel, user: Option<String>, message: impl Into<String>) -> Self {
        Self {
            level,
            user,
            message: message.into(),
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct RequestLogRecord {
    /// log level
    pub level: LogLevel,
    /// User id
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    /// remote ip address
    pub remote_addr: String,
    /// local ip address
    pub local_addr: String,
    /// request method
    /// e.g. GET, POST, PUT, DELETE
    pub method: String,
    /// request path
    /// e.g. /v1/chat/completions
    pub path: String,
    /// http version
    /// e.g. HTTP/1.1
    pub version: String,
    /// size of request body in bytes
    pub size: usize,
    /// time stamp
    pub timestamp: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ChatResponseLogRecord {
    /// log level
    pub level: LogLevel,
    /// User id
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    /// http version
    /// e.g. HTTP/1.1
    pub version: String,
    /// lower bound size of response body in bytes
    pub size: u64,
    /// response status code
    pub status: u16,
    /// Check if status is within 100-199
    pub is_informational: bool,
    /// Check if status is within 200-299
    pub is_success: bool,
    /// Check if status is within 300-399
    pub is_redirection: bool,
    /// Check if status is within 400-499
    pub is_client_error: bool,
    /// Check if status is within 500-599
    pub is_server_error: bool,
    /// time stamp
    pub timestamp: String,
}
