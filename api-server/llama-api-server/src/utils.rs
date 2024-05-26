use crate::LOG_TARGET;
use serde::{Deserialize, Serialize};

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

pub(crate) fn info(value: serde_json::Value) {
    let record = NewLogRecord::new(LogLevel::Info, None, value);
    let record = serde_json::to_string(&record).unwrap();
    log!(target: LOG_TARGET, log::Level::Info, "{}", record.to_string());
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct NewLogRecord {
    /// log level
    level: LogLevel,
    /// User id
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<String>,
    /// log message
    #[serde(flatten)]
    message: serde_json::Value,
    /// time stamp
    timestamp: String,
}

impl NewLogRecord {
    pub(crate) fn new(level: LogLevel, user: Option<String>, message: serde_json::Value) -> Self {
        Self {
            level,
            user,
            message,
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }
}
