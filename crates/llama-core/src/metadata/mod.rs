//! Define the types for model metadata.

pub mod ggml;
pub mod piper;
#[cfg(feature = "whisper")]
#[cfg_attr(docsrs, doc(cfg(feature = "whisper")))]
pub mod whisper;

/// Base metadata trait
pub trait BaseMetadata {
    fn model_name(&self) -> &str;
    fn model_alias(&self) -> &str;
}
