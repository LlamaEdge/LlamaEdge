pub mod ggml;
pub mod piper;
#[cfg(feature = "whisper")]
#[cfg_attr(docsrs, doc(cfg(feature = "whisper")))]
pub mod whisper;

pub trait BaseMetadata {
    fn model_name(&self) -> &str;
    fn model_alias(&self) -> &str;
}
