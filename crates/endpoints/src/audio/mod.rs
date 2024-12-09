//! Define types for turning audio into text or text into audio.

pub mod speech;
#[cfg(feature = "whisper")]
#[cfg_attr(docsrs, doc(cfg(feature = "whisper")))]
pub mod transcription;
#[cfg(feature = "whisper")]
#[cfg_attr(docsrs, doc(cfg(feature = "whisper")))]
pub mod translation;
