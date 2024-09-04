//! Define APIs for audio generation, transcription, and translation.

use crate::{error::LlamaCoreError, utils::set_tensor_data, AUDIO_GRAPH, MAX_BUFFER_SIZE};
use endpoints::audio::transcription::{TranscriptionObject, TranscriptionRequest};
use std::path::Path;

/// Transcribe audio into the input language.
pub async fn audio_transcriptions(
    request: TranscriptionRequest,
) -> Result<TranscriptionObject, LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "processing audio transcription request");

    let path = Path::new("archives")
        .join(&request.file.id)
        .join(&request.file.filename);

    #[cfg(feature = "logging")]
    info!(target: "stdout", "audio file path: {:?}", &path);

    // load the audio waveform
    let wav_buf = load_audio_waveform(path)?;

    #[cfg(feature = "logging")]
    info!(target: "stdout", "read input tensor, size in bytes: {}", wav_buf.len());

    #[cfg(feature = "logging")]
    info!(target: "stdout", "Get the model instance.");
    let graph = match AUDIO_GRAPH.get() {
        Some(graph) => graph,
        None => {
            let err_msg = "The GRAPH is not initialized.";

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", &err_msg);

            return Err(LlamaCoreError::Operation(err_msg.to_owned()));
        }
    };

    let mut graph = match graph.lock() {
        Ok(graph) => graph,
        Err(e) => {
            let err_msg = format!("Failed to lock the graph. {}", e);

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", &err_msg);

            return Err(LlamaCoreError::Operation(err_msg));
        }
    };

    // set the input tensor
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Feed the audio data to the model.");
    set_tensor_data(&mut graph, 0, &wav_buf, [1, wav_buf.len()])?;

    // compute the graph
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Transcribe audio to text.");
    if let Err(e) = graph.compute() {
        let err_msg = format!("Failed to compute the graph. {}", e);

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", &err_msg);

        return Err(LlamaCoreError::Operation(err_msg));
    }

    // get the output tensor
    #[cfg(feature = "logging")]
    info!(target: "stdout", "[INFO] Retrieve the transcription data.");

    // Retrieve the output.
    let mut output_buffer = vec![0u8; MAX_BUFFER_SIZE];
    let output_size = graph.get_output(0, &mut output_buffer).map_err(|e| {
        let err_msg = format!("Failed to get the output tensor. {}", e);

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", &err_msg);

        LlamaCoreError::Operation(err_msg)
    })?;

    #[cfg(feature = "logging")]
    info!(target: "stdout", "Output buffer size: {}", output_size);

    // decode the output buffer
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Decode the transcription data to plain text.");

    let text = std::str::from_utf8(&output_buffer[..output_size]).map_err(|e| {
        let err_msg = format!(
            "Failed to decode the gerated buffer to a utf-8 string. {}",
            e
        );

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", &err_msg);

        LlamaCoreError::Operation(err_msg)
    })?;

    let obj = TranscriptionObject {
        text: text.trim().to_owned(),
    };

    #[cfg(feature = "logging")]
    info!(target: "stdout", "End of the chat completion.");

    Ok(obj)
}

fn load_audio_waveform(filename: impl AsRef<std::path::Path>) -> Result<Vec<u8>, LlamaCoreError> {
    std::fs::read(filename)
        .map_err(|e| {
            let err_msg = format!("Failed to read the input tensor. {}", e);

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", &err_msg);

            LlamaCoreError::Operation(err_msg)
        })
        .map_err(|e| LlamaCoreError::Operation(e.to_string()))
}
