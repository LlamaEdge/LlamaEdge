//! Define APIs for audio generation, transcription, and translation.

use crate::{
    error::LlamaCoreError,
    utils::{get_output_buffer, set_tensor_data},
    AUDIO_GRAPH, OUTPUT_TENSOR,
};
use endpoints::audio::transcription::{TranscriptionObject, TranscriptionRequest};
use hound::{self, SampleFormat};
use std::path::Path;

pub async fn audio_transcriptions(
    request: TranscriptionRequest,
) -> Result<TranscriptionObject, LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "processing audio transcription request");

    let path = Path::new("archives")
        .join(&request.file.id)
        .join(&request.file.filename);

    // load the audio waveform
    let (waveform, sample_rate) = load_audio_waveform(path)?;
    assert_eq!(sample_rate, 16000, "The audio sample rate must be 16k.");

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
    set_tensor_data(&mut graph, 0, &waveform)?;

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
    let output_buffer = get_output_buffer(&graph, OUTPUT_TENSOR)?;

    // decode the output buffer
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Decode the transcription data to plain text.");
    let text = std::str::from_utf8(&output_buffer[..]).map_err(|e| {
        let err_msg = format!(
            "Failed to decode the gerated buffer to a utf-8 string. {}",
            e
        );

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", &err_msg);

        LlamaCoreError::Operation(err_msg)
    })?;

    let obj = TranscriptionObject {
        text: text.to_owned(),
    };

    #[cfg(feature = "logging")]
    info!(target: "stdout", "End of the chat completion.");

    Ok(obj)
}

fn load_audio_waveform(
    filename: impl AsRef<std::path::Path>,
) -> Result<(Vec<f32>, usize), LlamaCoreError> {
    let reader =
        hound::WavReader::open(filename).map_err(|e| LlamaCoreError::Operation(e.to_string()))?;
    let spec = reader.spec();

    // let duration = reader.duration() as usize;
    let channels = spec.channels as usize;
    let sample_rate = spec.sample_rate as usize;
    // let bits_per_sample = spec.bits_per_sample;
    let sample_format = spec.sample_format;

    assert_eq!(sample_rate, 16000, "The audio sample rate must be 16k.");
    assert_eq!(channels, 1, "The audio must be single-channel.");

    let max_int_val = 2_u32.pow(spec.bits_per_sample as u32 - 1) - 1;

    let floats = match sample_format {
        SampleFormat::Float => reader
            .into_samples::<f32>()
            .collect::<hound::Result<_>>()
            .map_err(|e| LlamaCoreError::Operation(e.to_string()))?,
        SampleFormat::Int => reader
            .into_samples::<i32>()
            .map(|s| s.map(|s| s as f32 / max_int_val as f32))
            .collect::<hound::Result<_>>()
            .map_err(|e| LlamaCoreError::Operation(e.to_string()))?,
    };

    Ok((floats, sample_rate))
}
