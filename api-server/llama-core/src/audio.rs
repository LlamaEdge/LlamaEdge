//! Define APIs for audio generation, transcription, and translation.

use crate::{error::LlamaCoreError, MAX_BUFFER_SIZE};
use endpoints::{
    audio::transcription::{TranscriptionObject, TranscriptionRequest},
    files::FileObject,
};
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
    let graph = match GRAPH.get() {
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

    #[cfg(feature = "logging")]
    info!(target: "stdout", "Initialize the execution context.");
    graph.init_execution_context()?;

    // set the input tensor
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Feed the audio data to the model.");
    graph.set_input(
        0,
        wasmedge_wasi_nn::TensorType::F32,
        &[1, waveform.len()],
        &waveform,
    )?;

    // compute the graph
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Transcribe audio to text.");
    graph.compute()?;

    // get the output tensor
    #[cfg(feature = "logging")]
    info!(target: "stdout", "[INFO] Retrieve the transcription data.");
    let mut output_buffer = vec![0u8; MAX_BUFFER_SIZE];
    let size = graph.get_output(0, &mut output_buffer)?;
    unsafe {
        output_buffer.set_len(size);
    }
    info!(target: "stdout", "Output buffer size: {}", size);

    // decode the output buffer
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Decode the transcription data to plain text.");
    let text = match std::str::from_utf8(&output_buffer[..]) {
        Ok(output) => output.to_string(),
        Err(e) => {
            let err_msg = format!(
                "Failed to decode the gerated buffer to a utf-8 string. {}",
                e
            );

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", &err_msg);

            return Err(LlamaCoreError::Operation(err_msg));
        }
    };

    let obj = TranscriptionObject { text };

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
