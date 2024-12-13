//! Define APIs for audio generation, transcription, and translation.

#[cfg(feature = "whisper")]
use crate::AUDIO_GRAPH;
use crate::{error::LlamaCoreError, utils::set_tensor_data, MAX_BUFFER_SIZE, PIPER_GRAPH};
use endpoints::audio::speech::SpeechRequest;
#[cfg(feature = "whisper")]
use endpoints::audio::{
    transcription::{TranscriptionObject, TranscriptionRequest},
    translation::{TranslationObject, TranslationRequest},
};
#[cfg(feature = "whisper")]
use std::path::Path;

/// Transcribe audio into the input language.
#[cfg(feature = "whisper")]
#[cfg_attr(docsrs, doc(cfg(feature = "whisper")))]
pub async fn audio_transcriptions(
    request: TranscriptionRequest,
) -> Result<TranscriptionObject, LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "processing audio transcription request");

    let graph = match AUDIO_GRAPH.get() {
        Some(graph) => graph,
        None => {
            let err_msg = "The AUDIO_GRAPH is not initialized.";

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

    // check if the model metadata should be updated
    {
        let mut should_update = false;

        #[cfg(feature = "logging")]
        info!(target: "stdout", "current metadata: {:?}", &graph.metadata);

        #[cfg(feature = "logging")]
        info!(target: "stdout", "Check model metadata.");

        // check `translate` field
        if graph.metadata.translate {
            // update the metadata
            graph.metadata.translate = false;

            if !should_update {
                should_update = true;
            }
        }

        // check `language` field
        if let Some(language) = &request.language {
            if *language != graph.metadata.language {
                // update the metadata
                graph.metadata.language = language.clone();

                if !should_update {
                    should_update = true;
                }
            }
        }

        // check `detect_language` field
        if let Some(detect_language) = &request.detect_language {
            if *detect_language != graph.metadata.detect_language {
                // update the metadata
                graph.metadata.detect_language = *detect_language;

                if !should_update {
                    should_update = true;
                }
            }
        }

        // check `offset_time` field
        if let Some(offset_time) = &request.offset_time {
            if *offset_time != graph.metadata.offset_time {
                // update the metadata
                graph.metadata.offset_time = *offset_time;

                if !should_update {
                    should_update = true;
                }
            }
        }

        // check `duration` field
        if let Some(duration) = &request.duration {
            if *duration != graph.metadata.duration {
                // update the metadata
                graph.metadata.duration = *duration;

                if !should_update {
                    should_update = true;
                }
            }
        }

        // check `max_context` field
        if let Some(max_context) = &request.max_context {
            if *max_context != graph.metadata.max_context {
                // update the metadata
                graph.metadata.max_context = *max_context;

                if !should_update {
                    should_update = true;
                }
            }
        }

        // check `max_len` field
        if let Some(max_len) = &request.max_len {
            if *max_len != graph.metadata.max_len {
                // update the metadata
                graph.metadata.max_len = *max_len;

                if !should_update {
                    should_update = true;
                }
            }
        }

        // check `temperature` field
        if let Some(temperature) = &request.temperature {
            if *temperature != graph.metadata.temperature {
                // update the metadata
                graph.metadata.temperature = *temperature;

                if !should_update {
                    should_update = true;
                }
            }
        }

        // check `split_on_word` field
        if let Some(split_on_word) = &request.split_on_word {
            if *split_on_word != graph.metadata.split_on_word {
                // update the metadata
                graph.metadata.split_on_word = *split_on_word;

                if !should_update {
                    should_update = true;
                }
            }
        }

        // check `prompt` field
        if let Some(prompt) = &request.prompt {
            if *prompt != graph.metadata.prompt {
                // update the metadata
                graph.metadata.prompt = prompt.clone();

                if !should_update {
                    should_update = true;
                }
            }
        }

        #[cfg(feature = "logging")]
        info!(target: "stdout", "metadata: {:?}", &graph.metadata);

        if should_update {
            #[cfg(feature = "logging")]
            info!(target: "stdout", "Set the metadata to the model.");

            match serde_json::to_string(&graph.metadata) {
                Ok(config) => {
                    // update metadata
                    set_tensor_data(&mut graph, 1, config.as_bytes(), [1])?;

                    #[cfg(feature = "logging")]
                    info!(target: "stdout", "metadata updated");
                }
                Err(e) => {
                    let err_msg = format!("Fail to serialize metadata to a JSON string. {}", e);

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", &err_msg);

                    return Err(LlamaCoreError::Operation(err_msg));
                }
            };
        }
    }

    let path = Path::new("archives")
        .join(&request.file.id)
        .join(&request.file.filename);

    #[cfg(feature = "logging")]
    info!(target: "stdout", "audio file path: {:?}", &path);

    // load the audio waveform
    let wav_buf = load_audio_waveform(path)?;

    #[cfg(feature = "logging")]
    info!(target: "stdout", "read input tensor, size in bytes: {}", wav_buf.len());

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

    #[cfg(feature = "logging")]
    info!(target: "stdout", "raw transcription text:\n{}", &text);

    let obj = TranscriptionObject {
        text: text.trim().to_owned(),
    };

    #[cfg(feature = "logging")]
    info!(target: "stdout", "End of the audio transcription.");

    Ok(obj)
}

#[cfg(feature = "whisper")]
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

fn _remove_blank_audio(input: &str) -> String {
    let blank_audio_marker = "[BLANK_AUDIO]";

    // Split the input string by newline and filter out segments containing [BLANK_AUDIO]
    let filtered_segments: Vec<&str> = input
        .lines()
        .filter(|segment| !segment.contains(blank_audio_marker))
        .collect();

    // Rejoin the filtered segments with newline
    filtered_segments.join("\n")
}

/// Translate audio into the target language
#[cfg(feature = "whisper")]
#[cfg_attr(docsrs, doc(cfg(feature = "whisper")))]
pub async fn audio_translations(
    request: TranslationRequest,
) -> Result<TranslationObject, LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "processing audio translation request");

    let graph = match AUDIO_GRAPH.get() {
        Some(graph) => graph,
        None => {
            let err_msg = "The AUDIO_GRAPH is not initialized.";

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

    // check if the model metadata should be updated
    {
        let mut should_update = false;

        #[cfg(feature = "logging")]
        info!(target: "stdout", "current metadata: {:?}", &graph.metadata);

        #[cfg(feature = "logging")]
        info!(target: "stdout", "Check model metadata.");

        // check `translate` field
        if !graph.metadata.translate {
            graph.metadata.translate = true;

            if !should_update {
                should_update = true;
            }
        }

        // check `language` field
        if let Some(language) = &request.language {
            if *language != graph.metadata.language {
                graph.metadata.language = language.clone();

                if !should_update {
                    should_update = true;
                }
            }
        }

        // check `detect_language` field
        if let Some(detect_language) = &request.detect_language {
            if *detect_language != graph.metadata.detect_language {
                graph.metadata.detect_language = *detect_language;

                if !should_update {
                    should_update = true;
                }
            }
        }

        // check `offset_time` field
        if let Some(offset_time) = &request.offset_time {
            if *offset_time != graph.metadata.offset_time {
                // update the metadata
                graph.metadata.offset_time = *offset_time;

                if !should_update {
                    should_update = true;
                }
            }
        }

        // check `duration` field
        if let Some(duration) = &request.duration {
            if *duration != graph.metadata.duration {
                graph.metadata.duration = *duration;

                if !should_update {
                    should_update = true;
                }
            }
        }

        // check `max_context` field
        if let Some(max_context) = &request.max_context {
            if *max_context != graph.metadata.max_context {
                graph.metadata.max_context = *max_context;

                if !should_update {
                    should_update = true;
                }
            }
        }

        // check `max_len` field
        if let Some(max_len) = &request.max_len {
            if *max_len != graph.metadata.max_len {
                graph.metadata.max_len = *max_len;

                if !should_update {
                    should_update = true;
                }
            }
        }

        // check `temperature` field
        if let Some(temperature) = &request.temperature {
            if *temperature != graph.metadata.temperature {
                graph.metadata.temperature = *temperature;

                if !should_update {
                    should_update = true;
                }
            }
        }

        // check `split_on_word` field
        if let Some(split_on_word) = &request.split_on_word {
            if *split_on_word != graph.metadata.split_on_word {
                graph.metadata.split_on_word = *split_on_word;

                if !should_update {
                    should_update = true;
                }
            }
        }

        // check `prompt` field
        if let Some(prompt) = &request.prompt {
            if *prompt != graph.metadata.prompt {
                graph.metadata.prompt = prompt.clone();

                if !should_update {
                    should_update = true;
                }
            }
        }

        #[cfg(feature = "logging")]
        info!(target: "stdout", "metadata: {:?}", &graph.metadata);

        if should_update {
            #[cfg(feature = "logging")]
            info!(target: "stdout", "Set the metadata to the model.");

            match serde_json::to_string(&graph.metadata) {
                Ok(config) => {
                    // update metadata
                    set_tensor_data(&mut graph, 1, config.as_bytes(), [1])?;
                }
                Err(e) => {
                    let err_msg = format!("Fail to serialize metadata to a JSON string. {}", e);

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", &err_msg);

                    return Err(LlamaCoreError::Operation(err_msg));
                }
            };
        }
    }

    let path = Path::new("archives")
        .join(&request.file.id)
        .join(&request.file.filename);

    #[cfg(feature = "logging")]
    info!(target: "stdout", "audio file path: {:?}", &path);

    // load the audio waveform
    let wav_buf = load_audio_waveform(path)?;

    #[cfg(feature = "logging")]
    info!(target: "stdout", "read input tensor, size in bytes: {}", wav_buf.len());

    // set the input tensor
    #[cfg(feature = "logging")]
    info!(target: "stdout", "feed the audio data to the model.");
    set_tensor_data(&mut graph, 0, &wav_buf, [1, wav_buf.len()])?;

    // compute the graph
    #[cfg(feature = "logging")]
    info!(target: "stdout", "translate audio to text.");
    if let Err(e) = graph.compute() {
        let err_msg = format!("Failed to compute the graph. {}", e);

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", &err_msg);

        return Err(LlamaCoreError::Operation(err_msg));
    }

    // get the output tensor
    #[cfg(feature = "logging")]
    info!(target: "stdout", "[INFO] retrieve the translation data.");

    // Retrieve the output.
    let mut output_buffer = vec![0u8; MAX_BUFFER_SIZE];
    let output_size = graph.get_output(0, &mut output_buffer).map_err(|e| {
        let err_msg = format!("Failed to get the output tensor. {}", e);

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", &err_msg);

        LlamaCoreError::Operation(err_msg)
    })?;

    #[cfg(feature = "logging")]
    info!(target: "stdout", "output buffer size: {}", output_size);

    // decode the output buffer
    #[cfg(feature = "logging")]
    info!(target: "stdout", "decode the translation data to plain text.");

    let text = std::str::from_utf8(&output_buffer[..output_size]).map_err(|e| {
        let err_msg = format!(
            "Failed to decode the gerated buffer to a utf-8 string. {}",
            e
        );

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", &err_msg);

        LlamaCoreError::Operation(err_msg)
    })?;

    #[cfg(feature = "logging")]
    info!(target: "stdout", "raw translation text:\n{}", &text);

    let obj = TranslationObject {
        text: text.trim().to_owned(),
    };

    #[cfg(feature = "logging")]
    info!(target: "stdout", "End of the audio translation.");

    Ok(obj)
}

/// Generate audio from the input text.
pub async fn create_speech(request: SpeechRequest) -> Result<Vec<u8>, LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "processing audio speech request");

    #[cfg(feature = "logging")]
    info!(target: "stdout", "Get the model instance.");
    let graph = match PIPER_GRAPH.get() {
        Some(graph) => graph,
        None => {
            let err_msg = "The PIPER_GRAPH is not initialized.";

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
    info!(target: "stdout", "Feed the text to the model.");
    set_tensor_data(&mut graph, 0, request.input.as_bytes(), [1])?;

    // compute the graph
    #[cfg(feature = "logging")]
    info!(target: "stdout", "create audio.");
    if let Err(e) = graph.compute() {
        let err_msg = format!("Failed to compute the graph. {}", e);

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", &err_msg);

        return Err(LlamaCoreError::Operation(err_msg));
    }

    // get the output tensor
    #[cfg(feature = "logging")]
    info!(target: "stdout", "[INFO] Retrieve the audio.");

    let mut output_buffer = vec![0u8; MAX_BUFFER_SIZE];
    let output_size = graph.get_output(0, &mut output_buffer).map_err(|e| {
        let err_msg = format!("Failed to get the output tensor. {}", e);

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", &err_msg);

        LlamaCoreError::Operation(err_msg)
    })?;

    #[cfg(feature = "logging")]
    info!(target: "stdout", "Output buffer size: {}", output_size);

    Ok(output_buffer)
}
