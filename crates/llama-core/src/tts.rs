use crate::{
    error::LlamaCoreError,
    metadata::ggml::GgmlTtsMetadata,
    running_mode,
    utils::{set_tensor_data, set_tensor_data_u8},
    Graph, RunningMode, MAX_BUFFER_SIZE, TTS_GRAPHS,
};
use endpoints::audio::speech::SpeechRequest;

/// Generate audio from the input text.
pub async fn create_speech(request: SpeechRequest) -> Result<Vec<u8>, LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "processing audio speech request");

    let running_mode = running_mode()?;
    if !running_mode.contains(RunningMode::TTS) {
        let err_msg = "Generating audio speech is only supported in the tts mode.";

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", err_msg);

        return Err(LlamaCoreError::Operation(err_msg.into()));
    }

    let model_name = &request.model;

    let res = {
        let tts_graphs = match TTS_GRAPHS.get() {
            Some(tts_graphs) => tts_graphs,
            None => {
                let err_msg = "Fail to get the underlying value of `TTS_GRAPHS`.";

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", &err_msg);

                return Err(LlamaCoreError::Operation(err_msg.into()));
            }
        };

        let mut tts_graphs = tts_graphs.lock().map_err(|e| {
            let err_msg = format!("Fail to acquire the lock of `TTS_GRAPHS`. {}", e);

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", &err_msg);

            LlamaCoreError::Operation(err_msg)
        })?;

        match tts_graphs.contains_key(model_name) {
            true => {
                let graph = tts_graphs.get_mut(model_name).unwrap();

                compute_by_graph(graph, &request)
            }
            false => match tts_graphs.iter_mut().next() {
                Some((_name, graph)) => compute_by_graph(graph, &request),
                None => {
                    let err_msg = "There is no model available in the chat graphs.";

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", &err_msg);

                    Err(LlamaCoreError::Operation(err_msg.into()))
                }
            },
        }
    };

    #[cfg(feature = "logging")]
    info!(target: "stdout", "Reset the model metadata");

    // reset the model metadata
    reset_model_metadata(Some(model_name))?;

    res
}

fn compute_by_graph(
    graph: &mut Graph<GgmlTtsMetadata>,
    request: &SpeechRequest,
) -> Result<Vec<u8>, LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Input text: {}", &request.input);

    // set the input tensor
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Feed the text to the model.");
    set_tensor_data(graph, 0, request.input.as_bytes(), [1])?;

    // compute the graph
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Generate audio");
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

/// Get a copy of the metadata of the model.
fn get_model_metadata(model_name: Option<&String>) -> Result<GgmlTtsMetadata, LlamaCoreError> {
    let tts_graphs = match TTS_GRAPHS.get() {
        Some(tts_graphs) => tts_graphs,
        None => {
            let err_msg = "Fail to get the underlying value of `TTS_GRAPHS`.";

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", err_msg);

            return Err(LlamaCoreError::Operation(err_msg.into()));
        }
    };

    let tts_graphs = tts_graphs.lock().map_err(|e| {
        let err_msg = format!("Fail to acquire the lock of `TTS_GRAPHS`. {}", e);

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", &err_msg);

        LlamaCoreError::Operation(err_msg)
    })?;

    match model_name {
        Some(model_name) => match tts_graphs.contains_key(model_name) {
            true => {
                let graph = tts_graphs.get(model_name).unwrap();
                Ok(graph.metadata.clone())
            }
            false => match tts_graphs.iter().next() {
                Some((_, graph)) => Ok(graph.metadata.clone()),
                None => {
                    let err_msg = "There is no model available in the tts graphs.";

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", &err_msg);

                    Err(LlamaCoreError::Operation(err_msg.into()))
                }
            },
        },
        None => match tts_graphs.iter().next() {
            Some((_, graph)) => Ok(graph.metadata.clone()),
            None => {
                let err_msg = "There is no model available in the tts graphs.";

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", err_msg);

                Err(LlamaCoreError::Operation(err_msg.into()))
            }
        },
    }
}

fn update_model_metadata(
    model_name: Option<&String>,
    metadata: &GgmlTtsMetadata,
) -> Result<(), LlamaCoreError> {
    let config = match serde_json::to_string(metadata) {
        Ok(config) => config,
        Err(e) => {
            let err_msg = format!("Fail to serialize metadata to a JSON string. {}", e);

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", &err_msg);

            return Err(LlamaCoreError::Operation(err_msg));
        }
    };

    let tts_graphs = match TTS_GRAPHS.get() {
        Some(tts_graphs) => tts_graphs,
        None => {
            let err_msg = "Fail to get the underlying value of `TTS_GRAPHS`.";

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", err_msg);

            return Err(LlamaCoreError::Operation(err_msg.into()));
        }
    };

    let mut tts_graphs = tts_graphs.lock().map_err(|e| {
        let err_msg = format!("Fail to acquire the lock of `TTS_GRAPHS`. Reason: {}", e);

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", &err_msg);

        LlamaCoreError::Operation(err_msg)
    })?;

    match model_name {
        Some(model_name) => {
            match tts_graphs.contains_key(model_name) {
                true => {
                    let graph = tts_graphs.get_mut(model_name).unwrap();
                    // update metadata
                    set_tensor_data_u8(graph, 1, config.as_bytes())
                }
                false => match tts_graphs.iter_mut().next() {
                    Some((_, graph)) => {
                        // update metadata
                        set_tensor_data_u8(graph, 1, config.as_bytes())
                    }
                    None => {
                        let err_msg = "There is no model available in the tts graphs.";

                        #[cfg(feature = "logging")]
                        error!(target: "stdout", "{}", &err_msg);

                        Err(LlamaCoreError::Operation(err_msg.into()))
                    }
                },
            }
        }
        None => {
            match tts_graphs.iter_mut().next() {
                Some((_, graph)) => {
                    // update metadata
                    set_tensor_data_u8(graph, 1, config.as_bytes())
                }
                None => {
                    let err_msg = "There is no model available in the tts graphs.";

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", err_msg);

                    Err(LlamaCoreError::Operation(err_msg.into()))
                }
            }
        }
    }
}

fn reset_model_metadata(model_name: Option<&String>) -> Result<(), LlamaCoreError> {
    // get metadata
    let metadata = get_model_metadata(model_name)?;

    // update model with the original metadata
    update_model_metadata(model_name, &metadata)
}
