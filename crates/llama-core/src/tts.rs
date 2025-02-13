use crate::{
    error::LlamaCoreError, metadata::ggml::GgmlTtsMetadata, utils::set_tensor_data, Graph,
    MAX_BUFFER_SIZE, TTS_GRAPHS,
};
use endpoints::audio::speech::SpeechRequest;

/// Generate audio from the input text.
pub async fn create_speech(request: SpeechRequest) -> Result<Vec<u8>, LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "processing audio speech request");

    #[cfg(feature = "logging")]
    info!(target: "stdout", "Get the model instance.");

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

    match tts_graphs.contains_key(&request.model) {
        true => {
            let graph = tts_graphs.get_mut(&request.model).unwrap();

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
