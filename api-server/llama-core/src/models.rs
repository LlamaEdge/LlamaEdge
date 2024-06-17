//! Define APIs for querying models.

use crate::{error::LlamaCoreError, CHAT_GRAPHS, EMBEDDING_GRAPHS};
use endpoints::models::{ListModelsResponse, Model};

/// Lists models available
pub async fn models() -> Result<ListModelsResponse, LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "llama-core", "List models");

    let mut models = vec![];

    {
        if let Some(chat_graphs) = CHAT_GRAPHS.get() {
            for (name, (created, _graph)) in chat_graphs.iter() {
                models.push(Model {
                    id: name.clone(),
                    created: created.as_secs(),
                    object: String::from("model"),
                    owned_by: String::from("Not specified"),
                });
            }
        }
    }

    {
        if let Some(embedding_graphs) = EMBEDDING_GRAPHS.get() {
            let embedding_graphs = embedding_graphs.lock().map_err(|e| {
                LlamaCoreError::Operation(format!(
                    "Fail to acquire the lock of `EMBEDDING_GRAPHS`. {}",
                    e
                ))
            })?;

            if !embedding_graphs.is_empty() {
                for (name, graph) in embedding_graphs.iter() {
                    models.push(Model {
                        id: name.clone(),
                        created: graph.created.as_secs(),
                        object: String::from("model"),
                        owned_by: String::from("Not specified"),
                    });
                }
            }
        }
    }

    Ok(ListModelsResponse {
        object: String::from("list"),
        data: models,
    })
}
