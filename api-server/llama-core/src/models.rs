use crate::{error::LlamaCoreError, CHAT_GRAPHS, EMBEDDING_GRAPHS};
use endpoints::models::{ListModelsResponse, Model};

/// Lists models available
pub async fn models() -> Result<ListModelsResponse, LlamaCoreError> {
    let mut models = vec![];

    {
        let chat_graphs = CHAT_GRAPHS
            .get()
            .ok_or(LlamaCoreError::Operation(String::from(
                "Fail to get the underlying value of `CHAT_GRAPHS`.",
            )))?;

        let chat_graphs = chat_graphs.lock().map_err(|e| {
            LlamaCoreError::Operation(format!("Fail to acquire the lock of `CHAT_GRAPHS`. {}", e))
        })?;

        for (name, graph) in chat_graphs.iter() {
            models.push(Model {
                id: name.clone(),
                created: graph.created.as_secs(),
                object: String::from("model"),
                owned_by: String::from("Not specified"),
            });
        }
    }

    {
        let embedding_graphs =
            EMBEDDING_GRAPHS
                .get()
                .ok_or(LlamaCoreError::Operation(String::from(
                    "Fail to get the underlying value of `EMBEDDING_GRAPHS`.",
                )))?;

        let embedding_graphs = embedding_graphs.lock().map_err(|e| {
            LlamaCoreError::Operation(format!(
                "Fail to acquire the lock of `EMBEDDING_GRAPHS`. {}",
                e
            ))
        })?;

        for (name, graph) in embedding_graphs.iter() {
            models.push(Model {
                id: name.clone(),
                created: graph.created.as_secs(),
                object: String::from("model"),
                owned_by: String::from("Not specified"),
            });
        }
    }

    Ok(ListModelsResponse {
        object: String::from("list"),
        data: models,
    })
}
