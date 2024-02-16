use crate::{error::LlamaCoreError, get_graph};
use endpoints::models::{ListModelsResponse, Model};

/// Lists models available
pub async fn models() -> Result<ListModelsResponse, LlamaCoreError> {
    let graph = get_graph()?;
    let created = graph.created.as_secs();
    let id = graph.name.clone();

    let model = Model {
        id,
        created,
        object: String::from("model"),
        owned_by: String::from("Not specified"),
    };

    Ok(ListModelsResponse {
        object: String::from("list"),
        data: vec![model],
    })
}
