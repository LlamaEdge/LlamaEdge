use endpoints::models::{ListModelsResponse, Model};

/// Lists models available
pub async fn models() -> ListModelsResponse {
    // the timestamp when the server is created
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let model = Model {
        id: String::from("Dummy-model-name"),
        created,
        object: String::from("model"),
        owned_by: String::from("Not specified"),
    };

    ListModelsResponse {
        object: String::from("list"),
        data: vec![model],
    }
}
