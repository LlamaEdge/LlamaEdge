use crate::error::LlamaCoreError;
use endpoints::embeddings::{EmbeddingObject, EmbeddingRequest};

pub async fn embeddings(
    embedding_request: &mut EmbeddingRequest,
) -> Result<EmbeddingObject, LlamaCoreError> {
    // ! debug
    println!("embedding_request: {:?}", embedding_request);

    let fake_embedding_object = EmbeddingObject {
        index: 0,
        object: String::from("embedding"),
        embedding: vec![0.1, 0.2, 0.3, 0.4, 0.5],
    };

    Ok(fake_embedding_object)
}
