use chat_prompts::PromptTemplateType;
use endpoints::models::{ListModelsResponse, Model};

/// Lists models available
pub async fn models(
    name: impl AsRef<str>,
    template_ty: PromptTemplateType,
    created: u64,
) -> ListModelsResponse {
    let model = Model {
        id: format!(
            "{name}:{template}",
            name = name.as_ref(),
            template = template_ty.to_string()
        ),
        created: created.clone(),
        object: String::from("model"),
        owned_by: String::from("Not specified"),
    };

    ListModelsResponse {
        object: String::from("list"),
        data: vec![model],
    }
}
