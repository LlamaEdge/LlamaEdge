use hyper::{
    service::{make_service_fn, service_fn},
    Body, Request, Response, Server,
};
use once_cell::sync::OnceCell;
use prompt::PromptTemplateType;
use std::net::SocketAddr;
use std::str::FromStr;

mod backend;
use backend::ggml;

mod error;
use error::ServerError;

type Error = Box<dyn std::error::Error + Send + Sync + 'static>;

const DEFAULT_SOCKET_ADDRESS: &str = "0.0.0.0:8080";
const DEFAULT_CTX_SIZE: &str = "2048";

static CTX_SIZE: OnceCell<usize> = OnceCell::new();

use clap::{Arg, Command};

#[derive(Clone, Debug)]
pub struct AppState {
    pub state_thing: String,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), ServerError> {
    let matches = Command::new("Llama API Server")
        .arg(
            Arg::new("model_alias")
                .short('m')
                .long("model-alias")
                .value_name("ALIAS")
                .help("Sets the model alias")
                .required(true),
        )
        .arg(
            Arg::new("prompt_template")
                .short('p')
                .long("prompt-template")
                .value_parser([
                    "llama-2-chat",
                    "codellama-instruct",
                    "mistral-instruct-v0.1",
                ])
                .value_name("TEMPLATE")
                .help("Sets the prompt template.")
                .required(true),
        )
        .arg(
            Arg::new("socket_addr")
                .short('s')
                .long("socket-addr")
                .value_name("IP:PORT")
                .help("Sets the socket address")
                .default_value(DEFAULT_SOCKET_ADDRESS),
        )
        .arg(
            Arg::new("ctx_size")
                .short('c')
                .long("ctx-size")
                .value_parser(clap::value_parser!(u32))
                .value_name("CTX_SIZE")
                .help("Sets the prompt context size")
                .default_value(DEFAULT_CTX_SIZE),
        )
        .get_matches();

    // model alias
    let model_name = matches
        .get_one::<String>("model_alias")
        .unwrap()
        .to_string();
    println!("[SERVER] Model alias: {alias}", alias = &model_name);
    let ref_model_name = std::sync::Arc::new(model_name);

    // type of prompt template
    let prompt_template = matches
        .get_one::<String>("prompt_template")
        .unwrap()
        .to_string();
    let template_ty = match PromptTemplateType::from_str(&prompt_template) {
        Ok(template) => template,
        Err(e) => {
            return Err(ServerError::InvalidPromptTemplateType(e.to_string()));
        }
    };
    println!("[SERVER] Prompt template: {ty:?}", ty = &template_ty);
    let ref_template_ty = std::sync::Arc::new(template_ty);

    // socket address
    let socket_addr = matches
        .get_one::<String>("socket_addr")
        .unwrap()
        .to_string();
    let addr: SocketAddr = match socket_addr.parse() {
        Ok(addr) => addr,
        Err(e) => {
            return Err(ServerError::SocketAddr(e.to_string()));
        }
    };
    println!(
        "[SERVER] Socket address: {socket_addr}",
        socket_addr = socket_addr
    );

    // prompt context size
    let ctx_size = matches.get_one::<u32>("ctx_size").unwrap();
    if CTX_SIZE.set(*ctx_size as usize).is_err() {
        return Err(ServerError::PromptContextSize);
    }
    println!("[SERVER] Prompt context size: {size}", size = ctx_size);

    println!("[SERVER] Starting server ...");

    // the timestamp when the server is created
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let ref_created = std::sync::Arc::new(created);

    let new_service = make_service_fn(move |_| {
        let model_name = ref_model_name.clone();
        let prompt_template_ty = ref_template_ty.clone();
        let created = ref_created.clone();
        async {
            Ok::<_, Error>(service_fn(move |req| {
                handle_request(
                    req,
                    model_name.to_string(),
                    *prompt_template_ty.clone(),
                    *created.clone(),
                )
            }))
        }
    });

    let server = Server::bind(&addr).serve(new_service);

    println!("[SERVER] Listening on http://{}", addr);
    match server.await {
        Ok(_) => Ok(()),
        Err(e) => Err(ServerError::InternalServerError(e.to_string())),
    }
}

async fn handle_request(
    req: Request<Body>,
    model_name: impl AsRef<str>,
    template_ty: PromptTemplateType,
    created: u64,
) -> Result<Response<Body>, hyper::Error> {
    match req.uri().path() {
        "/echo" => {
            return Ok(Response::new(Body::from("echo test")));
        }
        _ => ggml::handle_llama_request(req, model_name.as_ref(), template_ty, created).await,
    }
}
