mod backend;
mod error;

use chat_prompts::PromptTemplateType;
use clap::{Arg, ArgAction, Command};
use error::ServerError;
use hyper::{
    header,
    service::{make_service_fn, service_fn},
    Body, Request, Response, Server, StatusCode,
};
use once_cell::sync::OnceCell;
use serde::{Deserialize, Serialize};
use std::{net::SocketAddr, path::PathBuf, str::FromStr, sync::Mutex};
use wasi_nn::{Error as WasiNnError, Graph as WasiNnGraph, GraphExecutionContext, TensorType};

type Error = Box<dyn std::error::Error + Send + Sync + 'static>;

const DEFAULT_SOCKET_ADDRESS: &str = "0.0.0.0:8080";
const DEFAULT_CTX_SIZE: &str = "4096";

static CTX_SIZE: OnceCell<usize> = OnceCell::new();

static GRAPH: OnceCell<Mutex<Graph>> = OnceCell::new();

#[derive(Clone, Debug)]
pub struct AppState {
    pub state_thing: String,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), ServerError> {
    let matches = Command::new("Llama API Server")
        .arg(
            Arg::new("socket_addr")
                .short('s')
                .long("socket-addr")
                .value_name("IP:PORT")
                .help("Sets the socket address")
                .default_value(DEFAULT_SOCKET_ADDRESS),
        )
        .arg(
            Arg::new("model_name")
                .short('m')
                .long("model-name")
                .value_name("MODEL-NAME")
                .help("Sets the model name")
                .default_value("default"),
        )
        .arg(
            Arg::new("model_alias")
                .short('a')
                .long("model-alias")
                .value_name("MODEL-ALIAS")
                .help("Sets the alias name of the model in WasmEdge runtime")
                .default_value("default"),
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
        .arg(
            Arg::new("n_predict")
                .short('n')
                .long("n-predict")
                .value_parser(clap::value_parser!(u32))
                .value_name("N_PRDICT")
                .help("Number of tokens to predict")
                .default_value("1024"),
        )
        .arg(
            Arg::new("n_gpu_layers")
                .short('g')
                .long("n-gpu-layers")
                .value_parser(clap::value_parser!(u32))
                .value_name("N_GPU_LAYERS")
                .help("Number of layers to run on the GPU")
                .default_value("100"),
        )
        .arg(
            Arg::new("batch_size")
                .short('b')
                .long("batch-size")
                .value_parser(clap::value_parser!(u32))
                .value_name("BATCH_SIZE")
                .help("Batch size for prompt processing")
                .default_value("4096"),
        )
        .arg(
            Arg::new("reverse_prompt")
                .short('r')
                .long("reverse-prompt")
                .value_name("REVERSE_PROMPT")
                .help("Halt generation at PROMPT, return control."),
        )
        .arg(
            Arg::new("prompt_template")
                .short('p')
                .long("prompt-template")
                .value_parser([
                    "llama-2-chat",
                    "codellama-instruct",
                    "mistral-instruct-v0.1",
                    "mistrallite",
                    "openchat",
                    "belle-llama-2-chat",
                    "vicuna-chat",
                    "vicuna-1.1-chat",
                    "chatml",
                    "baichuan-2",
                    "wizard-coder",
                    "zephyr",
                    "intel-neural",
                    "deepseek-chat",
                    "deepseek-coder",
                ])
                .value_name("TEMPLATE")
                .help("Sets the prompt template.")
                .default_value("llama-2-chat"),
        )
        .arg(
            Arg::new("log_prompts")
                .long("log-prompts")
                .value_name("LOG_PROMPTS")
                .help("Print prompt strings to stdout")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("log_stat")
                .long("log-stat")
                .value_name("LOG_STAT")
                .help("Print statistics to stdout")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("log_all")
                .long("log-all")
                .value_name("LOG_all")
                .help("Print all log information to stdout")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("web_ui")
                .long("web-ui")
                .value_name("WEB_UI")
                .help("Root path for the Web UI files")
                .default_value("chatbot-ui"),
        )
        .get_matches();

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
        "[INFO] Socket address: {socket_addr}",
        socket_addr = socket_addr
    );

    // model name
    let model_name = matches.get_one::<String>("model_name").unwrap().to_string();
    println!("[INFO] Model name: {name}", name = &model_name);

    // model alias
    let model_alias = matches
        .get_one::<String>("model_alias")
        .unwrap()
        .to_string();
    println!("[INFO] Model alias: {alias}", alias = &model_alias);

    // create a `ModelInfo` instance
    let model_info = ModelInfo::new(model_name);

    // create an `Options` instance
    let mut options = Metadata::default();

    // prompt context size
    let ctx_size = matches.get_one::<u32>("ctx_size").unwrap();
    if CTX_SIZE.set(*ctx_size as usize * 6).is_err() {
        return Err(ServerError::PromptContextSize);
    }
    println!("[INFO] Prompt context size: {size}", size = ctx_size);

    // number of tokens to predict
    let n_predict = matches.get_one::<u32>("n_predict").unwrap();
    println!("[INFO] Number of tokens to predict: {n}", n = n_predict);
    options.n_predict = *n_predict as u64;

    // n_gpu_layers
    let n_gpu_layers = matches.get_one::<u32>("n_gpu_layers").unwrap();
    println!(
        "[INFO] Number of layers to run on the GPU: {n}",
        n = n_gpu_layers
    );
    options.n_gpu_layers = *n_gpu_layers as u64;

    // batch size
    let batch_size = matches.get_one::<u32>("batch_size").unwrap();
    println!(
        "[INFO] Batch size for prompt processing: {size}",
        size = batch_size
    );
    options.batch_size = *batch_size as u64;

    // reverse_prompt
    if let Some(reverse_prompt) = matches.get_one::<String>("reverse_prompt") {
        println!("[INFO] Reverse prompt: {prompt}", prompt = &reverse_prompt);
        options.reverse_prompt = Some(reverse_prompt.to_string());
    }

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
    println!("[INFO] Prompt template: {ty:?}", ty = &template_ty);
    let ref_template_ty = std::sync::Arc::new(template_ty);

    // log prompts
    let log_prompts = matches.get_flag("log_prompts");
    println!("[INFO] Log prompts: {enable}", enable = log_prompts);
    let ref_log_prompts = std::sync::Arc::new(log_prompts);

    // log statistics
    let log_stat = matches.get_flag("log_stat");
    println!("[INFO] Log statistics: {enable}", enable = log_stat);

    // log all
    let log_all = matches.get_flag("log_all");
    println!("[INFO] Log all information: {enable}", enable = log_all);

    // set `log_enable`
    if log_stat || log_all {
        options.log_enable = true;
    }

    println!("[INFO] Starting server ...");

    let graph = Graph::new(model_alias, &options);
    if GRAPH.set(Mutex::new(graph)).is_err() {
        return Err(ServerError::InternalServerError(
            "The GRAPH has already been initialized".to_owned(),
        ));
    }

    // the timestamp when the server is created
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let ref_created = std::sync::Arc::new(created);

    let new_service = make_service_fn(move |_| {
        let model_info = model_info.clone();
        let prompt_template_ty = ref_template_ty.clone();
        let created = ref_created.clone();
        let log_prompts = ref_log_prompts.clone();
        let web_ui = matches
            .get_one::<String>("web_ui")
            .unwrap_or(&"chatbot-ui".to_owned())
            .to_string();
        async {
            Ok::<_, Error>(service_fn(move |req| {
                handle_request(
                    req,
                    model_info.clone(),
                    *prompt_template_ty.clone(),
                    *created.clone(),
                    *log_prompts.clone(),
                    web_ui.clone(),
                )
            }))
        }
    });

    let server = Server::bind(&addr).serve(new_service);

    println!("[INFO] Listening on http://{}", addr);

    match server.await {
        Ok(_) => Ok(()),
        Err(e) => Err(ServerError::InternalServerError(e.to_string())),
    }
}

async fn handle_request(
    req: Request<Body>,
    model_info: ModelInfo,
    template_ty: PromptTemplateType,
    created: u64,
    log_prompts: bool,
    web_ui: String,
) -> Result<Response<Body>, hyper::Error> {
    let path_str = req.uri().path();
    let path_buf = PathBuf::from(path_str);
    let mut path_iter = path_buf.iter();
    path_iter.next(); // Must be Some(OsStr::new(&path::MAIN_SEPARATOR.to_string()))
    let root_path = path_iter.next().unwrap_or_default();
    let root_path = "/".to_owned() + root_path.to_str().unwrap_or_default();

    match root_path.as_str() {
        "/echo" => {
            return Ok(Response::new(Body::from("echo test")));
        }
        "/v1" => {
            backend::handle_llama_request(req, model_info, template_ty, created, log_prompts).await
        }
        _ => Ok(static_response(path_str, web_ui)),
    }
}

fn static_response(path_str: &str, root: String) -> Response<Body> {
    let path = match path_str {
        "/" => "/index.html",
        _ => path_str,
    };

    let mime = mime_guess::from_path(path);

    match std::fs::read(format!("{root}/{path}")) {
        Ok(content) => Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, mime.first_or_text_plain().to_string())
            .body(Body::from(content))
            .unwrap(),
        Err(_) => {
            let body = Body::from(std::fs::read(format!("{root}/404.html")).unwrap_or_default());
            Response::builder()
                .status(StatusCode::NOT_FOUND)
                .header(header::CONTENT_TYPE, "text/html")
                .body(body)
                .unwrap()
        }
    }
}

#[derive(Debug, Default, Clone, Deserialize, Serialize)]
struct Metadata {
    #[serde(rename = "enable-log")]
    log_enable: bool,
    #[serde(rename = "stream-stdout")]
    stream_stdout: bool,
    #[serde(rename = "ctx-size")]
    ctx_size: u64,
    #[serde(rename = "n-predict")]
    n_predict: u64,
    #[serde(rename = "n-gpu-layers")]
    n_gpu_layers: u64,
    #[serde(rename = "batch-size")]
    batch_size: u64,
    #[serde(skip_serializing_if = "Option::is_none", rename = "reverse-prompt")]
    reverse_prompt: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ModelInfo {
    name: String,
}
impl ModelInfo {
    fn new(name: impl AsRef<str>) -> Self {
        Self {
            name: name.as_ref().to_string(),
        }
    }
}

#[derive(Debug)]
struct Graph {
    _graph: WasiNnGraph,
    context: GraphExecutionContext,
}
impl Graph {
    pub fn new(model_alias: impl AsRef<str>, options: &Metadata) -> Self {
        // load the model
        let graph = wasi_nn::GraphBuilder::new(
            wasi_nn::GraphEncoding::Ggml,
            wasi_nn::ExecutionTarget::AUTO,
        )
        .build_from_cache(model_alias.as_ref())
        .unwrap();

        // initialize the execution context
        let mut context = graph.init_execution_context().unwrap();

        // set metadata
        let buffer = serde_json::to_vec(options).unwrap();
        context
            .set_input(1, wasi_nn::TensorType::U8, &[1], buffer)
            .unwrap();

        Self {
            _graph: graph,
            context,
        }
    }

    pub fn set_input<T: Sized>(
        &mut self,
        index: usize,
        tensor_type: TensorType,
        dimensions: &[usize],
        data: impl AsRef<[T]>,
    ) -> Result<(), WasiNnError> {
        self.context.set_input(index, tensor_type, dimensions, data)
    }

    pub fn compute(&mut self) -> Result<(), WasiNnError> {
        self.context.compute()
    }

    pub fn get_output<T: Sized>(
        &self,
        index: usize,
        out_buffer: &mut [T],
    ) -> Result<usize, WasiNnError> {
        self.context.get_output(index, out_buffer)
    }
}
