mod backend;
mod error;

use chat_prompts::PromptTemplateType;
use clap::{crate_version, Arg, ArgAction, Command};
use error::ServerError;
use hyper::{
    header,
    service::{make_service_fn, service_fn},
    Body, Request, Response, Server, StatusCode,
};
use once_cell::sync::OnceCell;
use std::{net::SocketAddr, path::PathBuf, str::FromStr, sync::Mutex};
use wasi_nn::{Error as WasiNnError, Graph as WasiNnGraph, GraphExecutionContext, TensorType};

use llama_core::Metadata;

type Error = Box<dyn std::error::Error + Send + Sync + 'static>;

const DEFAULT_SOCKET_ADDRESS: &str = "0.0.0.0:8080";

static MAX_BUFFER_SIZE: OnceCell<usize> = OnceCell::new();
static CTX_SIZE: OnceCell<usize> = OnceCell::new();
static GRAPH: OnceCell<Mutex<Graph>> = OnceCell::new();
static METADATA: OnceCell<Metadata> = OnceCell::new();

#[derive(Clone, Debug)]
pub struct AppState {
    pub state_thing: String,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), ServerError> {
    let matches = Command::new("llama-api-server")
        .version(crate_version!())
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
                .default_value("512"),
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
                .default_value("512"),
        )
        .arg(
            Arg::new("temp")
                .long("temp")
                .value_parser(clap::value_parser!(f64))
                .value_name("TEMP")
                .help("Temperature for sampling")
                .default_value("1.0"),
        )
        .arg(
            Arg::new("top_p")
                .long("top-p")
                .value_parser(clap::value_parser!(f64))
                .value_name("TOP_P")
                .help("An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. 1.0 = disabled")
                .default_value("1.0"),
        )
        .arg(
            Arg::new("repeat_penalty")
                .long("repeat-penalty")
                .value_parser(clap::value_parser!(f64))
                .value_name("REPEAT_PENALTY")
                .help("Penalize repeat sequence of tokens")
                .default_value("1.1"),
        )
        .arg(
            Arg::new("presence_penalty")
                .long("presence-penalty")
                .value_parser(clap::value_parser!(f64))
                .value_name("PRESENCE_PENALTY")
                .help("Repeat alpha presence penalty. 0.0 = disabled")
                .default_value("0.0"),
        )
        .arg(
            Arg::new("frequency_penalty")
                .long("frequency-penalty")
                .value_parser(clap::value_parser!(f64))
                .value_name("FREQUENCY_PENALTY")
                .help("Repeat alpha frequency penalty. 0.0 = disabled")
                .default_value("0.0"),
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
                    "codellama-super-instruct",
                    "mistral-instruct-v0.1",
                    "mistral-instruct",
                    "mistrallite",
                    "openchat",
                    "human-assistant",
                    "vicuna-1.0-chat",
                    "vicuna-1.1-chat",
                    "chatml",
                    "baichuan-2",
                    "wizard-coder",
                    "zephyr",
                    "stablelm-zephyr",
                    "intel-neural",
                    "deepseek-chat",
                    "deepseek-coder",
                    "solar-instruct",
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
    let socket_addr = match matches.get_one::<String>("socket_addr") {
        Some(socket_addr) => socket_addr.to_string(),
        None => {
            return Err(ServerError::InternalServerError(
                "Failed to parse the value of `socket_addr` CLI option".to_owned(),
            ));
        }
    };
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
    let model_name = match matches.get_one::<String>("model_name") {
        Some(model_name) => model_name.to_string(),
        None => {
            return Err(ServerError::InternalServerError(
                "Failed to parse the value of `model_name` CLI option".to_owned(),
            ));
        }
    };
    println!("[INFO] Model name: {name}", name = &model_name);

    // model alias
    let model_alias = match matches.get_one::<String>("model_alias") {
        Some(model_alias) => model_alias.to_string(),
        None => {
            return Err(ServerError::InternalServerError(
                "Failed to parse the value of `model_alias` CLI option".to_owned(),
            ));
        }
    };
    println!("[INFO] Model alias: {alias}", alias = &model_alias);

    // create an `Options` instance
    let mut options = Metadata::default();

    // prompt context size
    let ctx_size = match matches.get_one::<u32>("ctx_size") {
        Some(ctx_size) => ctx_size,
        None => {
            return Err(ServerError::InternalServerError(
                "Failed to parse the value of `ctx_size` CLI option".to_owned(),
            ));
        }
    };
    println!("[INFO] Prompt context size: {size}", size = ctx_size);
    options.ctx_size = *ctx_size as u64;

    // set `CTX_SIZE`
    if CTX_SIZE.set(*ctx_size as usize).is_err() {
        return Err(ServerError::ContextSize);
    }
    // set `MAX_BUFFER_SIZE`
    if MAX_BUFFER_SIZE.set(*ctx_size as usize * 6).is_err() {
        return Err(ServerError::MaxBufferSize);
    }

    // number of tokens to predict
    let n_predict = match matches.get_one::<u32>("n_predict") {
        Some(n_predict) => n_predict,
        None => {
            return Err(ServerError::InternalServerError(
                "Failed to parse the value of `n_predict` CLI option".to_owned(),
            ));
        }
    };
    println!("[INFO] Number of tokens to predict: {n}", n = n_predict);
    options.n_predict = *n_predict as u64;

    // n_gpu_layers
    let n_gpu_layers = match matches.get_one::<u32>("n_gpu_layers") {
        Some(n_gpu_layers) => n_gpu_layers,
        None => {
            return Err(ServerError::InternalServerError(
                "Failed to parse the value of `n_gpu_layers` CLI option".to_owned(),
            ));
        }
    };
    println!(
        "[INFO] Number of layers to run on the GPU: {n}",
        n = n_gpu_layers
    );
    options.n_gpu_layers = *n_gpu_layers as u64;

    // batch size
    let batch_size = match matches.get_one::<u32>("batch_size") {
        Some(batch_size) => batch_size,
        None => {
            return Err(ServerError::InternalServerError(
                "Failed to parse the value of `batch_size` CLI option".to_owned(),
            ));
        }
    };
    println!(
        "[INFO] Batch size for prompt processing: {size}",
        size = batch_size
    );
    options.batch_size = *batch_size as u64;

    // temperature
    let temp = match matches.get_one::<f64>("temp") {
        Some(temp) => temp,
        None => {
            return Err(ServerError::InternalServerError(
                "Failed to parse the value of `temp` CLI option".to_owned(),
            ));
        }
    };
    println!("[INFO] Temperature for sampling: {temp}", temp = temp);
    options.temperature = *temp;

    // top-p
    let top_p = matches.get_one::<f64>("top_p").unwrap();
    println!(
        "[INFO] Top-p sampling (1.0 = disabled): {top_p}",
        top_p = top_p
    );

    // repeat penalty
    let repeat_penalty = match matches.get_one::<f64>("repeat_penalty") {
        Some(repeat_penalty) => repeat_penalty,
        None => {
            return Err(ServerError::InternalServerError(
                "Failed to parse the value of `repeat_penalty` CLI option".to_owned(),
            ));
        }
    };
    println!(
        "[INFO] Penalize repeat sequence of tokens: {penalty}",
        penalty = repeat_penalty
    );
    options.repeat_penalty = *repeat_penalty;

    // presence penalty
    let presence_penalty = matches.get_one::<f64>("presence_penalty").unwrap();
    println!(
        "[INFO] Presence penalty (0.0 = disabled): {penalty}",
        penalty = presence_penalty
    );
    options.presence_penalty = *presence_penalty;

    // frequency penalty
    let frequency_penalty = matches.get_one::<f64>("frequency_penalty").unwrap();
    println!(
        "[INFO] Frequency penalty (0.0 = disabled): {penalty}",
        penalty = frequency_penalty
    );
    options.frequency_penalty = *frequency_penalty;

    // reverse_prompt
    if let Some(reverse_prompt) = matches.get_one::<String>("reverse_prompt") {
        println!("[INFO] Reverse prompt: {prompt}", prompt = &reverse_prompt);
        options.reverse_prompt = Some(reverse_prompt.to_string());
    }

    // type of prompt template
    let prompt_template = match matches.get_one::<String>("prompt_template") {
        Some(prompt_template) => prompt_template.to_string(),
        None => {
            return Err(ServerError::InternalServerError(
                "Failed to parse the value of `prompt_template` CLI option".to_owned(),
            ))
        }
    };
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

    if log_stat || log_all {
        print_log_begin_separator(
            "MODEL INFO (Load Model & Init Execution Context)",
            Some("*"),
            None,
        );
    }

    // initialize the core context
    llama_core::init_core_context(&options, None).unwrap();

    if METADATA.set(options).is_err() {
        return Err(ServerError::Metadata);
    }

    let graph = {
        let metadata = match METADATA.get() {
            Some(metadata) => metadata,
            None => {
                return Err(ServerError::InternalServerError(
                    "The METADATA is not set".to_owned(),
                ));
            }
        };

        match Graph::new(model_alias, metadata) {
            Ok(graph) => graph,
            Err(e) => {
                return Err(ServerError::InternalServerError(e.to_string()));
            }
        }
    };

    if GRAPH.set(Mutex::new(graph)).is_err() {
        return Err(ServerError::InternalServerError(
            "The GRAPH has already been initialized".to_owned(),
        ));
    }

    // get the plugin version info
    {
        let graph = match GRAPH.get() {
            Some(graph) => graph,
            None => {
                return Err(ServerError::InternalServerError(
                    "The GRAPH is not set".to_owned(),
                ));
            }
        };

        let graph_locked = match graph.lock() {
            Ok(graph) => graph,
            Err(e) => {
                return Err(ServerError::InternalServerError(e.to_string()));
            }
        };

        // get version info
        let max_output_size = match MAX_BUFFER_SIZE.get() {
            Some(max_output_size) => *max_output_size,
            None => {
                return Err(ServerError::InternalServerError(
                    "The MAX_BUFFER_SIZE is not set".to_owned(),
                ));
            }
        };
        let mut output_buffer = vec![0u8; max_output_size];
        let mut output_size = match graph_locked.get_output(1, &mut output_buffer) {
            Ok(output_size) => output_size,
            Err(e) => {
                return Err(ServerError::InternalServerError(e.to_string()));
            }
        };
        output_size = std::cmp::min(max_output_size, output_size);
        let metadata: serde_json::Value =
            match serde_json::from_slice(&output_buffer[..output_size]) {
                Ok(metadata) => metadata,
                Err(e) => {
                    return Err(ServerError::InternalServerError(e.to_string()));
                }
            };

        let plugin_build_number = match metadata["llama_build_number"].as_u64() {
            Some(build_number) => build_number,
            None => {
                return Err(ServerError::InternalServerError(
                    "Failed to convert the build number of the plugin to u64".to_owned(),
                ));
            }
        };
        let plugin_commit = match metadata["llama_commit"].as_str() {
            Some(commit) => commit,
            None => {
                return Err(ServerError::InternalServerError(
                    "Failed to convert the commit id of the plugin to string".to_owned(),
                ));
            }
        };
        println!(
            "[INFO] Plugin version: b{} (commit {})",
            plugin_build_number, plugin_commit,
        );
    }

    if log_stat || log_all {
        print_log_end_separator(Some("*"), None);
    }

    let new_service = make_service_fn(move |_| {
        let prompt_template_ty = ref_template_ty.clone();
        let log_prompts = ref_log_prompts.clone();
        let web_ui = matches
            .get_one::<String>("web_ui")
            .unwrap_or(&"chatbot-ui".to_owned())
            .to_string();
        async move {
            Ok::<_, Error>(service_fn(move |req| {
                handle_request(
                    req,
                    *prompt_template_ty.clone(),
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
    template_ty: PromptTemplateType,
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
        "/v1" => backend::handle_llama_request(req, template_ty, log_prompts).await,
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

#[derive(Debug)]
pub(crate) struct Graph {
    _graph: WasiNnGraph,
    context: GraphExecutionContext,
}
impl Graph {
    pub fn new(model_alias: impl AsRef<str>, options: &Metadata) -> Result<Self, String> {
        let config = serde_json::to_string(&options).map_err(|e| e.to_string())?;

        // load the model
        let graph = wasi_nn::GraphBuilder::new(
            wasi_nn::GraphEncoding::Ggml,
            wasi_nn::ExecutionTarget::AUTO,
        )
        .config(config)
        .build_from_cache(model_alias.as_ref())
        .map_err(|e| e.to_string())?;

        // initialize the execution context
        let context = graph.init_execution_context().map_err(|e| e.to_string())?;

        Ok(Self {
            _graph: graph,
            context,
        })
    }

    pub fn _set_input<T: Sized>(
        &mut self,
        index: usize,
        tensor_type: TensorType,
        dimensions: &[usize],
        data: impl AsRef<[T]>,
    ) -> Result<(), WasiNnError> {
        self.context.set_input(index, tensor_type, dimensions, data)
    }

    pub fn _compute(&mut self) -> Result<(), WasiNnError> {
        self.context.compute()
    }

    pub fn _compute_single(&mut self) -> Result<(), WasiNnError> {
        self.context.compute_single()
    }

    pub fn get_output<T: Sized>(
        &self,
        index: usize,
        out_buffer: &mut [T],
    ) -> Result<usize, WasiNnError> {
        self.context.get_output(index, out_buffer)
    }

    pub fn _get_output_single<T: Sized>(
        &self,
        index: usize,
        out_buffer: &mut [T],
    ) -> Result<usize, WasiNnError> {
        self.context.get_output_single(index, out_buffer)
    }

    pub fn _finish_single(&mut self) -> Result<(), WasiNnError> {
        self.context.fini_single()
    }
}

pub(crate) fn print_log_begin_separator(
    title: impl AsRef<str>,
    ch: Option<&str>,
    len: Option<usize>,
) -> usize {
    let title = format!(" [LOG: {}] ", title.as_ref());

    let total_len: usize = len.unwrap_or(100);
    let separator_len: usize = (total_len - title.len()) / 2;

    let ch = ch.unwrap_or("-");
    let mut separator = "\n\n".to_string();
    separator.push_str(ch.repeat(separator_len).as_str());
    separator.push_str(&title);
    separator.push_str(ch.repeat(separator_len).as_str());
    separator.push_str("\n");
    println!("{}", separator);
    total_len
}

pub(crate) fn print_log_end_separator(ch: Option<&str>, len: Option<usize>) {
    let ch = ch.unwrap_or("-");
    let mut separator = "\n\n".to_string();
    separator.push_str(ch.repeat(len.unwrap_or(100)).as_str());
    separator.push_str("\n");
    println!("{}", separator);
}
