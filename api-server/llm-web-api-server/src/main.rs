use hyper::{
    service::{make_service_fn, service_fn},
    Body, Request, Response, Server,
};
use std::net::SocketAddr;

mod config;
use config::{load_config, GatewayConfig, ServiceConfig, ServiceType};

mod backend;
use backend::{ggml, openai};

mod error;

type Error = Box<dyn std::error::Error + Send + Sync + 'static>;

const SOCKET_ADDRESS: &str = "0.0.0.0:8080";

#[derive(Clone, Debug)]
pub struct AppState {
    pub state_thing: String,
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    println!("[SERVER] Starting server ...");

    let args: Vec<String> = std::env::args().collect();
    let model_name: String = match args.len() < 2 {
        true => String::new(),
        false => args[1].clone(),
    };
    let ref_model_name = std::sync::Arc::new(model_name);

    let gateway_config = load_config("config.yml");

    // ! todo: read socket address from config file or command line arguments
    // let socket_addr = format!(
    //     "{ip}:{port}",
    //     ip = gateway_config.socket_addr.ip,
    //     port = gateway_config.socket_addr.port
    // );
    let socket_addr = String::from(SOCKET_ADDRESS);
    let addr: SocketAddr = match socket_addr.parse() {
        Ok(addr) => addr,
        Err(e) => {
            eprintln!("[SERVER] Invalid socket address: {}", e);
            return;
        }
    };

    let new_service = make_service_fn(move |_| {
        let config = gateway_config.clone();
        let model_name = ref_model_name.clone();
        async {
            Ok::<_, Error>(service_fn(move |req| {
                let config = config.clone();
                handle_request(req, config, model_name.to_string())
            }))
        }
    });

    let server = Server::bind(&addr).serve(new_service);

    println!("[SERVER] Listening on http://{}", addr);
    if let Err(e) = server.await {
        eprintln!("[SERVER] Failed to start up: {}", e);
    }
}

async fn handle_request(
    req: Request<Body>,
    config: GatewayConfig,
    model_name: impl AsRef<str>,
) -> Result<Response<Body>, hyper::Error> {
    let path = req.uri().path();

    // get service config
    let service_config = match get_service_config(path, &config.service_type, &config.services) {
        Some(service_config) => service_config,
        None => {
            return error::not_found();
        }
    };

    // dbg!("The service type is {:?}", service_config.ty);

    match service_config.ty {
        config::ServiceType::OpenAI => openai::handle_openai_request(req, service_config).await,
        config::ServiceType::GGML_Llama2 => {
            ggml::handle_llama_request(req, service_config, model_name.as_ref()).await
        }
        config::ServiceType::Test => Ok(Response::new(Body::from("echo test"))),
    }
}

fn get_service_config<'a>(
    path: &str,
    service_type: &'a ServiceType,
    services: &'a [ServiceConfig],
) -> Option<&'a ServiceConfig> {
    if path == "/echo" {
        services.iter().find(|c| path.starts_with(&c.path))
    } else {
        services
            .iter()
            .find(|c| path.starts_with(&c.path) && service_type == &c.ty)
    }
}
