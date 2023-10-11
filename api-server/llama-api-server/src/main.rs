use hyper::{
    service::{make_service_fn, service_fn},
    Body, Request, Response, Server,
};
use std::net::SocketAddr;

mod backend;
use backend::ggml;

mod error;
use error::ServerError;

type Error = Box<dyn std::error::Error + Send + Sync + 'static>;

const SOCKET_ADDRESS: &str = "0.0.0.0:8080";

#[derive(Clone, Debug)]
pub struct AppState {
    pub state_thing: String,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), ServerError> {
    println!("[SERVER] Starting server ...");

    let args: Vec<String> = std::env::args().collect();
    let model_name: String = match args.len() < 2 {
        true => String::new(),
        false => args[1].clone(),
    };
    let ref_model_name = std::sync::Arc::new(model_name);

    // read socket address
    let socket_addr = std::env::var("SOCKET_ADDRESS").unwrap_or(SOCKET_ADDRESS.to_string());
    let addr: SocketAddr = match socket_addr.parse() {
        Ok(addr) => addr,
        Err(e) => {
            return Err(ServerError::SocketAddr(e.to_string()));
        }
    };

    let new_service = make_service_fn(move |_| {
        let model_name = ref_model_name.clone();
        async {
            Ok::<_, Error>(service_fn(move |req| {
                handle_request(req, model_name.to_string())
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
) -> Result<Response<Body>, hyper::Error> {
    match req.uri().path() {
        "/echo" => {
            return Ok(Response::new(Body::from("echo test")));
        }
        _ => ggml::handle_llama_request(req, model_name.as_ref()).await,
    }
}
