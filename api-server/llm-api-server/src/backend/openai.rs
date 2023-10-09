use super::ServiceConfig;
use hyper::{http::request::Parts, Body, Request, Response};

pub async fn handle_openai_request(
    req: Request<Body>,
    service_config: &ServiceConfig,
) -> Result<Response<Body>, hyper::Error> {
    // get openai_api_key
    let auth_token = format!(
        "Bearer {openai_api_key}",
        openai_api_key = std::env::var("OPENAI_API_KEY").unwrap()
    );

    let (parts, body) = req.into_parts();
    let downstream_req = build_downstream_request(parts, body, service_config, auth_token).await?;

    dbg!("downstream_req: {:?}", &downstream_req);

    match forward_request(downstream_req).await {
        Ok(res) => Ok(res),
        Err(e) => {
            dbg!(&e);

            service_unavailable(format!("Failed to connect to downstream service. {:?}", e))
        }
    }
}

async fn build_downstream_request(
    parts: Parts,
    body: Body,
    service_config: &ServiceConfig,
    auth_token: String,
) -> Result<Request<Body>, hyper::Error> {
    let req = Request::from_parts(parts, body);
    let uri = service_config.target_service.as_str();

    let mut downstream_req_builder = Request::builder().uri(uri).method(req.method());

    // headers
    let headers = downstream_req_builder.headers_mut().unwrap();
    headers.insert("Content-Type", "application/json".parse().unwrap());
    headers.insert("Authorization", auth_token.as_str().parse().unwrap());

    // body
    let body_bytes = hyper::body::to_bytes(req.into_body()).await?;
    let downstream_req = downstream_req_builder.body(Body::from(body_bytes)).unwrap();

    Ok(downstream_req)
}

async fn forward_request(req: Request<Body>) -> Result<Response<Body>, hyper::Error> {
    // create a https connector
    let https_conn = wasmedge_hyper_rustls::connector::new_https_connector(
        wasmedge_rustls_api::ClientConfig::default(),
    );

    let client = hyper::Client::builder().build::<_, hyper::Body>(https_conn);

    match client.request(req).await {
        Ok(res) => Ok(res),
        Err(e) => Err(e),
    }
}

fn service_unavailable<T>(reason: T) -> Result<Response<Body>, hyper::Error>
where
    T: Into<Body>,
{
    let mut response = Response::new(reason.into());
    *response.status_mut() = hyper::StatusCode::SERVICE_UNAVAILABLE;
    Ok(response)
}
