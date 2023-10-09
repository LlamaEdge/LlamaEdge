use serde::{Deserialize, Serialize};
use std::{fs::File, io::Read};

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct GatewayConfig {
    #[serde(rename = "socket_address")]
    pub socket_addr: SocketAddr,
    pub service_type: ServiceType,
    pub services: Vec<ServiceConfig>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct SocketAddr {
    #[serde(rename = "ip_address")]
    pub ip: String,
    pub port: String,
}

#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum ServiceType {
    #[serde(rename = "openai")]
    OpenAI,
    #[serde(rename = "ggml/llama2")]
    GGML_Llama2,
    #[serde(rename = "test")]
    Test,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ServiceConfig {
    pub path: String,
    pub target_service: String,
    pub ty: ServiceType,
}

pub fn load_config(file_path: &str) -> GatewayConfig {
    let mut file = File::open(file_path).unwrap();
    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();
    serde_yaml::from_str(&contents).unwrap()
}
