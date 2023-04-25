use super::client::OpenAIRequest;
use reqwest::Method;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct Usage {
    pub prompt_tokens: i64,
    pub total_tokens: i64,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Embedding {
    pub object: String,
    pub embedding: Vec<f64>,
    pub index: i64,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Embeddings {
    pub object: String,
    pub data: Vec<Embedding>,
    pub model: String,
    pub usage: Usage,
}

#[derive(Serialize)]
pub struct EmbeddingRequest {
    model: String,
    input: Vec<String>,
}

impl OpenAIRequest for EmbeddingRequest {
    type Response = Embeddings;

    fn method() -> Method {
        Method::POST
    }

    fn url() -> &'static str {
        "https://api.openai.com/v1/embeddings"
    }
}
