use eyre::WrapErr;
use reqwest::{IntoUrl, RequestBuilder};
use serde::Serialize;

#[derive(Clone)]
pub struct Client(reqwest::Client);

#[derive(thiserror::Error, Debug)]
#[error("Failed to build OpenAI client: {0}")]
pub struct ClientBuildError(#[from] eyre::Report);

#[derive(thiserror::Error, Debug)]
#[error("Failed to make OpenAI request: {0}")]
pub struct RequestError(#[from] eyre::Report);

impl Client {
    pub fn new(
        access_token: impl AsRef<str>,
        organization: impl AsRef<str>,
    ) -> Result<Self, ClientBuildError> {
        let header_map = reqwest::header::HeaderMap::from_iter([
            (
                reqwest::header::AUTHORIZATION,
                format!("Bearer {}", access_token.as_ref())
                    .try_into()
                    .wrap_err("Unable to convert access token to header value")?,
            ),
            (
                "OpenAI-Organization".try_into().unwrap(),
                organization
                    .as_ref()
                    .try_into()
                    .wrap_err("Unable to parse organization id into header value")?,
            ),
        ]);

        Ok(Self(
            reqwest::Client::builder()
                .default_headers(header_map)
                .build()
                .wrap_err("Unable to build OpenAI client")?,
        ))
    }

    pub async fn request<T: OpenAIRequest>(&self, req: T) -> Result<T::Response, RequestError> {
        let res = self
            .0
            .request(T::method(), T::url())
            .json(&req)
            .send()
            .await
            .wrap_err("Unable to build request")?;

        let output = res.text().await.wrap_err("Unable to get response text")?;

        Ok(serde_json::from_str(&output)
            .wrap_err_with(|| format!("Unable to parse response as JSON: {}", output))?)
    }
}

pub trait OpenAIRequest: Serialize {
    type Response: serde::de::DeserializeOwned;

    fn method() -> reqwest::Method;
    fn url() -> &'static str;
}

#[cfg(test)]
mod test {
    #[tokio::test]
    async fn client_builder() {
        let client = super::Client::new("test", "org name").unwrap();

        let mut server = mockito::Server::new();

        let mock = server
            .mock("GET", "/")
            .match_header("Authorization", "Bearer test")
            .match_header("OpenAI-Organization", "org name")
            .create();

        let request = client.0.get(server.url()).send().await;
        mock.assert();
    }
}
