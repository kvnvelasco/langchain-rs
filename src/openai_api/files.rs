use super::client::OpenAIRequest;
use reqwest::Method;
use serde::{Deserialize, Serialize, Serializer};
use std::collections::HashMap;

pub enum FilePayload<'a> {
    FineTuning {
        /// A string slice of the contents of the file
        payload: &'a str,
    },
}

impl<'a> Serialize for FilePayload<'a> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        #[derive(Serialize)]
        struct Payload<'a> {
            payload: &'a str,
            purpose: &'static str,
        }
        match self {
            FilePayload::FineTuning { payload } => Payload {
                payload,
                purpose: "fine-tuning",
            }
            .serialize(serializer),
        }
    }
}

impl<'a> OpenAIRequest for FilePayload<'a> {
    type Response = FileObject;

    fn method() -> Method {
        Method::POST
    }

    fn url() -> &'static str {
        todo!()
    }
}

#[derive(Deserialize)]
pub struct FileObject {
    pub id: String,
    pub object: String,
    pub bytes: i64,
    pub created_at: i64,
    pub filename: String,
    pub purpose: String,
}

#[cfg(test)]
mod test {
    #[test]
    fn serialisation() {
        use super::*;
        let payload = FilePayload::FineTuning {
            payload: "Hello world",
        };
        let json = serde_json::to_string(&payload).unwrap();
        assert_eq!(json, r#"{"payload":"Hello world","purpose":"fine-tuning"}"#);
    }
}
