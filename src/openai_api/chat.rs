use crate::openai_api::client::OpenAIRequest;
use reqwest::Method;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::HashMap;

#[derive(Deserialize, Debug)]
pub struct Usage {
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
    pub total_tokens: i64,
}

#[derive(Deserialize, Serialize, Debug, Copy, Clone)]
pub enum Role {
    #[serde(rename = "system")]
    System,
    #[serde(rename = "assistant")]
    Assistant,
    #[serde(rename = "user")]
    User,
}

impl ToString for Role {
    fn to_string(&self) -> String {
        match self {
            Role::System => "system".to_string(),
            Role::Assistant => "assistant".to_string(),
            Role::User => "user".to_string(),
        }
    }
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct Message<'a> {
    pub role: Role,
    pub content: Cow<'a, str>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct Choice {
    pub index: i64,
    pub message: Message<'static>,
    pub finish_reason: String,
}

#[derive(Deserialize, Debug)]
pub struct Chat {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Serialize, Debug, Default)]
pub struct ChatRequest<'a> {
    // ID of the model to use. See the model endpoint compatibility table for details on which models work with the Chat API.
    pub model: String,
    // The messages to generate chat completions for, in the chat format.
    pub messages: Vec<Message<'a>>,
    // What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. We generally recommend altering this or top_p but not both.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    // An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered. We generally recommend altering this or temperature but not both.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    // How many chat completion choices to generate for each input message.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<i32>,
    // If set, partial message deltas will be sent, like in ChatGPT. Tokens will be sent as data-only server-sent events as they become available, with the stream terminated by a data: [DONE] message. See the OpenAI Cookbook for example code.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    // Up to 4 sequences where the API will stop generating further tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    // The maximum number of tokens to generate in the chat completion. The total length of input tokens and generated tokens is limited by the model's context length.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<i32>,
    // Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics. See more information about frequency and presence penalties.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    // Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim. See more information about frequency and presence penalties.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    // Modify the likelihood of specified tokens appearing in the completion. Accepts a json object that maps tokens (specified by their token ID in the tokenizer) to an associated bias value from -100 to 100. Mathematically, the bias is added to the logits generated by the model prior to sampling. The exact effect will vary per model, but values between -1 and 1 should decrease or increase likelihood of selection; values like -100 or 100 should result in a ban or exclusive selection of the relevant token.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<HashMap<String, f32>>,
    // A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse. Learn more.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

impl<'a> OpenAIRequest for ChatRequest<'a> {
    type Response = Chat;

    fn method() -> Method {
        Method::POST
    }

    fn url() -> &'static str {
        "https://api.openai.com/v1/chat/completions"
    }
}
