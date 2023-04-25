mod openai_api;
mod prompt {
    use eyre::{bail, WrapErr};
    use handlebars::template::{HelperTemplate, Parameter, TemplateElement};
    use handlebars::{Context, Handlebars, RenderContext, Renderable, StringOutput, Template};
    use std::collections::HashMap;
    use std::str::FromStr;

    // All prompt templates may contain multiple arguments. Templates are in the handlebars templating language
    // thus any context is available to the template for as long as it is serializable.
    // However, the template is always called with the "question" data field when called in a chain.

    // Make sure that the template has the {{ question }} field always available
    // Metadata is data that can be passed forward to models or agents to know a little bit more about the structure
    // of the template. It's used by agents specifically to know what the Stop token should be and how to extract
    // data out of the responses.
    #[derive(Clone, Debug)]
    pub struct PromptTemplate {
        tpl: Template,
        data: HashMap<String, serde_json::Value>,
        pub meta: HashMap<String, serde_json::Value>,
    }

    impl FromStr for PromptTemplate {
        type Err = eyre::Report;

        fn from_str(s: &str) -> eyre::Result<Self> {
            let template = Template::compile(s)
                .wrap_err("Unable to compile handlebars template for prompt")?;

            // check to see if the template has the correct fields
            let has_question = template.elements.iter().any(
                |t| matches!(t, TemplateElement::Expression( e) if matches!( e.name.as_name(), Some(e) if e == "question")),
            );
            if !has_question {
                return bail!("Template must have a {{ question }} field");
            }

            Ok(Self {
                tpl: template,
                data: Default::default(),
                meta: Default::default(),
            })
        }
    }

    #[derive(Debug, thiserror::Error)]
    pub enum TemplateError {
        #[error("Invalid prompt argument. When using named arguments, positional arguments are not allowed and vise versa")]
        RenderFailure,
    }

    impl PromptTemplate {
        pub fn set_named_arg(&mut self, name: impl AsRef<str>, data: impl Into<serde_json::Value>) {
            self.data.insert(name.as_ref().to_string(), data.into());
        }

        pub fn render(&self, question: &str) -> Result<String, TemplateError> {
            let registry = Handlebars::new();

            let data = {
                let mut d = self.data.clone();
                d.insert("question".to_string(), question.into());
                d
            };

            let ctx = Context::wraps(data).map_err(|_| TemplateError::RenderFailure)?;

            let mut render_context = RenderContext::new(None);
            let mut output = StringOutput::default();

            self.tpl
                .render(&registry, &ctx, &mut render_context, &mut output)
                .map_err(|_| TemplateError::RenderFailure)?;

            output
                .into_string()
                .map_err(|_| TemplateError::RenderFailure)
        }
    }

    #[cfg(test)]
    mod tests {
        use crate::prompt::PromptTemplate;

        #[test]
        fn question_parameter_required() {
            let tpl: Result<PromptTemplate, _> =
                "Hello {{ name }}. Here is a question: {{ question }}?".parse();

            assert!(tpl.is_ok(), "{tpl:?}");

            let tpl: Result<PromptTemplate, _> = "Hello {{ name }}?".parse();

            assert!(tpl.is_err(), "{tpl:?}");
        }
    }
}

mod protocol {
    use crate::memory::Memory;
    use std::future::Future;

    #[derive(Debug)]
    pub struct Request {
        pub input: String,
        pub memory: Option<Memory>,
    }

    impl From<String> for Request {
        fn from(input: String) -> Self {
            Self {
                input,
                memory: None,
            }
        }
    }

    impl<'a> From<&'a str> for Request {
        fn from(value: &'a str) -> Self {
            value.to_owned().into()
        }
    }

    #[derive(Debug)]
    pub struct Response {
        pub output: String,
        pub memory: Memory,
    }

    #[derive(Debug, thiserror::Error)]
    pub enum Error {
        #[error("Reco")]
        Recoverable { request: Request },
        #[error("Unrecoverable error: {message}")]
        Unrecoverable { message: String },
    }

    pub type DynFuture = dyn Future<Output = Result<Response, Error>> + Send;
}

mod agent {
    use crate::memory::{Memory, MemoryOperation};
    use crate::model::{ModelService, NoModel};
    use crate::prompt::PromptTemplate;
    use crate::protocol::{DynFuture, Error, Request, Response};
    use async_trait::async_trait;
    use serde_json::Value;
    use std::collections::HashMap;
    use std::future::poll_fn;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::task::{Context, Poll, RawWaker, Waker};
    use tower_service::Service;

    pub struct Agent {
        name: Option<String>,
        template: PromptTemplate,
        model: ModelService,
        tools: Vec<Arc<dyn Tool + Send + Sync>>,
    }

    impl Service<Request> for Agent {
        type Response = Response;
        type Error = Error;
        type Future = Pin<Box<DynFuture>>;

        fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
            // Prepare templating
            let (tools, tool_names) =
                self.tools
                    .iter()
                    .fold((vec![], vec![]), |(mut tools, mut names), tool| {
                        tools.push(format!("{}: {}", tool.name(), tool.description()));
                        names.push(tool.name());
                        (tools, names)
                    });
            self.template.set_named_arg("tools", tools);
            self.template.set_named_arg("tool_names", tool_names);

            // check expected values in the template
            if !matches!(
                self.template.meta.get("observation_token"),
                Some(Value::String(_))
            ) {
                return Poll::Ready(Err(Error::Unrecoverable {
                    message: "Template must have a observation_token meta field".to_string(),
                }));
            }

            // check expected values in the template
            if !matches!(
                self.template.meta.get("action_token"),
                Some(Value::String(_))
            ) {
                return Poll::Ready(Err(Error::Unrecoverable {
                    message: "Template must have a action_token meta field".to_string(),
                }));
            }

            // check expected values in the template
            if !matches!(
                self.template.meta.get("action_input_token"),
                Some(Value::String(_))
            ) {
                return Poll::Ready(Err(Error::Unrecoverable {
                    message: "Template must have a action_input_token meta field".to_string(),
                }));
            }

            Poll::Ready(Ok(()))
        }

        fn call(&mut self, req: Request) -> Self::Future {
            async fn execute(
                req: Request,
                template: PromptTemplate,
                mut model: ModelService,
            ) -> Result<Response, Error> {
                let mut memory = {
                    let mut m = Memory::default();
                    let pr =
                        template
                            .render(req.input.as_str())
                            .map_err(|e| Error::Unrecoverable {
                                message: e.to_string(),
                            })?;
                    m.operate(MemoryOperation::System(pr));
                    m
                };

                loop {
                    poll_fn(|cx| model.poll_ready(cx)).await?;

                    let result = model
                        .call(Request {
                            input: Default::default(),
                            memory: Some(std::mem::take(&mut memory)),
                        })
                        .await?;

                    memory = result.memory;
                }
            }
            Box::pin(execute(req, self.template.clone(), self.model.clone()))
        }
    }

    impl Default for Agent {
        fn default() -> Self {
            let mut template: PromptTemplate =
                include_str!("./default_agent_prompt.hbs").parse().unwrap();
            template.meta = HashMap::from_iter([
                ("observation_token".to_string(), "Observation:".into()),
                ("action_token".to_string(), "Action:".into()),
                ("action_input_token".to_string(), "Action Input:".into()),
            ]);

            Self {
                name: None,
                template,
                model: NoModel.into(),
                tools: vec![],
            }
        }
    }

    pub enum ToolError {
        ParseError,
        RunError,
    }

    #[async_trait]
    pub trait Tool {
        fn name(&self) -> &str;
        fn description(&self) -> &str;
        async fn run(&self, input: &str) -> Result<ToolOperation, ToolError>;
    }

    #[derive(Debug)]
    pub struct ToolOperation {
        output: String,
    }
}

pub mod memory {
    #[derive(Debug)]
    #[non_exhaustive]
    pub enum MemoryOperation {
        /// Instructions and directives provided to LLMs
        System(String),
        /// Messages that come from LLMs
        Assistant(String),
        /// Messages that come from users
        User(String),
    }

    #[derive(Default, Debug)]
    pub struct Memory {
        operations: Vec<MemoryOperation>,
    }

    impl Memory {
        pub fn operate(&mut self, op: MemoryOperation) {
            self.operations.push(op);
        }

        pub fn recall(&self) -> impl Iterator<Item = &MemoryOperation> {
            self.operations.iter()
        }

        pub fn into_iter(self) -> impl Iterator<Item = MemoryOperation> {
            self.operations.into_iter()
        }

        pub fn last(&self) -> Option<&MemoryOperation> {
            self.operations.last()
        }
    }
}

mod model {
    use crate::memory::Memory;
    use crate::protocol::{DynFuture, Error, Request, Response};
    use async_trait::async_trait;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::task::{Context, Poll};
    use tower_service::Service;

    #[derive(Clone)]
    pub struct ModelService {
        exec: Arc<dyn Model + Send + Sync>,
        config: Option<Config>,
    }

    impl<T> From<T> for ModelService
    where
        T: Model + 'static,
        T: Send + Sync,
    {
        fn from(value: T) -> Self {
            Self {
                exec: Arc::new(value),
                config: None,
            }
        }
    }

    impl Service<Request> for ModelService {
        type Response = Response;
        type Error = Error;
        type Future = Pin<Box<DynFuture>>;

        fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
            Poll::Ready(Ok(()))
        }

        fn call(&mut self, req: Request) -> Self::Future {
            let exec = self.exec.clone();
            let conf = self.config.clone();
            Box::pin(async move {
                let memory = exec.init(&req.input, req.memory);
                exec.exec(&conf.unwrap_or_default(), memory).await
            })
        }
    }

    #[derive(Debug, Default, Clone)]
    pub struct Config {
        pub limit: Option<usize>,
        pub stop_tokens: Option<Vec<String>>,
    }

    #[async_trait]
    pub trait Model {
        /// When the model is first called it will attempt to generate it's memory from input and
        /// potentially existing memory. It's up to the model to decide how to behave when existing memory is present.
        ///
        /// Existing memory may come from a persistence layer or from retry. When models return recoverable errors they
        /// may be retried in the future with existing memory passed back in.
        fn init(&self, input: &str, memory: Option<Memory>) -> Memory;

        //// Modify memory in some way. This may represent an ai operation
        async fn exec(&self, config: &Config, input: Memory) -> Result<Response, Error>;
    }

    /// No-op model that can be used for testing
    pub struct NoModel;

    #[async_trait]
    impl Model for NoModel {
        fn init(&self, input: &str, memory: Option<Memory>) -> Memory {
            memory.unwrap_or_default()
        }

        async fn exec(&self, _config: &Config, input: Memory) -> Result<Response, Error> {
            Ok(Response {
                memory: input,
                output: Default::default(),
            })
        }
    }

    pub mod openai {
        use crate::memory::{Memory, MemoryOperation};
        use crate::model::{Config, Model};
        use crate::openai_api::chat::{ChatRequest, Message, Role};
        use crate::openai_api::client::Client;
        use crate::prompt::PromptTemplate;
        use crate::protocol::{Error, Response};
        use async_trait::async_trait;

        pub enum ModelType {
            Ada,
            Babbage,
            Curie,
            Davinci,
            Gpt35Turbo,
            Gpt4,
        }

        impl ModelType {
            fn to_string(&self) -> String {
                match self {
                    ModelType::Ada => "ada".to_string(),
                    ModelType::Babbage => "babbage".to_string(),
                    ModelType::Curie => "curie".to_string(),
                    ModelType::Davinci => "davinci".to_string(),
                    ModelType::Gpt35Turbo => "gpt-3.5-turbo".to_string(),
                    ModelType::Gpt4 => "gpt4".to_string(),
                }
            }
        }

        pub struct Chat {
            pub prompt: PromptTemplate,
            pub model: ModelType,
            client: Client,
        }

        impl Chat {
            pub fn new(
                prompt: PromptTemplate,
                model: ModelType,
                access_token: String,
                organization_id: String,
            ) -> Self {
                Self {
                    prompt,
                    model,
                    client: Client::new(access_token, organization_id).unwrap(),
                }
            }
        }

        #[async_trait]
        impl Model for Chat {
            fn init(&self, input: &str, memory: Option<Memory>) -> Memory {
                if let Some(memory) = memory {
                    // this is resumption
                    memory
                } else {
                    let initial = self.prompt.render(input).unwrap();
                    eprintln!("Prompting model with {initial}");
                    let mut mem = Memory::default();
                    mem.operate(MemoryOperation::System(initial));
                    mem
                }
            }

            async fn exec(&self, config: &Config, mut input: Memory) -> Result<Response, Error> {
                let messages = input
                    .recall()
                    .map(|m| match m {
                        MemoryOperation::System(s) => Message {
                            role: Role::System,
                            content: s.into(),
                        },
                        MemoryOperation::Assistant(s) => Message {
                            role: Role::Assistant,
                            content: s.into(),
                        },
                        MemoryOperation::User(s) => Message {
                            role: Role::User,
                            content: s.into(),
                        },
                    })
                    .collect::<Vec<_>>();

                let response = self
                    .client
                    .request(ChatRequest {
                        model: self.model.to_string(),
                        stop: config.stop_tokens.clone(),
                        messages,
                        ..Default::default()
                    })
                    .await
                    .map_err(|e| Error::Unrecoverable {
                        message: e.to_string(),
                    })?;

                // extract the messages out of the chat
                let response = response.choices.get(0).ok_or(Error::Unrecoverable {
                    message: "No response from openai".to_string(),
                })?;

                input.operate(match response.message.role {
                    Role::System => MemoryOperation::System(response.message.content.to_string()),
                    Role::Assistant => {
                        MemoryOperation::Assistant(response.message.content.to_string())
                    }
                    Role::User => MemoryOperation::User(response.message.content.to_string()),
                });

                eprintln!("{}", response.message.content);
                Ok(Response {
                    output: response.message.content.to_string(),
                    memory: input,
                })
            }
        }
    }
}

pub mod chain {
    use crate::agent::Agent;
    use crate::model::ModelService;
    use crate::protocol::{DynFuture, Error, Request, Response};
    use std::fmt::Display;
    use std::pin::Pin;
    use std::task::{Context, Poll};
    use tower::buffer::Buffer;
    use tower_service::Service;

    type Executor = dyn Service<Request, Response = Response, Error = Error, Future = Pin<Box<DynFuture>>>
        + Send;

    #[derive(Default)]
    pub struct Chain {
        services: Vec<Buffer<Box<Executor>, Request>>,
    }

    impl Chain {
        pub fn model<C>(mut self, implementor: C) -> Self
        where
            C: Into<ModelService>,
        {
            self.services
                .push(Buffer::new(Box::new(implementor.into()), 10));
            self
        }

        pub fn agent(mut self, agent: Agent) -> Self {
            self.services.push(Buffer::new(Box::new(agent), 10));
            self
        }

        pub fn service<S, C>(mut self, service: C) -> Self
        where
            C: Into<S>,
            S: Service<Request, Response = Response, Error = Error, Future = Pin<Box<DynFuture>>>,
            S: Send + Sync + 'static,
        {
            self.services
                .push(Buffer::new(Box::new(service.into()), 10));
            self
        }
    }

    impl Service<Request> for Chain {
        type Response = Response;
        type Error = Error;
        type Future = Pin<Box<DynFuture>>;

        fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
            if self.services.is_empty() {
                return Poll::Ready(Err(Error::Unrecoverable {
                    message: "no services in chain".to_string(),
                }));
            }
            for service in self.services.iter_mut() {
                let ready = service.poll_ready(cx);
                if ready.is_pending() {
                    return Poll::Pending;
                }

                if let Poll::Ready(Err(e)) = ready {
                    return Poll::Ready(Err(Error::Unrecoverable {
                        message: e.to_string(),
                    }));
                }
            }

            Poll::Ready(Ok(()))
        }

        fn call(&mut self, req: Request) -> Self::Future {
            // these are all buffered so the underlying services aren't clone;
            let clon = self.services.clone();
            let services = std::mem::replace(&mut self.services, clon);

            async fn execute<T>(services: Vec<T>, req: Request) -> Result<Response, Error>
            where
                T: Service<Request, Response = Response>,
                T::Error: Display,
            {
                let mut req = Some(req);
                for mut service in services {
                    let resp = service.call(req.take().unwrap()).await;
                    match resp {
                        Ok(resp) => {
                            req = Some(Request {
                                input: resp.output,
                                memory: None,
                            })
                        }
                        Err(e) => {
                            return Err(Error::Unrecoverable {
                                message: e.to_string(),
                            });
                        }
                    }
                }

                if let Some(req) = req {
                    Ok(Response {
                        output: req.input,
                        memory: req.memory.unwrap_or_default(),
                    })
                } else {
                    Err(Error::Unrecoverable {
                        message: "no response".to_string(),
                    })
                }
            }

            Box::pin(execute(services, req))
        }
    }

    #[cfg(test)]
    mod tests {
        use crate::chain::Chain;
        use crate::model::openai::{Chat, ModelType};
        use std::env::var;

        #[tokio::test]
        async fn simple_chain() {
            let mut chain = Chain::default().model(Chat::new(
                "{{question}}".parse().unwrap(),
                ModelType::Gpt35Turbo,
                var("OPENAI_TOKEN").unwrap(),
                var("OPENAI_ORG").unwrap(),
            ));

            let new_chain = Chain::default().model(Chat::new(
                "{{question}}".parse().unwrap(),
                ModelType::Gpt35Turbo,
                var("OPENAI_TOKEN").unwrap(),
                var("OPENAI_ORG").unwrap(),
            ));

            chain.service::<Chain, _>(new_chain);
        }
    }
}

pub mod executor {
    use crate::protocol::{Error, Request, Response};
    use std::future::poll_fn;
    use tower_service::Service;

    pub async fn execute<S>(mut service: S, request: impl Into<Request>) -> Result<Response, Error>
    where
        S: Service<Request, Response = Response, Error = Error>,
    {
        poll_fn(|cx| service.poll_ready(cx)).await?;
        service.call(request.into()).await
    }

    #[cfg(test)]
    mod tests {
        use dotenv::var;

        #[tokio::test]
        async fn simple_chain() {
            use crate::chain::Chain;
            use crate::executor::execute;
            use crate::model::openai::{Chat, ModelType};

            let chain = Chain::default().model(Chat::new (
                "Generate 50 words of random text about {{question}}".parse().unwrap(),
                ModelType::Gpt35Turbo,
                var("OPENAI_TOKEN").unwrap(),
                var("ORGANIZATION_ID").unwrap(),
            ))
            .model(Chat::new (
                 "Given the following summary, output 10 questions that can be asked about it. {{ question }}".parse().unwrap(),
                 ModelType::Gpt35Turbo,
                 var("OPENAI_TOKEN").unwrap(),
                 var("ORGANIZATION_ID").unwrap(),
            ));

            dbg!(execute(chain, "ducks").await.unwrap());
        }
    }
}
