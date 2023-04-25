use super::client::OpenAIRequest;
use reqwest::Method;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct FineTuningRequest {
    /// The ID of an uploaded file that contains training data.
    /// Your dataset must be formatted as a JSONL file, where each training example is a JSON object with the keys "prompt" and "completion".
    /// Additionally, you must upload your file with the purpose fine-tune.
    pub training_file: String,

    /// The ID of an uploaded file that contains validation data.
    /// If you provide this file, the data is used to generate validation metrics periodically during fine-tuning.
    /// These metrics can be viewed in the fine-tuning results file.
    /// Your train and validation data should be mutually exclusive.
    /// Your dataset must be formatted as a JSONL file, where each validation example is a JSON object with the keys "prompt" and "completion".
    /// Additionally, you must upload your file with the purpose fine-tune.
    pub validation_file: Option<String>,

    /// The name of the base model to fine-tune.
    /// You can select one of "ada", "babbage", "curie", "davinci", or a fine-tuned model created after 2022-04-21.
    pub model: Option<String>,

    /// The number of epochs to train the model for.
    pub n_epochs: Option<u32>,

    /// The batch size to use for training.
    pub batch_size: Option<u32>,

    /// The learning rate multiplier to use for training.
    pub learning_rate_multiplier: Option<f64>,

    /// The weight to use for loss on the prompt tokens.
    pub prompt_loss_weight: Option<f64>,

    /// If set, we calculate classification-specific metrics such as accuracy and F-1 score using the validation set at the end of every epoch.
    /// These metrics can be viewed in the results file.
    /// In order to compute classification metrics, you must provide a validation_file.
    /// Additionally, you must specify classification_n_classes for multiclass classification or classification_positive_class for binary classification.
    pub compute_classification_metrics: Option<bool>,

    /// The number of classes in a classification task.
    /// This parameter is required for multiclass classification.
    pub classification_n_classes: Option<u32>,

    /// The positive class in binary classification.
    /// This parameter is needed to generate precision, recall, and F1 metrics when doing binary classification.
    pub classification_positive_class: Option<String>,

    /// If this is provided, we calculate F-beta scores at the specified beta values.
    /// The F-beta score is a generalization of F-1 score.
    /// This is only used for binary classification.
    pub classification_betas: Option<Vec<f64>>,

    /// A string of up to 40 characters that will be added to your fine-tuned model name.
    pub suffix: Option<String>,
}

impl OpenAIRequest for FineTuningRequest {
    type Response = FineTuneResponse;

    fn method() -> Method {
        Method::POST
    }

    fn url() -> &'static str {
        "https://api.openai.com/v1/fine-tunes"
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct FineTuneResponse {
    id: String,
    object: String,
    model: String,
    created_at: i64,
    events: Vec<FineTuneEvent>,
    fine_tuned_model: Option<String>,
    hyperparams: Hyperparams,
    organization_id: String,
    result_files: Vec<String>,
    status: String,
    validation_files: Vec<String>,
    training_files: Vec<TrainingFile>,
    updated_at: i64,
}

#[derive(Debug, Deserialize, Serialize)]
struct FineTuneEvent {
    object: String,
    created_at: i64,
    level: String,
    message: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct Hyperparams {
    batch_size: i32,
    learning_rate_multiplier: f64,
    n_epochs: i32,
    prompt_loss_weight: f64,
}

#[derive(Debug, Deserialize, Serialize)]
struct TrainingFile {
    id: String,
    object: String,
    bytes: i64,
    created_at: i64,
    filename: String,
    purpose: String,
}
