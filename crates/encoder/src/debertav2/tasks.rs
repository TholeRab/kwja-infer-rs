use std::collections::HashMap;

use candle::{Context, Device, Module, Result, Tensor, bail};
use candle_nn::VarBuilder;

use super::config::{Config, Id2Label};
use super::layers::{HiddenActLayer, StableDropout};
use super::model::DebertaV2Model;

#[derive(Debug)]
pub struct NERItem {
    pub entity: String,
    pub word: String,
    pub score: f32,
    pub start: usize,
    pub end: usize,
    pub index: usize,
}

#[derive(Debug)]
pub struct TextClassificationItem {
    pub label: String,
    pub score: f32,
}

pub struct DebertaV2NERModel {
    pub device: Device,
    deberta: DebertaV2Model,
    dropout: candle_nn::Dropout,
    classifier: candle_nn::Linear,
}

fn id2label_len(config: &Config, id2label: Option<HashMap<u32, String>>) -> Result<usize> {
    let id2label_len = match (&config.id2label, id2label) {
        (None, None) => bail!(
            "Id2Label is either not present in the model configuration or not passed into DebertaV2NERModel::load as a parameter"
        ),
        (None, Some(id2label_p)) => id2label_p.len(),
        (Some(id2label_c), None) => id2label_c.len(),
        (Some(id2label_c), Some(id2label_p)) => {
            if *id2label_c == id2label_p {
                id2label_c.len()
            } else {
                bail!(
                    "Id2Label is both present in the model configuration and provided as a parameter, and they are different."
                )
            }
        }
    };
    Ok(id2label_len)
}

impl DebertaV2NERModel {
    pub fn load(vb: VarBuilder, config: &Config, id2label: Option<Id2Label>) -> Result<Self> {
        let id2label_len = id2label_len(config, id2label)?;

        let deberta = DebertaV2Model::load(vb.clone(), config)?;
        let dropout = candle_nn::Dropout::new(config.hidden_dropout_prob as f32);
        let classifier: candle_nn::Linear = candle_nn::linear_no_bias(
            config.hidden_size,
            id2label_len,
            vb.root().pp("classifier"),
        )?;

        Ok(Self {
            device: vb.device().clone(),
            deberta,
            dropout,
            classifier,
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: Option<Tensor>,
        attention_mask: Option<Tensor>,
    ) -> Result<Tensor> {
        let output = self
            .deberta
            .forward(input_ids, token_type_ids, attention_mask, None)?;
        let output = self.dropout.forward(&output, false)?;
        self.classifier.forward(&output)
    }
}

pub struct DebertaV2SeqClassificationModel {
    pub device: Device,
    deberta: DebertaV2Model,
    dropout: StableDropout,
    pooler: DebertaV2ContextPooler,
    classifier: candle_nn::Linear,
}

impl DebertaV2SeqClassificationModel {
    pub fn load(vb: VarBuilder, config: &Config, id2label: Option<Id2Label>) -> Result<Self> {
        let id2label_len = id2label_len(config, id2label)?;
        let deberta = DebertaV2Model::load(vb.clone(), config)?;
        let pooler = DebertaV2ContextPooler::load(vb.clone(), config)?;
        let output_dim = pooler.output_dim()?;
        let classifier = candle_nn::linear(output_dim, id2label_len, vb.root().pp("classifier"))?;
        let dropout = match config.cls_dropout {
            Some(cls_dropout) => StableDropout::new(cls_dropout),
            None => StableDropout::new(config.hidden_dropout_prob),
        };

        Ok(Self {
            device: vb.device().clone(),
            deberta,
            dropout,
            pooler,
            classifier,
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: Option<Tensor>,
        attention_mask: Option<Tensor>,
    ) -> Result<Tensor> {
        let encoder_layer =
            self.deberta
                .forward(input_ids, token_type_ids, attention_mask, None)?;
        let pooled_output = self.pooler.forward(&encoder_layer)?;
        let pooled_output = self.dropout.forward(&pooled_output)?;
        self.classifier.forward(&pooled_output)
    }
}

pub struct DebertaV2ContextPooler {
    dense: candle_nn::Linear,
    dropout: StableDropout,
    config: Config,
}

impl DebertaV2ContextPooler {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let pooler_hidden_size = config
            .pooler_hidden_size
            .context("config.pooler_hidden_size is required for DebertaV2ContextPooler")?;

        let pooler_dropout = config
            .pooler_dropout
            .context("config.pooler_dropout is required for DebertaV2ContextPooler")?;

        let dense = candle_nn::linear(
            pooler_hidden_size,
            pooler_hidden_size,
            vb.root().pp("pooler.dense"),
        )?;

        let dropout = StableDropout::new(pooler_dropout);

        Ok(Self {
            dense,
            dropout,
            config: config.clone(),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let context_token = hidden_states.narrow(1, 0, 1)?.squeeze(1)?;
        let context_token = self.dropout.forward(&context_token)?;

        let pooled_output = self.dense.forward(&context_token.contiguous()?)?;
        let pooler_hidden_act = self
            .config
            .pooler_hidden_act
            .context("Could not obtain pooler hidden act from config")?;

        HiddenActLayer::new(pooler_hidden_act).forward(&pooled_output)
    }

    pub fn output_dim(&self) -> Result<usize> {
        self.config.pooler_hidden_size.context("DebertaV2ContextPooler cannot return output_dim (pooler_hidden_size) since it is not specified in the model config")
    }
}
