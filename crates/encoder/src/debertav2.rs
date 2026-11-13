//! 基于 candle-transformers 0.9.2 的 debertav2.rs 改造：加入 KWJA 特有的
//! special_token_indices 相对位置扩展——special token 参与的 (query, key) 组，
//! 相对位置改用 -position_buckets，对角线保持 0。
//! 结构与 transformers 的 DeBERTa-v2 参考实现逐块对应。

use std::collections::HashMap;

use candle::{bail, Context, DType, Device, Module, Result, Tensor, D};
use candle_nn::{
    conv1d, embedding, layer_norm, Conv1d, Conv1dConfig, Embedding, LayerNorm, VarBuilder,
};
use serde::{Deserialize, Deserializer};

mod relpos;
pub(crate) use relpos::{build_relative_position, make_log_bucket_position};

mod config;
pub use config::{Config, DTYPE, HiddenAct, Id2Label, Label2Id};

mod layers;
pub use layers::{HiddenActLayer, StableDropout};

mod embeddings;
pub use embeddings::DebertaV2Embeddings;

mod conv;
pub use conv::ConvLayer;

mod feed_forward;
pub use feed_forward::{DebertaV2Intermediate, DebertaV2Output, DebertaV2SelfOutput};

mod attention;
pub use attention::{DebertaV2Attention, DebertaV2DisentangledSelfAttention};

mod layer;
pub use layer::DebertaV2Layer;

pub struct DebertaV2Encoder {
    layer: Vec<DebertaV2Layer>,
    relative_attention: bool,
    max_relative_positions: isize,
    position_buckets: isize,
    rel_embeddings: Option<Embedding>,
    norm_rel_ebd: String,
    layer_norm: Option<LayerNorm>,
    conv: Option<ConvLayer>,
    device: Device,
}

impl DebertaV2Encoder {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let layer = (0..config.num_hidden_layers)
            .map(|index| DebertaV2Layer::load(vb.pp(format!("layer.{index}")), config))
            .collect::<Result<Vec<_>>>()?;

        let relative_attention = config.relative_attention;
        let mut max_relative_positions = config.max_relative_positions;

        let position_buckets = config.position_buckets.unwrap_or(-1);

        let mut rel_embeddings: Option<Embedding> = None;

        if relative_attention {
            if max_relative_positions < 1 {
                max_relative_positions = config.max_position_embeddings as isize;
            }

            let mut pos_ebd_size = max_relative_positions * 2;

            if position_buckets > 0 {
                pos_ebd_size = position_buckets * 2;
            }

            rel_embeddings = Some(embedding(
                pos_ebd_size as usize,
                config.hidden_size,
                vb.pp("rel_embeddings"),
            )?);
        }

        // Python 实现假定 norm_rel_ebd 是数组，常见配置里实际是字符串；
        // 这里按字符串解析，等于 layer_norm 时给 rel_embeddings 配 LayerNorm。
        let norm_rel_ebd = match config.norm_rel_ebd.as_ref() {
            Some(nre) => nre.trim().to_string(),
            None => "none".to_string(),
        };

        let layer_norm: Option<LayerNorm> = if norm_rel_ebd == "layer_norm" {
            Some(layer_norm(
                config.hidden_size,
                config.layer_norm_eps,
                vb.pp("LayerNorm"),
            )?)
        } else {
            None
        };

        let conv: Option<ConvLayer> = if config.conv_kernel_size.unwrap_or(0) > 0 {
            Some(ConvLayer::load(vb.pp("conv"), config)?)
        } else {
            None
        };

        Ok(Self {
            layer,
            relative_attention,
            max_relative_positions,
            position_buckets,
            rel_embeddings,
            norm_rel_ebd,
            layer_norm,
            conv,
            device: vb.device().clone(),
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
        query_states: Option<&Tensor>,
        relative_pos: Option<&Tensor>,
        special_token_indices: Option<&Tensor>,
    ) -> Result<Tensor> {
        let input_mask = if attention_mask.dims().len() <= 2 {
            attention_mask.clone()
        } else {
            attention_mask
                .sum_keepdim(attention_mask.rank() - 2)?
                .gt(0.)?
        };

        let attention_mask = self.get_attention_mask(attention_mask.clone())?;

        let relative_pos =
            self.get_rel_pos(hidden_states, query_states, relative_pos, special_token_indices)?;

        let mut next_kv: Tensor = hidden_states.clone();
        let rel_embeddings = self.get_rel_embedding()?;
        let mut output_states = next_kv.to_owned();
        let mut query_states: Option<Tensor> = query_states.cloned();

        for (i, layer_module) in self.layer.iter().enumerate() {
            // 只实现推理分支；训练/微调分支不在本实现内。

            output_states = layer_module.forward(
                next_kv.as_ref(),
                &attention_mask,
                query_states.as_ref(),
                relative_pos.as_ref(),
                rel_embeddings.as_ref(),
            )?;

            if i == 0 {
                if let Some(conv) = &self.conv {
                    output_states = conv.forward(hidden_states, &output_states, &input_mask)?;
                }
            }

            if query_states.is_some() {
                query_states = Some(output_states.clone());
            } else {
                next_kv = output_states.clone();
            }
        }

        Ok(output_states)
    }

    fn get_attention_mask(&self, mut attention_mask: Tensor) -> Result<Tensor> {
        match attention_mask.dims().len() {
            0..=2 => {
                let extended_attention_mask = attention_mask.unsqueeze(1)?.unsqueeze(2)?;
                attention_mask = extended_attention_mask.broadcast_mul(
                    &extended_attention_mask
                        .squeeze(D::Minus2)?
                        .unsqueeze(D::Minus1)?,
                )?;
            }
            3 => attention_mask = attention_mask.unsqueeze(1)?,
            len => bail!("Unsupported attentiom mask size length: {len}"),
        }

        Ok(attention_mask)
    }

    fn get_rel_pos(
        &self,
        hidden_states: &Tensor,
        query_states: Option<&Tensor>,
        relative_pos: Option<&Tensor>,
        special_token_indices: Option<&Tensor>,
    ) -> Result<Option<Tensor>> {
        if self.relative_attention && relative_pos.is_none() {
            let q = if let Some(query_states) = query_states {
                query_states.dim(D::Minus2)?
            } else {
                hidden_states.dim(D::Minus2)?
            };
            let k = hidden_states.dim(D::Minus2)?;
            let rel_pos = build_relative_position(
                q,
                k,
                &self.device,
                Some(self.position_buckets),
                Some(self.max_relative_positions),
            )?;

            // KWJA 扩展：special token 参与的 (query, key) 组，相对位置改成
            // -position_buckets（对角线保持 0），其余组用原分桶值。
            if let Some(specials) = special_token_indices {
                let batch = hidden_states.dim(0)?;
                let q_ids = Tensor::arange(0, q as i64, &self.device)?; // (q,)
                let k_ids = Tensor::arange(0, k as i64, &self.device)?; // (k,)
                // candle 的逐元素比较不隐式广播，先显式 broadcast 成同形再比。
                let row_special = specials
                    .unsqueeze(2)? // (b,7,1)
                    .broadcast_as((batch, 7, q))?
                    .eq(&q_ids.broadcast_as((batch, 7, q))?)? // (b,7,q)
                    .sum(1)? // (b,q)（candle 的 sum 会把维去掉）
                    .unsqueeze(2)?; // (b,q,1)
                let col_special = specials
                    .unsqueeze(1)? // (b,1,7)
                    .broadcast_as((batch, k, 7))?
                    .eq(&k_ids.unsqueeze(0)?.unsqueeze(2)?.broadcast_as((batch, k, 7))?)? // (b,k,7)
                    .sum(2)? // (b,k)
                    .unsqueeze(2)? // (b,k,1)
                    .transpose(1, 2)?; // (b,1,k)
                let mask = row_special.broadcast_add(&col_special)?.gt(0u8)?; // (b,q,k)
                let off_diagonal = k_ids
                    .unsqueeze(0)? // (1,k)
                    .broadcast_sub(&q_ids.unsqueeze(1)?)? // (q,k)
                    .abs()? // (q,k)
                    .gt(0u8)? // (q,k)
                    .unsqueeze(0)?; // (1,q,k)
                let neg = Tensor::new(&[-self.position_buckets as i64], &self.device)?
                    .broadcast_as((1, q, k))?;
                let zero = Tensor::new(&[0i64], &self.device)?.broadcast_as((1, q, k))?;
                let special_token_pos = off_diagonal.where_cond(&neg, &zero)?; // (1,q,k)
                let on_true = special_token_pos.broadcast_as((batch, q, k))?;
                let on_false = rel_pos.broadcast_as((batch, q, k))?;
                return Ok(Some(mask.where_cond(&on_true, &on_false)?));
            }

            return Ok(Some(rel_pos));
        }

        if relative_pos.is_some() {
            Ok(relative_pos.cloned())
        } else {
            Ok(None)
        }
    }
    fn get_rel_embedding(&self) -> Result<Option<Tensor>> {
        if !self.relative_attention {
            return Ok(None);
        }

        let rel_embeddings = self
            .rel_embeddings
            .as_ref()
            .context("self.rel_embeddings not present when using relative_attention")?
            .embeddings()
            .clone();

        if !self.norm_rel_ebd.contains("layer_norm") {
            return Ok(Some(rel_embeddings));
        }

        let layer_normed_embeddings = self
            .layer_norm
            .as_ref()
            .context("DebertaV2Encoder layer_norm is None when norm_rel_ebd contains layer_norm")?
            .forward(&rel_embeddings)?;

        Ok(Some(layer_normed_embeddings))
    }
}

pub struct DebertaV2Model {
    embeddings: DebertaV2Embeddings,
    encoder: DebertaV2Encoder,
    z_steps: usize,
    pub device: Device,
}

impl DebertaV2Model {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let vb = vb.clone();
        let embeddings = DebertaV2Embeddings::load(vb.pp("embeddings"), config)?;
        let encoder = DebertaV2Encoder::load(vb.pp("encoder"), config)?;
        let z_steps: usize = 0;

        Ok(Self {
            embeddings,
            encoder,
            z_steps,
            device: vb.device().clone(),
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: Option<Tensor>,
        attention_mask: Option<Tensor>,
        special_token_indices: Option<&Tensor>,
    ) -> Result<Tensor> {
        let input_ids_shape = input_ids.shape();

        let attention_mask = match attention_mask {
            Some(mask) => mask,
            None => Tensor::ones(input_ids_shape, DType::I64, &self.device)?,
        };

        let token_type_ids = match token_type_ids {
            Some(ids) => ids,
            None => Tensor::zeros(input_ids_shape, DType::U32, &self.device)?,
        };

        let embedding_output = self.embeddings.forward(
            Some(input_ids),
            Some(&token_type_ids),
            None,
            Some(&attention_mask),
            None,
        )?;

        let encoder_output = self.encoder.forward(
            &embedding_output,
            &attention_mask,
            None,
            None,
            special_token_indices,
        )?;

        if self.z_steps > 1 {
            todo!("Complete DebertaV2Model forward() when z_steps > 1 -- Needs a model to test this situation.")
        }

        Ok(encoder_output)
    }
}

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
        (None, None) => bail!("Id2Label is either not present in the model configuration or not passed into DebertaV2NERModel::load as a parameter"),
        (None, Some(id2label_p)) => id2label_p.len(),
        (Some(id2label_c), None) => id2label_c.len(),
        (Some(id2label_c), Some(id2label_p)) => {
          if *id2label_c == id2label_p {
            id2label_c.len()
          } else {
            bail!("Id2Label is both present in the model configuration and provided as a parameter, and they are different.")
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
        let encoder_layer = self
            .deberta
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



