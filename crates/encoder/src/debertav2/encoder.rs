use super::config::Config;
use super::conv::ConvLayer;
use super::layer::DebertaV2Layer;
use super::build_relative_position;

use candle::{bail, Context, Device, Module, Result, Tensor, D};
use candle_nn::{embedding, layer_norm, Embedding, LayerNorm, VarBuilder};

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
