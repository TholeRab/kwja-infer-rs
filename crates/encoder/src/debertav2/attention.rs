use super::config::Config;
use super::feed_forward::DebertaV2SelfOutput;
use super::layers::StableDropout;
use super::build_relative_position;

use candle::{bail, Context, DType, Device, Module, Result, Tensor, D};
use candle_nn::VarBuilder;

struct XSoftmax {}

impl XSoftmax {
    pub fn apply(input: &Tensor, mask: &Tensor, dim: D, device: &Device) -> Result<Tensor> {
        // candle 没有逻辑非算子：这里用 mask < 1 生成反掩码，再以 where_cond 选极小值/清零。
        let mut rmask = mask.broadcast_as(input.shape())?.to_dtype(DType::F32)?;

        rmask = rmask
            .broadcast_lt(&Tensor::new(&[1.0_f32], device)?)?
            .to_dtype(DType::U8)?;

        let min_value_tensor = Tensor::new(&[f32::MIN], device)?.broadcast_as(input.shape())?;
        let mut output = rmask.where_cond(&min_value_tensor, input)?;

        output = candle_nn::ops::softmax(&output, dim)?;

        let t_zeroes = Tensor::new(&[0f32], device)?.broadcast_as(input.shape())?;
        output = rmask.where_cond(&t_zeroes, &output)?;

        Ok(output)
    }
}

pub struct DebertaV2DisentangledSelfAttention {
    config: Config,
    num_attention_heads: usize,
    query_proj: candle_nn::Linear,
    key_proj: candle_nn::Linear,
    value_proj: candle_nn::Linear,
    dropout: StableDropout,
    device: Device,
    relative_attention: bool,
    pos_dropout: Option<StableDropout>,
    position_buckets: isize,
    max_relative_positions: isize,
    pos_ebd_size: isize,
    share_att_key: bool,
    pos_key_proj: Option<candle_nn::Linear>,
    pos_query_proj: Option<candle_nn::Linear>,
}

impl DebertaV2DisentangledSelfAttention {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let config = config.clone();
        let vb = vb.clone();

        if !config
            .hidden_size
            .is_multiple_of(config.num_attention_heads)
        {
            return Err(candle::Error::Msg(format!(
                "The hidden size {} is not a multiple of the number of attention heads {}",
                config.hidden_size, config.num_attention_heads
            )));
        }

        let num_attention_heads = config.num_attention_heads;

        let attention_head_size = config
            .attention_head_size
            .unwrap_or(config.hidden_size / config.num_attention_heads);

        let all_head_size = num_attention_heads * attention_head_size;

        let query_proj = candle_nn::linear(config.hidden_size, all_head_size, vb.pp("query_proj"))?;
        let key_proj = candle_nn::linear(config.hidden_size, all_head_size, vb.pp("key_proj"))?;
        let value_proj = candle_nn::linear(config.hidden_size, all_head_size, vb.pp("value_proj"))?;

        let share_att_key = config.share_att_key.unwrap_or(false);
        let relative_attention = config.relative_attention;
        let mut max_relative_positions = config.max_relative_positions;

        let mut pos_ebd_size: isize = 0;
        let position_buckets = config.position_buckets.unwrap_or(-1);
        let mut pos_dropout: Option<StableDropout> = None;
        let mut pos_key_proj: Option<candle_nn::Linear> = None;
        let mut pos_query_proj: Option<candle_nn::Linear> = None;

        if relative_attention {
            if max_relative_positions < 1 {
                max_relative_positions = config.max_position_embeddings as isize;
            }
            pos_ebd_size = max_relative_positions;
            if position_buckets > 0 {
                pos_ebd_size = position_buckets
            }

            pos_dropout = Some(StableDropout::new(config.hidden_dropout_prob));

            if !share_att_key {
                if config.pos_att_type.iter().any(|s| s == "c2p") {
                    pos_key_proj = Some(candle_nn::linear(
                        config.hidden_size,
                        all_head_size,
                        vb.pp("pos_key_proj"),
                    )?);
                }
                if config.pos_att_type.iter().any(|s| s == "p2c") {
                    pos_query_proj = Some(candle_nn::linear(
                        config.hidden_size,
                        all_head_size,
                        vb.pp("pos_query_proj"),
                    )?);
                }
            }
        }

        let dropout = StableDropout::new(config.attention_probs_dropout_prob);
        let device = vb.device().clone();

        Ok(Self {
            config,
            num_attention_heads,
            query_proj,
            key_proj,
            value_proj,
            dropout,
            device,
            relative_attention,
            pos_dropout,
            position_buckets,
            max_relative_positions,
            pos_ebd_size,
            share_att_key,
            pos_key_proj,
            pos_query_proj,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
        query_states: Option<&Tensor>,
        relative_pos: Option<&Tensor>,
        rel_embeddings: Option<&Tensor>,
    ) -> Result<Tensor> {
        let query_states = match query_states {
            Some(qs) => qs,
            None => hidden_states,
        };

        let query_layer = self.transpose_for_scores(&self.query_proj.forward(query_states)?)?;
        let key_layer = self.transpose_for_scores(&self.key_proj.forward(query_states)?)?;
        let value_layer = self.transpose_for_scores(&self.value_proj.forward(query_states)?)?;

        let mut rel_att: Option<Tensor> = None;

        let mut scale_factor: usize = 1;

        if self.config.pos_att_type.iter().any(|s| s == "c2p") {
            scale_factor += 1;
        }

        if self.config.pos_att_type.iter().any(|s| s == "p2c") {
            scale_factor += 1;
        }

        let scale = {
            let q_size = query_layer.dim(D::Minus1)?;
            Tensor::new(&[(q_size * scale_factor) as f32], &self.device)?.sqrt()?
        };

        let mut attention_scores: Tensor = {
            let key_layer_transposed = key_layer.t()?;
            let div = key_layer_transposed
                .broadcast_div(scale.to_dtype(query_layer.dtype())?.as_ref())?;
            query_layer.matmul(&div)?
        };

        if self.relative_attention {
            if let Some(rel_embeddings) = rel_embeddings {
                let rel_embeddings = self
                    .pos_dropout
                    .as_ref()
                    .context("relative_attention requires pos_dropout")?
                    .forward(rel_embeddings)?;
                rel_att = Some(self.disentangled_attention_bias(
                    query_layer,
                    key_layer,
                    relative_pos,
                    rel_embeddings,
                    scale_factor,
                )?);
            }
        }

        if let Some(rel_att) = rel_att {
            attention_scores = attention_scores.broadcast_add(&rel_att)?;
        }

        attention_scores = attention_scores.reshape((
            (),
            self.num_attention_heads,
            attention_scores.dim(D::Minus2)?,
            attention_scores.dim(D::Minus1)?,
        ))?;

        let mut attention_probs =
            XSoftmax::apply(&attention_scores, attention_mask, D::Minus1, &self.device)?;

        attention_probs = self.dropout.forward(&attention_probs)?;

        let mut context_layer = attention_probs
            .reshape((
                (),
                attention_probs.dim(D::Minus2)?,
                attention_probs.dim(D::Minus1)?,
            ))?
            .matmul(&value_layer)?;

        context_layer = context_layer
            .reshape((
                (),
                self.num_attention_heads,
                context_layer.dim(D::Minus2)?,
                context_layer.dim(D::Minus1)?,
            ))?
            .permute((0, 2, 1, 3))?
            .contiguous()?;

        let dims = context_layer.dims();

        context_layer = match dims.len() {
            2 => context_layer.reshape(())?,
            3 => context_layer.reshape((dims[0], ()))?,
            4 => context_layer.reshape((dims[0], dims[1], ()))?,
            5 => context_layer.reshape((dims[0], dims[1], dims[2], ()))?,
            _ => {
                bail!(
                    "Invalid shape for DisentabgledSelfAttention context layer: {:?}",
                    dims
                )
            }
        };

        Ok(context_layer)
    }

    fn transpose_for_scores(&self, xs: &Tensor) -> Result<Tensor> {
        let dims = xs.dims().to_vec();
        match dims.len() {
            3 => {
                let reshaped = xs.reshape((dims[0], dims[1], self.num_attention_heads, ()))?;

                reshaped.transpose(1, 2)?.contiguous()?.reshape((
                    (),
                    reshaped.dim(1)?,
                    reshaped.dim(D::Minus1)?,
                ))
            }
            shape => {
                bail!("Invalid shape for transpose_for_scores. Expected 3 dimensions, got {shape}")
            }
        }
    }

    fn disentangled_attention_bias(
        &self,
        query_layer: Tensor,
        key_layer: Tensor,
        relative_pos: Option<&Tensor>,
        rel_embeddings: Tensor,
        scale_factor: usize,
    ) -> Result<Tensor> {
        let mut relative_pos = relative_pos.map_or(
            build_relative_position(
                query_layer.dim(D::Minus2)?,
                key_layer.dim(D::Minus2)?,
                &self.device,
                Some(self.position_buckets),
                Some(self.max_relative_positions),
            )?,
            |pos| pos.clone(),
        );

        relative_pos = match relative_pos.dims().len() {
            2 => relative_pos.unsqueeze(0)?.unsqueeze(0)?,
            3 => relative_pos.unsqueeze(1)?,
            other => {
                bail!("Relative position ids must be of dim 2 or 3 or 4. Got dim of size {other}")
            }
        };

        let att_span = self.pos_ebd_size;

        let rel_embeddings = rel_embeddings
            .narrow(0, 0, (att_span * 2) as usize)?
            .unsqueeze(0)?;

        let mut pos_query_layer: Option<Tensor> = None;
        let mut pos_key_layer: Option<Tensor> = None;

        let repeat_with = query_layer.dim(0)? / self.num_attention_heads;
        if self.share_att_key {
            pos_query_layer = Some(
                self.transpose_for_scores(&self.query_proj.forward(&rel_embeddings)?)?
                    .repeat(repeat_with)?,
            );

            pos_key_layer = Some(
                self.transpose_for_scores(&self.key_proj.forward(&rel_embeddings)?)?
                    .repeat(repeat_with)?,
            )
        } else {
            if self.config.pos_att_type.iter().any(|s| s == "c2p") {
                pos_key_layer = Some(
                    self.transpose_for_scores(
                        &self
                            .pos_key_proj
                            .as_ref()
                            .context(
                                "Need pos_key_proj when share_att_key is false or not specified",
                            )?
                            .forward(&rel_embeddings)?,
                    )?
                    .repeat(repeat_with)?,
                )
            }
            if self.config.pos_att_type.iter().any(|s| s == "p2c") {
                pos_query_layer = Some(self.transpose_for_scores(&self
                    .pos_query_proj
                    .as_ref()
                    .context("Need a pos_query_proj when share_att_key is false or not specified")?
                    .forward(&rel_embeddings)?)?.repeat(repeat_with)?)
            }
        }

        let mut score = Tensor::new(&[0 as f32], &self.device)?;

        if self.config.pos_att_type.iter().any(|s| s == "c2p") {
            let pos_key_layer = pos_key_layer.context("c2p without pos_key_layer")?;

            let scale = Tensor::new(
                &[(pos_key_layer.dim(D::Minus1)? * scale_factor) as f32],
                &self.device,
            )?
            .sqrt()?;

            let mut c2p_att = query_layer.matmul(&pos_key_layer.t()?)?;

            let c2p_pos = relative_pos
                .broadcast_add(&Tensor::new(&[att_span as i64], &self.device)?)?
                .clamp(0 as f32, (att_span * 2 - 1) as f32)?;

            c2p_att = c2p_att.gather(
                &c2p_pos
                    .squeeze(0)?
                    .expand(&[
                        query_layer.dim(0)?,
                        query_layer.dim(1)?,
                        relative_pos.dim(D::Minus1)?,
                    ])?
                    .contiguous()?,
                D::Minus1,
            )?;

            score = score.broadcast_add(
                &c2p_att.broadcast_div(scale.to_dtype(c2p_att.dtype())?.as_ref())?,
            )?;
        }

        if self.config.pos_att_type.iter().any(|s| s == "p2c") {
            let pos_query_layer = pos_query_layer.context("p2c without pos_key_layer")?;

            let scale = Tensor::new(
                &[(pos_query_layer.dim(D::Minus1)? * scale_factor) as f32],
                &self.device,
            )?
            .sqrt()?;

            let r_pos = {
                if key_layer.dim(D::Minus2)? != query_layer.dim(D::Minus2)? {
                    build_relative_position(
                        key_layer.dim(D::Minus2)?,
                        key_layer.dim(D::Minus2)?,
                        &self.device,
                        Some(self.position_buckets),
                        Some(self.max_relative_positions),
                    )?
                    .unsqueeze(0)?
                } else {
                    relative_pos
                }
            };

            let p2c_pos = r_pos
                .to_dtype(DType::F32)?
                .neg()?
                .broadcast_add(&Tensor::new(&[att_span as f32], &self.device)?)?
                .clamp(0f32, (att_span * 2 - 1) as f32)?;

            let p2c_att = key_layer
                .matmul(&pos_query_layer.t()?)?
                .gather(
                    &p2c_pos
                        .squeeze(0)?
                        .expand(&[
                            query_layer.dim(0)?,
                            key_layer.dim(D::Minus2)?,
                            key_layer.dim(D::Minus2)?,
                        ])?
                        .contiguous()?
                        .to_dtype(DType::U32)?,
                    D::Minus1,
                )?
                .t()?;

            score =
                score.broadcast_add(&p2c_att.broadcast_div(&scale.to_dtype(p2c_att.dtype())?)?)?;
        }

        Ok(score)
    }
}

pub struct DebertaV2Attention {
    dsa: DebertaV2DisentangledSelfAttention,
    output: DebertaV2SelfOutput,
}

impl DebertaV2Attention {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let dsa = DebertaV2DisentangledSelfAttention::load(vb.pp("attention.self"), config)?;
        let output = DebertaV2SelfOutput::load(vb.pp("attention.output"), config)?;
        Ok(Self { dsa, output })
    }

    pub(crate) fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
        query_states: Option<&Tensor>,
        relative_pos: Option<&Tensor>,
        rel_embeddings: Option<&Tensor>,
    ) -> Result<Tensor> {
        let self_output = self.dsa.forward(
            hidden_states,
            attention_mask,
            query_states,
            relative_pos,
            rel_embeddings,
        )?;

        self.output
            .forward(&self_output, query_states.unwrap_or(hidden_states))
    }
}
