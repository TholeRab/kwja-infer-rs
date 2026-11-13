use candle::{D, DType, Device, Result, Tensor};

pub(crate) fn build_relative_position(
    query_size: usize,
    key_size: usize,
    device: &Device,
    bucket_size: Option<isize>,
    max_position: Option<isize>,
) -> Result<Tensor> {
    let q_ids = Tensor::arange(0, query_size as i64, device)?.unsqueeze(0)?;
    let k_ids: Tensor = Tensor::arange(0, key_size as i64, device)?.unsqueeze(D::Minus1)?;
    let mut rel_pos_ids = k_ids.broadcast_sub(&q_ids)?;
    let bucket_size = bucket_size.unwrap_or(-1);
    let max_position = max_position.unwrap_or(-1);

    if bucket_size > 0 && max_position > 0 {
        rel_pos_ids = make_log_bucket_position(rel_pos_ids, bucket_size, max_position, device)?;
    }

    rel_pos_ids = rel_pos_ids.to_dtype(DType::I64)?;
    rel_pos_ids = rel_pos_ids.narrow(0, 0, query_size)?;
    rel_pos_ids.unsqueeze(0)
}

pub(crate) fn make_log_bucket_position(
    relative_pos: Tensor,
    bucket_size: isize,
    max_position: isize,
    device: &Device,
) -> Result<Tensor> {
    let sign = relative_pos.to_dtype(DType::F32)?.sign()?;

    let mid = bucket_size / 2;

    let lt_mid = relative_pos.lt(mid as i64)?;
    let gt_neg_mid = relative_pos.gt(-mid as i64)?;

    let condition = lt_mid
        .to_dtype(candle::DType::F32)?
        .mul(&gt_neg_mid.to_dtype(candle::DType::F32)?)?
        .to_dtype(DType::U8)?;

    let on_true = Tensor::new(&[(mid - 1) as u32], device)?
        .broadcast_as(relative_pos.shape())?
        .to_dtype(relative_pos.dtype())?;

    let on_false = relative_pos
        .to_dtype(DType::F32)?
        .abs()?
        .to_dtype(DType::I64)?;

    let abs_pos = condition.where_cond(&on_true, &on_false)?;

    let mid_as_tensor = Tensor::from_slice(&[mid as f32], (1,), device)?;

    let log_pos = {
        let first_log = abs_pos
            .to_dtype(DType::F32)?
            .broadcast_div(&mid_as_tensor)?
            .log()?;

        let second_log =
            Tensor::from_slice(&[((max_position as f32 - 1.0) / mid as f32)], (1,), device)?
                .log()?;

        let first_div_second = first_log.broadcast_div(&second_log)?;

        let to_ceil = first_div_second
            .broadcast_mul(Tensor::from_slice(&[(mid - 1) as f32], (1,), device)?.as_ref())?;

        let ceil = to_ceil.ceil()?;

        ceil.broadcast_add(&mid_as_tensor)?
    };

    Ok({
        let abs_pos_lte_mid = abs_pos.to_dtype(DType::F32)?.broadcast_le(&mid_as_tensor)?;
        let relative_pos = relative_pos.to_dtype(relative_pos.dtype())?;
        let log_pos_mul_sign = log_pos.broadcast_mul(&sign.to_dtype(DType::F32)?)?;
        abs_pos_lte_mid.where_cond(&relative_pos.to_dtype(DType::F32)?, &log_pos_mul_sign)?
    })
}
