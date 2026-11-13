//! heads 的逐 token 解码与 Python 参考实现全等。
//!
//! fixtures 随机数固定种子，期望值由参考 Python 实现
//! （掩码 argmax、CRF Viterbi、topk、softmax）现场算出。

use kwja_heads::lora_multi_labeling::{
    WORD_FEATURE_BP_END, WORD_FEATURE_PH_END, closes_base_phrase, closes_phrase, multi_label_flags,
};
use kwja_heads::lora_relation_wise::{discourse_decode, topk_dim2};
use kwja_heads::sequence_labeling::{
    CrfParameters, argmax, get_morpheme_attribute_predictions, viterbi_decode,
};

fn load(file: &str) -> Vec<serde_json::Value> {
    let path = format!("{}/tests/fixtures/{}", env!("CARGO_MANIFEST_DIR"), file);
    let json = std::fs::read_to_string(path).unwrap();
    let cases: Vec<serde_json::Value> = serde_json::from_str(&json).unwrap();
    assert!(!cases.is_empty());
    cases
}

fn f32_vec(value: &serde_json::Value) -> Vec<f32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect()
}

fn f32_rows(value: &serde_json::Value) -> Vec<Vec<f32>> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            r.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect()
        })
        .collect()
}

fn f32_cube(value: &serde_json::Value) -> Vec<Vec<Vec<f32>>> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r.as_array().unwrap().iter().map(f32_vec).collect())
        .collect()
}

fn f32_tensor4(value: &serde_json::Value) -> Vec<Vec<Vec<Vec<f32>>>> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r.as_array().unwrap().iter().map(f32_rows).collect())
        .collect()
}

fn usize_vec(value: &serde_json::Value) -> Vec<usize> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect()
}

fn usize_rows(value: &serde_json::Value) -> Vec<Vec<usize>> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            r.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect()
        })
        .collect()
}

fn usize_cube(value: &serde_json::Value) -> Vec<Vec<Vec<usize>>> {
    value.as_array().unwrap().iter().map(usize_rows).collect()
}

#[test]
fn morpheme_attribute_predictions_match_python() {
    for case in load("morpheme_attr_cases.json") {
        let name = case["name"].as_str().unwrap();
        let rows = |key: &str| -> Vec<Vec<f32>> { f32_rows(&case[key]) };
        let got = get_morpheme_attribute_predictions(
            &rows("pos_logits"),
            &rows("subpos_logits"),
            &rows("conjtype_logits"),
            &rows("conjform_logits"),
        );
        let expected = &case["expected"];
        assert_eq!(got.pos, usize_vec(&expected["pos"]), "{name}: pos 不一致");
        assert_eq!(
            got.subpos,
            usize_vec(&expected["subpos"]),
            "{name}: subpos 不一致"
        );
        assert_eq!(
            got.conjtype,
            usize_vec(&expected["conjtype"]),
            "{name}: conjtype 不一致"
        );
        assert_eq!(
            got.conjform,
            usize_vec(&expected["conjform"]),
            "{name}: conjform 不一致"
        );
    }
}

#[test]
fn ne_viterbi_matches_python() {
    for case in load("ne_viterbi_cases.json") {
        let name = case["name"].as_str().unwrap();
        let emissions = f32_cube(&case["emissions"]);
        let mask: Vec<Vec<bool>> = case["mask"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| {
                r.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_bool().unwrap())
                    .collect()
            })
            .collect();
        let params = CrfParameters {
            start_transitions: f32_vec(&case["start_transitions"]),
            transitions: f32_rows(&case["transitions"]),
            end_transitions: f32_vec(&case["end_transitions"]),
            o_index: case["o_index"].as_u64().unwrap() as usize,
        };
        let got = viterbi_decode(&emissions, &mask, &params);
        assert_eq!(
            got,
            usize_rows(&case["expected"]),
            "{name}: viterbi 路径不一致"
        );
    }
}

#[test]
fn dependency_topk_matches_python() {
    for case in load("dependency_topk_cases.json") {
        let name = case["name"].as_str().unwrap();
        let got = topk_dim2(
            &f32_cube(&case["logits"]),
            case["k"].as_u64().unwrap() as usize,
        );
        assert_eq!(got, usize_cube(&case["expected"]), "{name}: topk 不一致");
    }
}

#[test]
fn discourse_decode_matches_python() {
    for case in load("discourse_cases.json") {
        let name = case["name"].as_str().unwrap();
        let got = discourse_decode(
            &f32_tensor4(&case["logits"]),
            case["threshold"].as_f64().unwrap() as f32,
            case["none_index"].as_u64().unwrap() as usize,
        );
        assert_eq!(
            got,
            usize_cube(&case["expected"]),
            "{name}: discourse 不一致"
        );
    }
}

#[test]
fn reading_argmax_matches_python() {
    for case in load("reading_argmax_cases.json") {
        let got: Vec<Vec<usize>> = f32_cube(&case["logits"])
            .iter()
            .map(|rows| rows.iter().map(|row| argmax(row)).collect())
            .collect();
        assert_eq!(got, usize_rows(&case["expected"]), "reading argmax 不一致");
    }
}

#[test]
fn multi_label_thresholds_match_python() {
    for case in load("multi_label_cases.json") {
        let name = case["name"].as_str().unwrap();
        let threshold = case["threshold"].as_f64().unwrap() as f32;
        let probs = f32_rows(&case["probs"]);
        let expected_flags = case["expected_flags"].as_array().unwrap();
        for (row, expected) in probs.iter().zip(expected_flags.iter()) {
            let flags: Vec<bool> = expected
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_bool().unwrap())
                .collect();
            assert_eq!(
                multi_label_flags(row, threshold),
                flags,
                "{name}: flags 不一致"
            );
        }
        if let Some(expected_bp) = case.get("expected_bp_close") {
            for (row, expected) in probs.iter().zip(expected_bp.as_array().unwrap().iter()) {
                assert_eq!(
                    closes_base_phrase(row[WORD_FEATURE_BP_END], row[WORD_FEATURE_PH_END]),
                    expected.as_bool().unwrap(),
                    "{name}: 基本句切分条件不一致"
                );
            }
        }
        if let Some(expected_ph) = case.get("expected_ph_close") {
            for (row, expected) in probs.iter().zip(expected_ph.as_array().unwrap().iter()) {
                assert_eq!(
                    closes_phrase(row[WORD_FEATURE_BP_END], row[WORD_FEATURE_PH_END]),
                    expected.as_bool().unwrap(),
                    "{name}: 文节切分条件不一致"
                );
            }
        }
    }
}
