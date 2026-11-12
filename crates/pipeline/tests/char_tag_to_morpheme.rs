//! 阶段 3 闸（端到端）：三头 logits → argmax 标签 → 分句/分词/表层归一化 →
//! Juman++ 行文本，与 Python（convert_char_predictions_into_tags ＋ flat_char 全链）
//! 逐字符、逐词素、逐字节全等。fixtures 由
//! 导出工具/导出Rust侧阶段3基准.py 现场算出。

use kwja_pipeline::char_tags::{
    argmax, convert_char_predictions_into_tags, sentence_jumanpp, set_morphemes_plain,
    set_sentences_plain,
};

#[test]
fn char_tags_end_to_end_match_python() {
    let json = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/char_tag_cases.json"
    ))
    .unwrap();
    let cases: Vec<serde_json::Value> = serde_json::from_str(&json).unwrap();
    assert!(!cases.is_empty());
    for case in &cases {
        let name = case["name"].as_str().unwrap();
        let text = case["text"].as_str().unwrap();
        let kwja_version = case["kwja_version"].as_str().unwrap();

        // 三个头的 argmax（numpy 语义：并列取首）
        let argmax_rows = |key: &str| -> Vec<usize> {
            case[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|row| {
                    let logits: Vec<f32> = row
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_f64().unwrap() as f32)
                        .collect();
                    argmax(&logits)
                })
                .collect()
        };
        let sent_predictions = argmax_rows("sent_logits");
        let word_predictions = argmax_rows("word_logits");
        let norm_predictions = argmax_rows("norm_logits");
        let indices: Vec<usize> = case["indices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();

        let (sent_t, word_t, norm_t) = convert_char_predictions_into_tags(
            &sent_predictions,
            &word_predictions,
            &norm_predictions,
            &indices,
        );

        // 分句
        let sentences = set_sentences_plain(text, &sent_t);
        let expected_sentences: Vec<String> = case["expected_sentences"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        assert_eq!(sentences, expected_sentences, "{name}: 分句不一致");

        // 分词＋归一化
        let morphemes = set_morphemes_plain(&sentences, &word_t, &norm_t);
        let expected_morphemes = case["expected_morphemes"].as_array().unwrap();
        assert_eq!(
            morphemes.len(),
            expected_morphemes.len(),
            "{name}: 句数不一致"
        );
        for (ms, ems) in morphemes.iter().zip(expected_morphemes.iter()) {
            let expected: Vec<(String, String)> = ems
                .as_array()
                .unwrap()
                .iter()
                .map(|m| {
                    (
                        m["surf"].as_str().unwrap().to_string(),
                        m["lemma"].as_str().unwrap().to_string(),
                    )
                })
                .collect();
            let got: Vec<(String, String)> = ms
                .iter()
                .map(|m| (m.surf.clone(), m.lemma.clone()))
                .collect();
            assert_eq!(got, expected, "{name}: 词素 surf/lemma 不一致");
        }

        // Juman++ 行文本（含 S-ID 注释行与 EOS，逐字节）
        let doc_id = case["doc_id"].as_str().unwrap();
        let start_sid = case["start_sid"].as_u64().unwrap();
        let mut juman = String::new();
        for (offset, ms) in morphemes.iter().enumerate() {
            juman.push_str(&sentence_jumanpp(
                &format!("# S-ID:{doc_id}-{} kwja:{kwja_version}", start_sid + offset as u64),
                ms,
            ));
        }
        assert_eq!(
            juman,
            case["expected_jumanpp"].as_str().unwrap(),
            "{name}: Juman++ 文本不一致"
        );
    }
}
