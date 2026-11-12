//! SequenceSplitter 单测（自 src/window.rs 迁出；硬规矩：src 只放实现）。
use kwja_pipeline::window::{SequenceSplitter, target_sentence_flags};

#[test]
fn splitter_first_window_takes_all_and_advances_by_stride() {
    // 每句 10 token、上限 246：首窗 _search_end 装到 24 句（24*10=240≤246），
    // 之后每窗只推进 stride=2 句（滑窗，不是「装不下才开新窗」）。
    let lens = vec![10usize; 30];
    let splitter = SequenceSplitter::new(lens, 246, 2);
    let spans = splitter.split_into_spans();
    assert_eq!(spans[0].start, 0);
    assert_eq!(spans[0].end, 24);
    assert_eq!(spans[0].stride, 24);
    assert_eq!(spans[1].start, 2);
    assert_eq!(spans[1].end, 26);
    assert_eq!(spans[1].stride, 2);
    // 末窗覆盖到 30
    assert_eq!(spans[spans.len() - 1].end, 30);
    // 窗与窗之间终点单调递增 2
    for w in spans.windows(2) {
        assert_eq!(w[1].end - w[0].end, 2);
    }
}

#[test]
fn target_flags_follow_doc_id_stride() {
    assert_eq!(target_sentence_flags("c-0", 5), vec![true; 5]);
    assert_eq!(
        target_sentence_flags("c-0-s2i3", 5),
        vec![false, false, false, true, true]
    );
    assert_eq!(
        target_sentence_flags("c-0-s1i9", 5),
        vec![false, false, false, false, true]
    );
}
