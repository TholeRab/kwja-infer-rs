//! stride=2 分窗：SequenceSplitter 逐行移植，外加目标句判定。
//!
//! 必须照抄参考算法，不许自己写贪心：word 阶段 document_split_stride=2
//! 走的是 search_sub_sequence_span 的滑窗分支——每窗只前进 stride 句、
//! 窗口大量重叠；首窗的 stride 是本窗句数（end - prev_end），子文档 id
//! 据此带 -s21i0 这类后缀。窗数一错，目标句集合与之后各窗的 buffer 全错。

/// 一个候选窗的跨度信息（起点、终点、长度与跨步）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpanCandidate {
    pub stride: i64,
    pub length: usize,
    pub start: usize,
    pub end: usize,
}

/// SequenceSplitter：纯整数运算，逐行对照参考实现。
pub struct SequenceSplitter {
    sequence_lengths: Vec<usize>,
    max_length: usize,
    stride: i64,
    cumulative_lengths: Vec<usize>,
}

impl SequenceSplitter {
    pub fn new(sequence_lengths: Vec<usize>, max_length: usize, stride: i64) -> Self {
        let mut cumulative_lengths = Vec::with_capacity(sequence_lengths.len() + 1);
        cumulative_lengths.push(0usize);
        for length in &sequence_lengths {
            let last = cumulative_lengths[cumulative_lengths.len() - 1];
            cumulative_lengths.push(last + length);
        }
        Self {
            sequence_lengths,
            max_length,
            stride,
            cumulative_lengths,
        }
    }

    /// split_into_spans：生成器改成 Vec；终止条件同为 prev_end < len。
    pub fn split_into_spans(&self) -> Vec<SpanCandidate> {
        let mut spans: Vec<SpanCandidate> = Vec::new();
        let mut prev_start = 0usize;
        let mut prev_end = 0usize;
        while prev_end < self.sequence_lengths.len() {
            let (span, _) = self.search_sub_sequence_span(prev_start, prev_end);
            prev_start = span.start;
            prev_end = span.end;
            spans.push(span);
        }
        spans
    }

    /// search_sub_sequence_span：分支顺序照抄参考实现。
    fn search_sub_sequence_span(
        &self,
        prev_start: usize,
        prev_end: usize,
    ) -> (SpanCandidate, Vec<SpanCandidate>) {
        if self.stride == -1 {
            let start = prev_end;
            let end = self.search_end(start, prev_end + 1);
            let span = self.gen_span_candidate(start, end, (end - prev_end) as i64);
            return (span.clone(), vec![span]);
        }
        let mut candidates: Vec<SpanCandidate> = Vec::new();
        for start in prev_start..self.sequence_lengths.len() {
            if start > prev_end {
                return (self.choose_best_candidate(&candidates), candidates);
            }
            let end = self.search_end(start, prev_end + 1);
            let span = self.gen_span_candidate(start, end, (end - prev_end) as i64);
            candidates.push(span.clone());
            if span.length > self.max_length {
                // 长度条件不满足，换下一个起点
                continue;
            }
            if span.stride < self.stride {
                // stride 条件不满足，换下一个起点
                continue;
            }
            let mut chosen = span;
            if prev_end > 0 {
                // 非首窗：终点精确推进 stride 句
                let end = prev_end + self.stride as usize;
                chosen = self.gen_span_candidate(start, end, self.stride);
                candidates.push(chosen.clone());
            }
            return (chosen, candidates);
        }
        (self.choose_best_candidate(&candidates), candidates)
    }

    /// search_end：Python 原式 end + 1 <= len(...) 与 end < len(...) 整数等价。
    fn search_end(&self, start: usize, initial: usize) -> usize {
        let mut end = initial;
        while end < self.sequence_lengths.len()
            && self.sub_sequence_length(start, end + 1) <= self.max_length
        {
            end += 1;
        }
        end
    }

    fn gen_span_candidate(&self, start: usize, end: usize, stride: i64) -> SpanCandidate {
        SpanCandidate {
            stride,
            length: self.sub_sequence_length(start, end),
            start,
            end,
        }
    }

    /// 子序列长度：累计长度作差。
    fn sub_sequence_length(&self, start: usize, end: usize) -> usize {
        self.cumulative_lengths[end] - self.cumulative_lengths[start]
    }

    /// 候选择优：Python sorted(..., reverse=True)[0] 是稳定排序，并列时取先出现的，
    /// 因此这里用「严格大于才替换」，不能用 max_by（它取最后一个并列项）。
    fn choose_best_candidate(&self, candidates: &[SpanCandidate]) -> SpanCandidate {
        assert!(!candidates.is_empty(), "候选窗为空（上游会 IndexError）");
        let mut best = &candidates[0];
        let mut best_score = self.priority_score(best);
        for candidate in &candidates[1..] {
            let score = self.priority_score(candidate);
            if score > best_score {
                best = candidate;
                best_score = score;
            }
        }
        best.clone()
    }

    /// 候选优先级打分。
    fn priority_score(&self, span: &SpanCandidate) -> i64 {
        if span.length <= self.max_length {
            let weight = self.max_length as i64 + 1; // 任意更大的数都行
            ((span.stride == self.stride) as i64) * weight + span.length as i64
        } else {
            -(span.length as i64)
        }
    }
}

/// 目标句判定：doc_id 不带 -s{stride}i{idx} 后缀时全部是目标句；
/// 带后缀时，末尾 stride 句是目标句。
pub fn target_sentence_flags(doc_id: &str, n_sentences: usize) -> Vec<bool> {
    match sub_doc_pattern().captures(doc_id) {
        Some(caps) => {
            let stride: usize = caps[2].parse().expect("stride 应为非负整数");
            (0..n_sentences).map(|i| i + stride >= n_sentences).collect()
        }
        None => vec![true; n_sentences],
    }
}

/// 子文档 id 正则（与 char_stage::to_orig_doc_id 同源）。
fn sub_doc_pattern() -> &'static regex::Regex {
    static PAT: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    PAT.get_or_init(|| {
        regex::Regex::new(r"^([a-zA-Z\d\-_]+?)-s(\d+?)i(\d+?)$").expect("子文档 id 正则编译失败")
    })
}
