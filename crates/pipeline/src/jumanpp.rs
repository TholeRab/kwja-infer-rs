//! Juman++ 文本解析（阶段 6 落地；Python 侧 `utils/flat_juman.py` 的对应物）。
//!
//! word 阶段的输入是 char 阶段的产出：一句一段的 Juman++ 文本
//! （`# S-ID:...` 注释行 ＋ 词素行 ＋ `EOS`）。本模块把它切成句子，
//! 并取出分窗需要的最小字段：sid（取自注释行，不重编）与各词素 surf
//! （词素行首字段，按 rhoknp 的 `^([^ ]+| [^ ]*)` 取）。
//! 词素其余字段（读方／原形／活用／意味素性…）阶段 7 做 KNP 输出时再解析。

/// 一条词素行解析后的最小集：surf 供分窗计 token 数，raw 留给阶段 7。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MorphemeLine {
    pub surf: String,
    pub raw: String,
}

/// 一句的解析结果（`flat_juman.parse_sentence` 的最小集）。
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ParsedSentence {
    pub sid: String,
    pub morphemes: Vec<MorphemeLine>,
}

/// 按 EOS 切成「一句一段」的文本（`flat_juman.split_sentences`，
/// 对照 rhoknp utils/reader.py：空行丢弃，EOS 归入当前句）。
pub fn split_sentences(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut buf: Vec<&str> = Vec::new();
    for line in text.split('\n') {
        if line.trim() == "EOS" {
            buf.push(line);
            out.push(buf.join("\n") + "\n");
            buf.clear();
        } else if line.trim().is_empty() {
            continue;
        } else {
            buf.push(line);
        }
    }
    if !buf.is_empty() {
        out.push(buf.join("\n") + "\n");
    }
    out
}

/// 一段 Juman++ 文本（含注释行与 EOS）→ sid ＋ 词素行
/// （`flat_juman.parse_sentence`）。
///
/// 与 Python 同口径：注释行只保留最后一条；`@ ` 开头的同形行并入前一条词素；
/// 遇到 EOS 停止。
pub fn parse_sentence(text: &str) -> ParsedSentence {
    let mut comment = String::new();
    let mut morphemes: Vec<MorphemeLine> = Vec::new();
    let mut pending: Vec<&str> = Vec::new();
    for line in text.split('\n') {
        if line.trim().is_empty() {
            continue;
        }
        if line.starts_with('#') {
            // units/sentence.py:339-341：注释行赋给 comment（只保留最后一条）
            comment = line.to_string();
            continue;
        }
        if line.starts_with("@ ") {
            // morpheme.py:378-381 同形行：跟在某条词素行之后
            pending.push(line);
            continue;
        }
        if line.trim() == "EOS" {
            break;
        }
        if !pending.is_empty() {
            morphemes.push(flush(&pending));
        }
        pending = vec![line];
    }
    if !pending.is_empty() {
        morphemes.push(flush(&pending));
    }
    ParsedSentence {
        sid: sid_of_comment(&comment),
        morphemes,
    }
}

/// `_flush`：首行是词素本行，其余是 `@ ` 同形行（原文保留）。
fn flush(lines: &[&str]) -> MorphemeLine {
    let raw = lines[0].to_string();
    MorphemeLine {
        surf: unescape(surf_of_line(lines[0]).unwrap_or_default()),
        raw,
    }
}

/// rhoknp morpheme.py:44-49 的 surf 段：`^([^ ]+| [^ ]*)`。
/// 第二个分支处理「surf 本身以空格开头」的行（如半角空格词素）。
fn surf_of_line(line: &str) -> Option<String> {
    let bytes = line.as_bytes();
    if bytes.is_empty() {
        return None;
    }
    if bytes[0] == b' ' {
        // ` [^ ]*`：空格后到下一个空格（或行尾）为止
        let end = line[1..]
            .find(' ')
            .map(|i| i + 1)
            .unwrap_or(line.len());
        return Some(line[..end].to_string());
    }
    let end = line.find(' ').unwrap_or(line.len());
    Some(line[..end].to_string())
}

/// rhoknp morpheme.py:310-311 的反转义：`\t`→制表符、`\␣`→空格。
fn unescape(s: String) -> String {
    s.replace("\\t", "\t").replace("\\␣", " ")
}

/// `# S-ID:<sid> ...` → sid（`flat_juman.sid_of_comment`）。
fn sid_of_comment(comment: &str) -> String {
    let bytes = comment.as_bytes();
    // 手写 `^#\s*S-ID:(\S+)`：跳过 # 与空白，取 S-ID: 后到下一空白为止
    if !comment.starts_with('#') {
        return String::new();
    }
    let mut i = 1usize;
    while i < bytes.len() && (bytes[i] == b' ' || bytes[i] == b'\t') {
        i += 1;
    }
    let rest = &comment[i..];
    match rest.strip_prefix("S-ID:") {
        Some(after) => after.split_whitespace().next().unwrap_or("").to_string(),
        None => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_sentences_keeps_eos_and_drops_blank_lines() {
        let text = "# S-ID:a-1 kwja:2.6.0\nりんご _ りんご 名詞 6 普通名詞 1 * 0 * 0\nEOS\n\n# S-ID:a-2 kwja:2.6.0\nみかん _ みかん 名詞 6 普通名詞 1 * 0 * 0\nEOS\n";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 2);
        assert!(sentences[0].ends_with("EOS\n"));
        assert!(sentences[1].starts_with("# S-ID:a-2"));
    }

    #[test]
    fn parse_sentence_reads_sid_and_surfs() {
        let sent = "# S-ID:c-0-1 kwja:2.6.0\nさっき _ さっき 未定義語 15 その他 1 * 0 * 0\n、 _ 、 未定義語 15 その他 1 * 0 * 0\nEOS\n";
        let parsed = parse_sentence(sent);
        assert_eq!(parsed.sid, "c-0-1");
        assert_eq!(parsed.morphemes.len(), 2);
        assert_eq!(parsed.morphemes[0].surf, "さっき");
        assert_eq!(parsed.morphemes[1].surf, "、");
    }

    #[test]
    fn surf_of_line_handles_leading_space_surf() {
        assert_eq!(surf_of_line("␣ _ ␣ 未定義語 15 その他 1 * 0 * 0").unwrap(), "␣");
        // surf 为半角空格时，行首是空格：` [^ ]*` 分支
        assert_eq!(surf_of_line("  _   未定義語 15 その他 1 * 0 * 0").unwrap(), " ");
    }

    #[test]
    fn sid_of_comment_matches_python() {
        assert_eq!(sid_of_comment("# S-ID:c-58-214 kwja:2.6.0"), "c-58-214");
        assert_eq!(sid_of_comment("# 其他注释"), "");
        assert_eq!(sid_of_comment(""), "");
    }
}
