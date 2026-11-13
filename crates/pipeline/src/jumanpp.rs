//! Juman++ 文本解析：把 char 阶段产出的 Juman++ 文本切成句子并取最小字段。
//!
//! word 阶段的输入是 char 阶段的产出：一句一段的 Juman++ 文本
//! （`# S-ID:...` 注释行加词素行加 `EOS`）。本模块把它切成句子，
//! 并取出分窗需要的最小字段：sid（取自注释行，不重编）与各词素 surf
//! （词素行首字段）。
//! 词素其余字段（读方、原形、活用、意味素性等）在拼 KNP 输出时再解析。

/// 一条词素行解析后的最小集：surf/lemma 供分窗与 KNP 组装，raw 留档。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MorphemeLine {
    pub surf: String,
    pub lemma: String,
    pub raw: String,
}

/// 一句的解析结果。
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ParsedSentence {
    pub sid: String,
    pub morphemes: Vec<MorphemeLine>,
}

/// 按 EOS 切成一句一段的文本：空行丢弃，EOS 归入当前句。
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

/// 解析一段 Juman++ 文本（含注释行与 EOS），得到 sid 与词素行。
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
            // 注释行赋给 comment（只保留最后一条）
            comment = line.to_string();
            continue;
        }
        if line.starts_with("@ ") {
            // 同形行：跟在某条词素行之后
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

/// 首行是词素本行，其余是 @ 开头的同形行（原文保留）。
fn flush(lines: &[&str]) -> MorphemeLine {
    let raw = lines[0].to_string();
    MorphemeLine {
        surf: unescape(surf_of_line(lines[0]).unwrap_or_default()),
        lemma: unescape(lemma_of_line(lines[0]).unwrap_or_default()),
        raw,
    }
}

/// 取词素行的 surf 段。
/// 第二个分支处理「surf 本身以空格开头」的行（如半角空格词素）。
pub fn surf_of_line(line: &str) -> Option<String> {
    let bytes = line.as_bytes();
    if bytes.is_empty() {
        return None;
    }
    if bytes[0] == b' ' {
        // ` [^ ]*`：空格后到下一个空格（或行尾）为止
        let end = line[1..].find(' ').map(|i| i + 1).unwrap_or(line.len());
        return Some(line[..end].to_string());
    }
    let end = line.find(' ').unwrap_or(line.len());
    Some(line[..end].to_string())
}

/// 词素行第 3 段（lemma）。char 阶段产出里空格已转义成 `\␣`，
/// 段内不会有裸空格，按「跳过 surf、reading 两段」取第三段即可。
pub fn lemma_of_line(line: &str) -> Option<String> {
    let bytes = line.as_bytes();
    if bytes.is_empty() {
        return None;
    }
    // surf 段宽度与 surf_of_line 相同
    let mut pos = if bytes[0] == b' ' {
        line[1..].find(' ').map(|i| i + 1).unwrap_or(line.len())
    } else {
        line.find(' ').unwrap_or(line.len())
    };
    if pos >= line.len() {
        return None;
    }
    let rest = line[pos..].trim_start_matches(' ');
    // reading 段
    pos = rest.find(' ').unwrap_or(rest.len());
    if pos >= rest.len() {
        return None;
    }
    let rest2 = rest[pos..].trim_start_matches(' ');
    // lemma 段
    let end = rest2.find(' ').unwrap_or(rest2.len());
    Some(rest2[..end].to_string())
}

/// 反转义：\t 还原成制表符，\␣ 还原成空格。
fn unescape(s: String) -> String {
    s.replace("\\t", "\t").replace("\\␣", " ")
}

/// 从 # S-ID 注释行取出 sid。
pub fn sid_of_comment(comment: &str) -> String {
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
