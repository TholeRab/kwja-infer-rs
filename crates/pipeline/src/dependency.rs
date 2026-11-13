//! 依存解析：DependencyManager 图结构与逐基本句接根算法。
//!
//! `DependencyManager` 是纯图结构：add_edge/remove_edge/has_cycle（DFS 找环）。
//! `add_dependency` 对每个基本句按 topk 候选依次试接，成环或已有根时回退，
//! 全部失败走 `_resolve`（先后向、再前向找第一个可接的点）。

use std::collections::{HashMap, HashSet};

/// 依存类型表 0 号必须是 D（普通依存）。
pub const DEPENDENCY_TYPE_D: usize = 0;

/// 依存边集合管理器（DFS 判环）。
#[derive(Debug, Default)]
pub struct DependencyManager {
    directed_graph: HashMap<usize, HashSet<usize>>,
    pub root: bool,
}

impl DependencyManager {
    pub fn add_edge(&mut self, source: usize, target: i32) {
        self.directed_graph.entry(source).or_default().insert(target as usize);
    }

    pub fn remove_edge(&mut self, source: usize, target: i32) {
        if let Some(set) = self.directed_graph.get_mut(&source) {
            set.remove(&(target as usize));
        }
    }

    fn is_cyclic(&self, source: usize, visited: &mut HashSet<usize>, cache: &mut HashMap<usize, bool>) -> bool {
        if let Some(&cached) = cache.get(&source) {
            return cached;
        }
        if visited.contains(&source) {
            return true;
        }
        visited.insert(source);
        let mut ret = false;
        if let Some(targets) = self.directed_graph.get(&source) {
            for &target in targets {
                if self.is_cyclic(target, visited, cache) {
                    ret = true;
                    break;
                }
            }
        }
        cache.insert(source, ret);
        ret
    }

    pub fn has_cycle(&self) -> bool {
        let mut visited = HashSet::new();
        let mut cache = HashMap::new();
        let sources: Vec<usize> = self.directed_graph.keys().copied().collect();
        for source in sources {
            if self.is_cyclic(source, &mut visited, &mut cache) {
                return true;
            }
        }
        false
    }
}

/// 一条依存边候选：母节的全局词素下标 + 依存类型下标。
#[derive(Debug, Clone, Copy)]
pub struct DepCandidate {
    pub parent_gi: usize,
    pub dep_type_idx: usize,
}

/// 逐基本句接依存边；参数与返回值见下。
///
/// `bps`：一句的基本句列表（ms 为窗内全局下标）；`head_gi_of_bp`：各基本句
/// 主辞的全局词素下标；`dep_preds[head_gi]`：该主辞的 topk 候选（母节 gi、类型）。
/// `root_gi`：[ROOT] 特殊词素的窗内全局下标。
///
/// 返回 (parents, dtypes)：各基本句的母节下标（句内序号，-1 = 根）与类型下标。
pub fn add_dependency(
    bps: &[&crate::chunking::BasePhrase],
    dep_preds: &[Vec<DepCandidate>],
    head_gi_of_bp: &[usize],
    root_gi: usize,
    num_dependency_types: usize,
) -> (Vec<i32>, Vec<usize>) {
    let n = bps.len();
    let mut gi2bp: HashMap<usize, i32> = HashMap::new();
    for (bi, bp) in bps.iter().enumerate() {
        for &mi in &bp.ms {
            gi2bp.insert(mi, bi as i32);
        }
    }
    gi2bp.insert(root_gi, -1);
    let mut mgr = DependencyManager::default();
    let mut parents: Vec<Option<i32>> = vec![None; n];
    let mut dtypes: Vec<Option<usize>> = vec![None; n];

    for bi in 0..n {
        let mut made = false;
        let head_gi = head_gi_of_bp[bi];
        let cands = dep_preds.get(head_gi).map_or(&[][..], |v| v.as_slice());
        for cand in cands {
            // 参考实现是严格查表；这里 .get 跳过候选里不在本句的母节
            // （含 [ROOT] 之外的 special，以及跨句词素——参考实现的 mask 已挡掉大多数）。
            let Some(&pidx) = gi2bp.get(&cand.parent_gi) else {
                continue;
            };
            mgr.add_edge(bi, pidx);
            if mgr.has_cycle() || (pidx == -1 && mgr.root) {
                mgr.remove_edge(bi, pidx);
            } else {
                parents[bi] = Some(pidx);
                dtypes[bi] = Some(cand.dep_type_idx.min(num_dependency_types - 1));
                made = true;
                break;
            }
        }
        if !made {
            if bi == n - 1 && !mgr.root {
                parents[bi] = Some(-1);
                dtypes[bi] = Some(DEPENDENCY_TYPE_D);
            } else {
                resolve(bi, n, &mut mgr, &mut parents, &mut dtypes);
            }
        }
        assert!(parents[bi].is_some(), "依存解析留空（bp {bi}）");
        if parents[bi] == Some(-1) {
            mgr.root = true;
        }
    }
    (
        parents.into_iter().map(|p| p.unwrap()).collect(),
        dtypes.into_iter().map(|d| d.unwrap()).collect(),
    )
}

/// 兜底接根：先后向（source+1..n）再前向（source-1..0）找第一个不成环的接点，类型固定 D。
fn resolve(
    source: usize,
    n: usize,
    mgr: &mut DependencyManager,
    parents: &mut [Option<i32>],
    dtypes: &mut [Option<usize>],
) {
    for target in source + 1..n {
        mgr.add_edge(source, target as i32);
        if mgr.has_cycle() {
            mgr.remove_edge(source, target as i32);
        } else {
            parents[source] = Some(target as i32);
            dtypes[source] = Some(DEPENDENCY_TYPE_D);
            return;
        }
    }
    for target in (0..source).rev() {
        mgr.add_edge(source, target as i32);
        if mgr.has_cycle() {
            mgr.remove_edge(source, target as i32);
        } else {
            parents[source] = Some(target as i32);
            dtypes[source] = Some(DEPENDENCY_TYPE_D);
            return;
        }
    }
    panic!("couldn't resolve dependency");
}
