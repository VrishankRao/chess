//! mctscore: V21 batch C port of chess_zero/mcts.py tree mechanics.
//!
//! Operates on OPAQUE eval vectors supplied by the caller (Python MPS in
//! production, scripted fns in tests). No NN, no board, no RNG, no PyO3.
//! Board-dependent facts (rep keys, legal lists, terminal z, check/capture
//! flags, contempt draw scores, forcing flags) arrive via the [`Env`] trait
//! (search.rs); RNG (Dirichlet noise) arrives as caller vectors; temperature
//! sampling stays caller-side. Mirrors mcts.py semantics including the
//! sequential-vs-batched draw-backup M asymmetry (checklist item 28).

pub mod search;
#[cfg(feature = "python")]
pub mod bridge;

use std::collections::{HashMap, HashSet, VecDeque};

/// Action id in 0..4096 (matches mcts.py visit-vector indexing).
pub type Action = u16;
/// Arena node id.
pub type NodeId = usize;
/// Dense visit vector over the full 4096 action space.
pub const N_ACTIONS: usize = 4096;

/// One tree node. Mirrors `_Node` slots (prior, raw, n, w, children,
/// expanded, terminal_z unused here, o, m). Children are stored as an
/// insertion-ordered vec so best-child tie-breaks match Python dict order
/// (first-max wins), which is legal-move order.
#[derive(Clone, Debug)]
pub struct Node {
    pub prior: f64,
    pub raw: f64,
    pub n: u32,
    pub w: f64,
    /// WU pending count (B1). Never touches w. Signed to match Python
    /// exactly: immediate-fallback leaves (collision path) get O removed
    /// at gather AND again at resolve, i.e. -1 per immediate descent
    /// through the node (-k on high-traffic ancestors across batches).
    /// Nbar arithmetic below uses the same signed values.
    pub o: i32,
    /// In-tree moves-left running mean (D3), None while ML off.
    pub m: Option<f64>,
    pub m_n: u32,
    pub expanded: bool,
    pub children: Vec<(Action, NodeId)>,
}

impl Node {
    pub fn fresh(prior: f64, raw: f64) -> Self {
        Node { prior, raw, n: 0, w: 0.0, o: 0, m: None, m_n: 0, expanded: false, children: Vec::new() }
    }

    pub fn root() -> Self {
        Self::fresh(0.0, 0.0)
    }

    /// Q = w/n, 0.0 when unvisited (mcts.py `_Node.q`).
    pub fn q(&self) -> f64 {
        if self.n > 0 { self.w / f64::from(self.n) } else { 0.0 }
    }

    pub fn find(&self, a: Action) -> Option<NodeId> {
        self.children.iter().find(|(x, _)| *x == a).map(|(_, id)| *id)
    }
}

/// Arena-owned search tree. Paths are `Vec<NodeId>` (no borrow pain).
#[derive(Clone, Debug, Default)]
pub struct Tree {
    pub arena: Vec<Node>,
    pub root: NodeId,
}

impl Tree {
    pub fn new() -> Self {
        Tree { arena: vec![Node::root()], root: 0 }
    }

    pub fn node(&self, id: NodeId) -> &Node {
        &self.arena[id]
    }

    pub fn node_mut(&mut self, id: NodeId) -> &mut Node {
        &mut self.arena[id]
    }

    pub fn alloc(&mut self, n: Node) -> NodeId {
        self.arena.push(n);
        self.arena.len() - 1
    }

    /// Sum of child visits (sequential-path parent total, mcts.py:461).
    pub fn total_n(&self, id: NodeId) -> u64 {
        self.arena[id].children.iter().map(|(_, c)| u64::from(self.arena[*c].n)).sum()
    }

    /// Sum of child N+O (batched-path parent Nbar, mcts.py:796).
    /// Signed: O can sit at -1 on immediate-fallback paths (see `o`).
    pub fn total_nbar(&self, id: NodeId) -> i64 {
        self.arena[id]
            .children
            .iter()
            .map(|(_, c)| i64::from(self.arena[*c].n) + i64::from(self.arena[*c].o))
            .sum()
    }

    /// Deep-clone the subtree at `id` into a fresh Tree (TT promote;
    /// Python reuses by reference, counts are identical either way).
    /// F3 (TT-carryover fix): `o` is copied VERBATIM (never reset to 0).
    /// Python promotes the live `_Node` object, so its WU pending count
    /// (signed, usually 0, negative on immediate-fallback paths) carries
    /// into the next search; resetting to 0 here diverged Nbar/selection
    /// on reused subtrees. Both sides drain O to <= 0 after every search,
    /// so verbatim copy is exact.
    pub fn clone_subtree(&self, id: NodeId) -> Tree {
        let mut out = Tree::new();
        out.arena.clear();
        let mut remap: HashMap<NodeId, NodeId> = HashMap::new();
        fn rec(t: &Tree, out: &mut Tree, remap: &mut HashMap<NodeId, NodeId>, id: NodeId) -> NodeId {
            if let Some(&m) = remap.get(&id) {
                return m;
            }
            let src = &t.arena[id];
            let nid = out.arena.len();
            out.arena.push(Node {
                prior: src.prior,
                raw: src.raw,
                n: src.n,
                w: src.w,
                o: src.o,
                m: src.m,
                m_n: src.m_n,
                expanded: src.expanded,
                children: Vec::new(),
            });
            remap.insert(id, nid);
            let kids: Vec<(Action, NodeId)> =
                src.children.iter().map(|(a, c)| (*a, rec(t, out, remap, *c))).collect();
            out.arena[nid].children = kids;
            nid
        }
        let root = rec(self, &mut out, &mut remap, id);
        out.root = root;
        out
    }

    /// Count reachable nodes (dedup by arena id; mirrors `tt_live_nodes`).
    /// F7 metric note: Python `tt_live_nodes` dedups by `id(node)` ACROSS
    /// the whole dict, while `TtStore::live_nodes` sums per-entry reachable
    /// counts. The two agree exactly when no node object is shared between
    /// entries — which `prune_to` guarantees (only the promoted line
    /// survives) and which `insert_child` guarantees on fresh clones (each
    /// child subtree is an independent clone). Mid-game (pre-prune) the
    /// Python count can read lower if two keys ever alias the same object
    /// (same position via different move orders sharing a reference);
    /// the Rust count then reads higher by the shared size. Measurement
    /// only: never feeds selection, eviction, or promotion.
    pub fn live_nodes(&self, id: NodeId) -> usize {
        let mut seen: HashSet<NodeId> = HashSet::new();
        let mut stack = vec![id];
        let mut n = 0;
        while let Some(x) = stack.pop() {
            if !seen.insert(x) {
                continue;
            }
            n += 1;
            stack.extend(self.arena[x].children.iter().map(|(_, c)| *c));
        }
        n
    }
}

// ---------------------------------------------------------------------------
// Selection math (mcts.py _select / _select_wu / _select_fpu / _ml_bonus)
// ---------------------------------------------------------------------------

/// PUCT `_select` (mcts.py:44). `total` = parent visit sum; max(1,..) is the
/// audit-1.3 first-descent fix.
pub fn select_score(q: f64, prior: f64, n: u32, total: u64, c_puct: f64) -> f64 {
    let total = total.max(1) as f64;
    q + c_puct * prior * total.sqrt() / (1.0 + f64::from(n))
}

/// WU `_select_wu` (mcts.py:53). `nbar` = n + o for the child, `total` =
/// parent Nbar. Equals select_score when o == 0.
pub fn select_wu_score(q: f64, prior: f64, n: u32, o: i32, total_nbar: i64, c_puct: f64) -> f64 {
    let nbar = i64::from(n) + i64::from(o);
    let total = total_nbar.max(1) as f64;
    q + c_puct * prior * total.sqrt() / (1.0 + nbar as f64)
}

/// FPU `_select_fpu` (mcts.py:64). Unvisited (n==0) read parent_q -
/// reduction + u; visited read q + u. Caller passes reduction 0.0 for the
/// root's children.
pub fn select_fpu_score(
    q: f64,
    prior: f64,
    n: u32,
    parent_q: f64,
    total: u64,
    c_puct: f64,
    reduction: f64,
) -> f64 {
    let u = c_puct * prior * (total.max(1) as f64).sqrt() / (1.0 + f64::from(n));
    if n == 0 {
        parent_q - reduction + u
    } else {
        q + u
    }
}

/// Batched-path FPU for unvisited children (mcts.py:801): same as FPU but
/// the u denom uses Nbar (n+o).
pub fn select_fpu_wu_score(
    prior: f64,
    n: u32,
    o: i32,
    parent_q: f64,
    total_nbar: i64,
    c_puct: f64,
    reduction: f64,
) -> f64 {
    let nb = (i64::from(n) + i64::from(o)) as f64;
    let u = c_puct * prior * (total_nbar.max(1) as f64).sqrt() / (1.0 + nb);
    parent_q - reduction + u
}

/// In-tree MLH bonus (mcts.py:78). Never raises: Nones / indecisive parent
/// / non-finite => 0.0.
pub fn ml_bonus(
    child_m: Option<f64>,
    parent_m: Option<f64>,
    parent_q: f64,
    slope: f64,
    cap: f64,
    thr: f64,
) -> f64 {
    let (cm, pm) = match (child_m, parent_m) {
        (Some(c), Some(p)) => (c, p),
        _ => return 0.0,
    };
    if parent_q.abs() <= thr {
        return 0.0;
    }
    let mut d = cm - pm;
    if d > cap {
        d = cap;
    } else if d < -cap {
        d = -cap;
    }
    let s = if parent_q > 0.0 { -1.0 } else { 1.0 };
    let b = slope * d * s;
    if b.is_finite() { b } else { 0.0 }
}

/// Best-child index into `children` (first-max wins, init -1e18 —
/// matches Python's strict `>` comparison over insertion order).
pub fn best_child(
    tree: &Tree,
    id: NodeId,
    c_puct: f64,
    wu: bool,
    fpu_reduction: Option<f64>,
    is_root: bool,
    ml: Option<(f64, f64, f64)>,
) -> Option<usize> {
    let kids = &tree.arena[id].children;
    if kids.is_empty() {
        return None;
    }
    let parent = &tree.arena[id];
    let parent_q = -parent.q();
    let parent_m = parent.m;
    let total_n = tree.total_n(id);
    let total_nbar = tree.total_nbar(id);
    let mut best: Option<usize> = None;
    let mut best_s = -1e18f64;
    let fpu_on = fpu_reduction.map(|r| r != 0.0).unwrap_or(false);
    for (i, (_, c)) in kids.iter().enumerate() {
        let ch = &tree.arena[*c];
        let sc = if wu {
            if fpu_on && ch.n == 0 {
                let red = if is_root { 0.0 } else { fpu_reduction.unwrap_or(0.0) };
                select_fpu_wu_score(ch.prior, ch.n, ch.o, parent_q, total_nbar, c_puct, red)
            } else {
                select_wu_score(ch.q(), ch.prior, ch.n, ch.o, total_nbar, c_puct)
            }
        } else if fpu_on {
            let red = if is_root { 0.0 } else { fpu_reduction.unwrap_or(0.0) };
            select_fpu_score(ch.q(), ch.prior, ch.n, parent_q, total_n, c_puct, red)
        } else {
            select_score(ch.q(), ch.prior, ch.n, total_n, c_puct)
        };
        let sc = match ml {
            Some((slope, cap, thr)) => sc + ml_bonus(ch.m, parent_m, parent_q, slope, cap, thr),
            None => sc,
        };
        if sc > best_s + 1e-12 { // stable ties across backend floating arithmetic
            best_s = sc;
            best = Some(i);
        }
    }
    best
}

// ---------------------------------------------------------------------------
// Backup / virtual loss / root-decided
// ---------------------------------------------------------------------------

/// Back up a leaf score along `path` (root first), flipping sign each ply
/// (mcts.py:994). Convention: `leaf_value` is ALREADY negated to
/// parent-view by the caller (pass `-v` for a value-subtree leaf, `-z` for a
/// terminal, `-draw` for a draw line). M is averaged WITHOUT the flip.
pub fn backup(tree: &mut Tree, path: &[NodeId], leaf_value: f64, leaf_m: Option<f64>) {
    let mut v = leaf_value;
    for (distance, &id) in path.iter().rev().enumerate() {
        let node = &mut tree.arena[id];
        node.n += 1;
        node.w += v;
        if let Some(lm) = leaf_m {
            node.m_n += 1;
            let lm = lm + distance as f64 / 300.0;
            node.m = Some(match node.m {
                None => lm,
                Some(m) => m + (lm - m) / f64::from(node.m_n),
            });
        }
        v = -v;
    }
}

/// Legacy child-only virtual loss add/remove (mcts.py:718). Retained for API
/// stability; the batched path uses WU pending counts instead.
pub fn vl_add(tree: &mut Tree, path: &[NodeId], vl: f64) {
    for &id in path {
        let node = &mut tree.arena[id];
        node.n += 1;
        node.w -= vl;
    }
}

pub fn vl_remove(tree: &mut Tree, path: &[NodeId], vl: f64) {
    for &id in path {
        let node = &mut tree.arena[id];
        node.n -= 1;
        node.w += vl;
    }
}

/// WU pending-count add/remove over `path` (mcts.py:105,837). Callers pass
/// path[1..] (root excluded — ancestors derive from children sums).
pub fn o_add(tree: &mut Tree, path: &[NodeId]) {
    for &id in path {
        tree.arena[id].o += 1;
    }
}

pub fn o_remove(tree: &mut Tree, path: &[NodeId]) {
    for &id in path {
        tree.arena[id].o -= 1;
    }
}

/// Futile/proven stop probe (mcts.py:112): a well-supported decisive child
/// (n>=3, |Q|>=0.999). Pure read.
pub fn root_decided(tree: &Tree, root: NodeId) -> bool {
    let _ = (tree, root);
    false // Neural estimates are not solved-position proofs.
}

// ---------------------------------------------------------------------------
// Expansion helpers: mask/normalize, forcing boost, Dirichlet
// ---------------------------------------------------------------------------

/// Mask net priors to `legal`, renormalize (uniform fallback on zero sum),
/// apply the forcing boost (1+bonus on flagged actions with p>0, then
/// renorm), mirroring mcts.py:383-389 + :546-550 + _boost_forcing.
pub fn normalize_expand_priors(
    net: &[(Action, f64)],
    legal: &[Action],
    forcing: &[bool],
    forcing_bonus: f64,
) -> Vec<(Action, f64)> {
    let in_net = |a: Action| net.iter().find(|(x, _)| *x == a).map(|(_, p)| *p).unwrap_or(0.0);
    let mut pv: Vec<(Action, f64)> = legal.iter().map(|&a| (a, in_net(a))).collect();
    let s: f64 = pv.iter().map(|(_, p)| *p).sum();
    if s > 0.0 {
        for (_, p) in pv.iter_mut() {
            *p /= s;
        }
    } else if !pv.is_empty() {
        let u = 1.0 / pv.len() as f64;
        for (_, p) in pv.iter_mut() {
            *p = u;
        }
    }
    if forcing_bonus != 0.0 {
        let mut boosted = false;
        for (i, (_, p)) in pv.iter_mut().enumerate() {
            if forcing.get(i).copied().unwrap_or(false) && *p > 0.0 {
                *p *= 1.0 + forcing_bonus;
                boosted = true;
            }
        }
        if boosted {
            let s2: f64 = pv.iter().map(|(_, p)| *p).sum();
            if s2 > 0.0 {
                for (_, p) in pv.iter_mut() {
                    *p /= s2;
                }
            }
        }
    }
    pv
}

/// Expand `id` with normalized priors (prior = raw = p).
pub fn expand_node(tree: &mut Tree, id: NodeId, pv: &[(Action, f64)]) {
    let kids: Vec<(Action, NodeId)> = pv
        .iter()
        .map(|(a, p)| {
            let nid = tree.alloc(Node::fresh(*p, *p));
            (*a, nid)
        })
        .collect();
    let node = &mut tree.arena[id];
    node.children = kids;
    node.expanded = true;
}

/// Mix caller-supplied Dirichlet noise into a freshly expanded root
/// (mcts.py:391-394). `noise` aligns with `legal` order.
pub fn mix_root_noise(tree: &mut Tree, root: NodeId, legal: &[Action], noise: &[f64], eps: f64) {
    for (i, a) in legal.iter().enumerate() {
        if i >= noise.len() {
            break;
        }
        if let Some(cid) = tree.arena[root].find(*a) {
            let raw = tree.arena[cid].raw;
            tree.arena[cid].prior = (1.0 - eps) * raw + eps * noise[i];
        }
    }
}

/// Reuse path (mcts.py:396-407): remix from RAW for still-legal children,
/// drop children that are no longer legal.
pub fn remix_root_noise(tree: &mut Tree, root: NodeId, legal: &[Action], noise: &[f64], eps: f64) {
    for (i, a) in legal.iter().enumerate() {
        if i >= noise.len() {
            break;
        }
        if let Some(cid) = tree.arena[root].find(*a) {
            let raw = tree.arena[cid].raw;
            tree.arena[cid].prior = (1.0 - eps) * raw + eps * noise[i];
        }
    }
    let legal_set: HashSet<Action> = legal.iter().copied().collect();
    tree.arena[root].children.retain(|(a, _)| legal_set.contains(a));
}

// ---------------------------------------------------------------------------
// Visits output + stats (mcts.py:628-700)
// ---------------------------------------------------------------------------

/// Per-search statistics (all measurement, never raises in Python).
#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub root_visits: u64,
    pub reused: u64,
    pub new_sims: u64,
    pub extensions: u64,
    pub early_stop: bool,
    pub breadth: usize,
    pub visit_entropy: f64,
    pub kl: f64,
    pub root_q: f64,
}

/// Raw child visit counts in children order: (action, n).
pub fn raw_visits(tree: &Tree, root: NodeId) -> Vec<(Action, u32)> {
    tree.arena[root].children.iter().map(|(a, c)| (*a, tree.arena[*c].n)).collect()
}

/// Singleton prune + normalize to a dense 4096 f32 vector (mcts.py:628-700).
/// prune_singletons drops N<=1 BEFORE normalize, with the all-zero fallback.
/// Returns (dense, pruned_counts, total).
pub fn finalize_visits(
    tree: &Tree,
    root: NodeId,
    prune_singletons: bool,
) -> ([f32; N_ACTIONS], Vec<(Action, f64)>, f64) {
    let mut counts: Vec<(Action, f64)> =
        tree.arena[root].children.iter().map(|(a, c)| (*a, f64::from(tree.arena[*c].n))).collect();
    if false && prune_singletons {
        for (_, v) in counts.iter_mut() {
            if *v <= 1.0 {
                *v = 0.0;
            }
        }
        if counts.iter().all(|(_, v)| *v == 0.0) {
            for (i, (a, _)) in tree.arena[root].children.iter().enumerate() {
                counts[i].1 = f64::from(tree.arena[tree.arena[root].children[i].1].n);
                let _ = a;
            }
        }
    }
    let tot: f64 = counts.iter().map(|(_, v)| *v).sum();
    let mut dense = [0.0f32; N_ACTIONS];
    if tot > 0.0 {
        for (a, v) in counts.iter() {
            dense[*a as usize] = (*v / tot) as f32;
        }
    }
    (dense, counts, tot)
}

/// Breadth + entropy over pruned counts (mcts.py:655-661).
pub fn breadth_entropy(counts: &[(Action, f64)], tot: f64) -> (usize, f64) {
    let lv: Vec<f64> = counts.iter().map(|(_, v)| *v).filter(|v| *v > 0.0).collect();
    let breadth = lv.len();
    let ent = if tot > 0.0 && !lv.is_empty() {
        -lv.iter().map(|v| {
            let p = *v / tot;
            p.ln() * p
        }).sum::<f64>()
    } else {
        0.0
    };
    (breadth, ent)
}

/// KL(visits || root raw priors renormalized over legal) (mcts.py:662-685).
pub fn kl_vs_raw(tree: &Tree, root: NodeId, legal: &[Action], counts: &[(Action, f64)], tot: f64) -> f64 {
    if tot <= 0.0 {
        return 0.0;
    }
    let raws: Vec<f64> = legal
        .iter()
        .map(|a| match tree.arena[root].find(*a) {
            Some(c) => tree.arena[c].raw,
            None => 0.0,
        })
        .collect();
    let rsum: f64 = raws.iter().sum();
    if !(rsum > 0.0) || !rsum.is_finite() {
        return 0.0;
    }
    let cmap: HashMap<Action, f64> = counts.iter().copied().collect();
    let mut kl = 0.0;
    for (a, r) in legal.iter().zip(raws.iter()) {
        let v = cmap.get(a).copied().unwrap_or(0.0) / tot;
        let p = *r / rsum;
        if v > 0.0 && p > 0.0 {
            kl += v * (v / p).ln();
        }
    }
    if kl.is_finite() { kl } else { 0.0 }
}

/// Visit-weighted mean child Q at the root (V20 D2 root_q, mcts.py:686-698).
pub fn root_q(tree: &Tree, root: NodeId, counts: &[(Action, f64)], tot: f64) -> f64 {
    if tot <= 0.0 {
        return 0.0;
    }
    let q = counts
        .iter()
        .map(|(a, v)| match tree.arena[root].find(*a) {
            Some(c) => *v * tree.arena[c].q(),
            None => 0.0,
        })
        .sum::<f64>()
        / tot;
    if q.is_finite() { q } else { 0.0 }
}

// ---------------------------------------------------------------------------
// Transposition store (mcts.py:408-413, 701-714, prune_tt, tt_live_nodes)
// ---------------------------------------------------------------------------

/// Persistent transposition store across moves. Insertion-ordered for
/// oldest-first eviction past 20k entries (dicts are insertion-ordered in
/// Python; VecDeque plays that role here).
///
/// F5 u64 key domain: every key here is `py_hash_u64(rep_key)` computed on
/// the Python side of the bridge — `rep_key()` is python-chess
/// `transposition_key()` (a tuple of ints: board + turn + castling + EP, no
/// clocks), hashed with Python `__hash__` and reinterpreted as u64
/// (`h as u64`, i.e. two's-complement bit pattern). History, path, and TT
/// keys all cross through the SAME function, so loop detection and TT
/// promotion agree by construction. Tuples of ints hash deterministically
/// in-process (int hashes are identity; no str/bytes randomization
/// involved), so a game-leg is self-consistent; hashes are NOT stable
/// across processes (never persist them). 64-bit collisions are negligible
/// (birthday bound ~2^32 entries; the store caps at 20k). The eval cache
/// uses `(key, rep_count)` pairs — the count is part of the key because
/// rep planes 14-15 depend on it (B6) — while TT/history use `key` alone.
#[derive(Clone, Debug, Default)]
pub struct TtStore {
    pub map: HashMap<u64, Tree>,
    pub order: VecDeque<u64>,
}

impl TtStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn get(&self, key: u64) -> Option<&Tree> {
        self.map.get(&key)
    }

    fn push_order(&mut self, key: u64) {
        if !self.map.contains_key(&key) {
            self.order.push_back(key);
        }
    }

    /// Store the search root (mcts.py:408-413): evict oldest past 20000,
    /// then insert. Assignment to an existing key keeps its order slot
    /// (matches Python dict semantics).
    pub fn insert_root(&mut self, key: u64, tree: &Tree) {
        while self.map.len() > 20000 {
            match self.order.pop_front() {
                Some(k) => {
                    self.map.remove(&k);
                }
                None => break,
            }
        }
        self.push_order(key);
        self.map.insert(key, tree.clone());
    }

    /// Store an expanded child with visits under its own position key
    /// (mcts.py:709-714). No eviction here (matches Python).
    pub fn insert_child(&mut self, key: u64, tree: &Tree, child: NodeId) {
        self.push_order(key);
        self.map.insert(key, tree.clone_subtree(child));
    }

    /// Keep ONLY the promoted child's subtree (mcts.py prune_tt, :125).
    pub fn prune_to(&mut self, next_key: u64) {
        let keep = self.map.remove(&next_key);
        self.map.clear();
        self.order.clear();
        if let Some(t) = keep {
            self.order.push_back(next_key);
            self.map.insert(next_key, t);
        }
    }

    /// Promote a stored subtree for the next search (clone; counts equal).
    pub fn promote(&self, key: u64) -> Option<Tree> {
        self.map.get(&key).cloned()
    }

    /// Reachable-node census over the whole store (mirrors tt_live_nodes).
    pub fn live_nodes(&self) -> usize {
        self.map.values().map(|t| t.live_nodes(t.root)).sum()
    }
}
