//! Search drivers: sequential (leaf_batch==1) and batched-leaf (leaf_batch>1)
//! MCTS over a generic [`Position`], with evals arriving as opaque vectors
//! through `EvalFn`. Faithful port of `search()` + `_search_batched()`.

use crate::{
    Action, Node, NodeId, Stats, Tree, TtStore, backup, best_child, breadth_entropy,
    expand_node, finalize_visits, kl_vs_raw, mix_root_noise, normalize_expand_priors, o_add,
    o_remove, remix_root_noise, root_decided, root_q, N_ACTIONS,
};
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Environment + eval plumbing (board stays caller-side)
// ---------------------------------------------------------------------------

/// Board handle owned by the caller. The tree never sees pieces; it queries
/// these scalars per position. V21.1/D implements this over
/// chesscore.Position snapshots; tests use a scripted mock.
pub trait Position: Clone {
    /// Transposition / repetition key (rep_key in mcts.py).
    fn rep_key(&self) -> u64;
    fn tt_key(&self) -> u64 { self.rep_key() }
    fn repetition_draw(&self) -> bool { false }
    /// Legal actions in deterministic order (drives tie-breaks + noise map).
    fn legal_actions(&self) -> Vec<Action>;
    fn apply(&mut self, a: Action);
    /// Terminal score in side-to-move view (z in mcts.py), None if ongoing.
    fn stm_terminal_z(&self) -> Option<f64>;
    fn in_check(&self) -> bool;
    fn piece_count(&self) -> u32;
    fn white_pawns(&self) -> u32;
    fn black_pawns(&self) -> u32;
    fn white_to_move(&self) -> bool;
    /// Either side a pawn from queening (mcts.py _has_promo_threat).
    fn has_promo_threat(&self) -> bool;
    /// Forcing-first flag for one legal action: gives-check / capture /
    /// promotion / pawn push to 6th-7th (mcts.py _boost_forcing).
    fn forcing_action(&self, a: Action) -> bool;
    /// Contempt-shaped draw score, side-to-move LEAF view
    /// (selfplay.draw_leaf_value; 0.0 reproduces yesterday).
    fn draw_score_stm(&self) -> f64;
}

/// One eval reply for a batch of positions: opaque net priors (sparse
/// action,p pairs — masking/normalization happen here), values, and
/// moves-left reads (normalized units; non-finite already sanitized to
/// None by the caller, mirroring _ml_single / batch ml_arr handling).
#[derive(Clone, Debug)]
pub struct EvalOut {
    pub priors: Vec<Vec<(Action, f64)>>,
    pub values: Vec<f64>,
    pub mls: Vec<Option<f64>>,
}

/// Search knobs. `virtual_loss` is accepted-but-ignored (B1: the batched
/// path uses WU O+=1 instead; mirrors mcts.py's accepted-but-ignored arg).
#[derive(Clone, Debug)]
pub struct SearchParams {
    pub c_puct: f64,
    pub leaf_batch: usize,
    pub virtual_loss: f64,
    pub fpu_reduction: f64,
    pub futile_stop: bool,
    pub quiescence_depth: usize,
    pub forcing_bonus: f64,
    pub ml_slope: f64,
    pub ml_cap: f64,
    pub ml_thr: f64,
    /// ml_fn present AND slope != 0 (mcts.py _ml_on).
    pub ml_on: bool,
    pub prune_singletons: bool,
}

impl Default for SearchParams {
    /// Yesterday bit-exact defaults: batch 1, FPU/ML/prune/stop off.
    fn default() -> Self {
        SearchParams {
            c_puct: 1.414,
            leaf_batch: 1,
            virtual_loss: 1.0,
            fpu_reduction: 0.0,
            futile_stop: false,
            quiescence_depth: 2,
            forcing_bonus: 0.25,
            ml_slope: 0.0,
            ml_cap: 0.07,
            ml_thr: 0.8,
            ml_on: false,
            prune_singletons: false,
        }
    }
}

impl SearchParams {
    fn ml_cfg(&self) -> Option<(f64, f64, f64)> {
        if self.ml_on {
            Some((self.ml_slope, self.ml_cap, self.ml_thr))
        } else {
            None
        }
    }

    fn fpu(&self) -> Option<f64> {
        if self.fpu_reduction != 0.0 {
            Some(self.fpu_reduction)
        } else {
            None
        }
    }

    /// M attached to terminal/loop lines when ML is on (mcts.py _lm_term).
    fn lm_term(&self) -> Option<f64> {
        if self.ml_on {
            Some(0.0)
        } else {
            None
        }
    }
}

/// Search result: normalized visits over 4096, the grown tree (hand back
/// for TT stores), stats, and per-sim depths.
pub struct SearchOut {
    pub visits: [f32; N_ACTIONS],
    pub tree: Tree,
    pub stats: Stats,
    pub depths: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Internal cursor helpers
// ---------------------------------------------------------------------------

fn pawn_count<P: Position>(pos: &P, white: bool) -> u32 {
    if white {
        pos.white_pawns()
    } else {
        pos.black_pawns()
    }
}

struct Cursor<P> {
    cur: P,
    node: NodeId,
    path: Vec<NodeId>,
    /// Number of actions taken this sim (selection + quiescence).
    actions: usize,
    path_keys: Vec<u64>,
    path_set: HashSet<u64>,
    n_pieces: u32,
    mover_white: bool,
    mover_pawns: u32,
    arrived_by_capture: bool,
    arrived_by_promo: bool,
}

impl<P: Position> Cursor<P> {
    fn new(root_state: &P, root: NodeId) -> Self {
        let mut path_set = HashSet::new();
        path_set.insert(root_state.rep_key());
        let mover_white = root_state.white_to_move();
        Cursor {
            cur: root_state.clone(),
            node: root,
            path: vec![root],
            actions: 0,
            path_keys: vec![root_state.rep_key()],
            path_set,
            n_pieces: root_state.piece_count(),
            mover_white,
            mover_pawns: pawn_count(root_state, mover_white),
            arrived_by_capture: false,
            arrived_by_promo: false,
        }
    }

    /// Apply `a` and refresh arrival proxies (capture = piece-count drop;
    /// promo = just-moved pawn-count drop; mcts.py:485-503).
    fn step(&mut self, a: Action) {
        self.cur.apply(a);
        self.actions += 1;
        let n2 = self.cur.piece_count();
        self.arrived_by_capture = n2 < self.n_pieces;
        self.n_pieces = n2;
        let just_moved_white = !self.cur.white_to_move();
        let n_pawns = pawn_count(&self.cur, just_moved_white);
        self.arrived_by_promo = n_pawns < self.mover_pawns_of(just_moved_white);
        self.mover_white = self.cur.white_to_move();
        self.mover_pawns = pawn_count(&self.cur, self.mover_white);
    }

    fn mover_pawns_of(&self, _just_moved_white: bool) -> u32 {
        // Pawns of the side that just moved, as tracked pre-move.
        // self.mover_white/mover_pawns describe the pre-move side to move
        // (= the side that just moved after step), so compare directly.
        self.mover_pawns
    }

    fn push_key(&mut self) {
        let rk = self.cur.rep_key();
        self.path_keys.push(rk);
        self.path_set.insert(rk);
    }
}

/// Expand `cursor.node` from one eval reply. Returns (value, ml).
fn expand_from_eval<P: Position>(
    tree: &mut Tree,
    cursor: &Cursor<P>,
    params: &SearchParams,
    priors: &[(Action, f64)],
    value: f64,
    ml: Option<f64>,
) -> (f64, Option<f64>) {
    let legal = cursor.cur.legal_actions();
    let forcing: Vec<bool> = legal.iter().map(|&a| cursor.cur.forcing_action(a)).collect();
    let pv = normalize_expand_priors(priors, &legal, &forcing, params.forcing_bonus);
    expand_node(tree, cursor.node, &pv);
    let leaf_m = if params.ml_on { ml } else { None };
    (value, leaf_m)
}

/// Quiescence loop shared by both paths (mcts.py:567-621, 933-986).
/// `draw_m`: M attached to loop/terminal-DRAW backups inside quiescence —
/// Some(_lm_term) on the SEQUENTIAL path, None on the BATCHED path
/// (checklist item 28; the asymmetry is faithful, not a bug).
/// `start_value` is the expansion leaf's value; returns None when the leaf
/// was consumed (loop/terminal backed up), else the value for the caller
/// to back up. The expansion M is kept, never recomputed here.

fn quiescence_run<P: Position, F: FnMut(&[P]) -> EvalOut>(
    tree: &mut Tree,
    cursor: &mut Cursor<P>,
    params: &SearchParams,
    draw_m: Option<f64>,
    start_value: f64,
    eval: &mut F,
    extensions: &mut u64,
    history_set: &HashSet<u64>,
    has_history: bool,
) -> Option<f64> {
    let mut leaf_v: Option<f64> = Some(start_value);
    let mut qext = 0usize;
    while params.quiescence_depth > 0
        && qext < params.quiescence_depth
        && (cursor.cur.in_check()
            || cursor.arrived_by_capture
            || cursor.arrived_by_promo
            || cursor.cur.has_promo_threat())
        && !tree.node(cursor.node).children.is_empty()
    {
        qext += 1;
        *extensions += 1;
        // Prior-best child, first-max (mcts.py:574,940).
        let mut best_a: Option<Action> = None;
        let mut best_p = f64::NEG_INFINITY;
        for (a, c) in tree.node(cursor.node).children.clone() {
            let p = tree.node(c).prior;
            if p > best_p {
                best_p = p;
                best_a = Some(a);
            }
        }
        let ba = match best_a {
            Some(a) => a,
            None => break,
        };
        let cid = tree.node(cursor.node).find(ba).unwrap();
        cursor.step(ba);
        cursor.node = cid;
        cursor.path.push(cid);
        if cursor.cur.repetition_draw() {
            backup(tree, &cursor.path, -cursor.cur.draw_score_stm(), draw_m);
            leaf_v = None;
            break;
        }
        cursor.push_key();
        match cursor.cur.stm_terminal_z() {
            Some(z) => {
                if z == 0.0 {
                    backup(tree, &cursor.path, -cursor.cur.draw_score_stm(), draw_m);
                } else {
                    backup(tree, &cursor.path, -z, params.lm_term());
                }
                leaf_v = None;
                break;
            }
            None => {}
        }
        let out = eval(std::slice::from_ref(&cursor.cur));
        let pv = out.priors.into_iter().next().unwrap_or_default();
        let v = out.values.into_iter().next().unwrap_or(0.0);
        leaf_v = Some(v);
        let legal = cursor.cur.legal_actions();
        let forcing: Vec<bool> = legal.iter().map(|&a| cursor.cur.forcing_action(a)).collect();
        let pv = normalize_expand_priors(&pv, &legal, &forcing, params.forcing_bonus);
        expand_node(tree, cursor.node, &pv);
    }
    leaf_v
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Run MCTS from `root_state`. `root` is a fresh `Tree::new()` or a TT
/// promoted subtree. `root_noise`: Some(noise over legal order) iff
/// dirichlet_eps>0 (fresh expand mixes, reuse remixes from raw).
/// `dir_eps`: the Dirichlet epsilon matching `root_noise`.
pub fn search<P: Position, F: FnMut(&[P]) -> EvalOut>(
    root_state: &P,
    mut tree: Tree,
    params: &SearchParams,
    n_sims: usize,
    history: &[u64],
    root_noise: Option<&[f64]>,
    dir_eps: f64,
    tt: Option<&mut TtStore>,
    eval: &mut F,
) -> SearchOut {
    let mut tt_opt = tt;
    let root_key = root_state.tt_key();
    let legal0 = root_state.legal_actions();
    let stats = Stats::default();
    let mut depths: Vec<usize> = Vec::new();
    let mut extensions: u64 = 0;

    if legal0.is_empty() {
        return SearchOut {
            visits: [0.0f32; N_ACTIONS],
            tree,
            stats,
            depths,
        };
    }

    // Root expand-or-remix (mcts.py:378-413).
    let root_id = tree.root;
    let was_expanded = tree.node(root_id).expanded;
    let reused: u64 = if was_expanded {
        tree.arena[root_id].children.iter().map(|(_, c)| u64::from(tree.arena[*c].n)).sum()
    } else {
        0
    };
    if !was_expanded {
        tree.arena[root_id] = Node::root();
        let out = eval(std::slice::from_ref(root_state));
        let pv = out.priors.into_iter().next().unwrap_or_default();
        let forcing: Vec<bool> = legal0.iter().map(|&a| root_state.forcing_action(a)).collect();
        let pv = normalize_expand_priors(&pv, &legal0, &forcing, params.forcing_bonus);
        expand_node(&mut tree, root_id, &pv);
        if let Some(noise) = root_noise {
            mix_root_noise(&mut tree, root_id, &legal0, noise, dir_eps);
        }
    } else if let Some(noise) = root_noise {
        remix_root_noise(&mut tree, root_id, &legal0, noise, dir_eps);
    }
    if let Some(store) = tt_opt.as_deref_mut() {
        store.insert_root(root_key, &tree);
    }

    let history_set: HashSet<u64> = history.iter().copied().collect();
    let has_history = !history.is_empty();
    let lm_term = params.lm_term();
    let mut early_fired = false;

    if params.leaf_batch > 1 {
        extensions += search_batched(
            &mut tree,
            root_state,
            params,
            n_sims,
            &history_set,
            has_history,
            lm_term,
            &mut early_fired,
            &mut depths,
            eval,
        );
    } else {
        for _ in 0..n_sims {
            if params.futile_stop && root_decided(&tree, root_id) {
                early_fired = true; // B4 measurement-only
                break;
            }
            let mut cursor = Cursor::new(root_state, root_id);
            // Selection (mcts.py:460-517).
            let mut broke = false;
            while tree.node(cursor.node).expanded && !tree.node(cursor.node).children.is_empty() {
                let is_root = cursor.node == root_id;
                let ci = best_child(&tree, cursor.node, params.c_puct, false, params.fpu(), is_root, params.ml_cfg());
                let ci = match ci {
                    Some(i) => i,
                    None => break,
                };
                let (ba, bc) = tree.node(cursor.node).children[ci];
                cursor.step(ba);
                cursor.node = bc;
                cursor.path.push(bc);
                if cursor.cur.repetition_draw() {
                    backup(&mut tree, &cursor.path, -cursor.cur.draw_score_stm(), lm_term);
                    depths.push(cursor.actions);
                    broke = true;
                    break;
                }
                cursor.push_key();
                if let Some(z) = cursor.cur.stm_terminal_z() {
                    if z == 0.0 {
                        backup(&mut tree, &cursor.path, -cursor.cur.draw_score_stm(), lm_term);
                    } else {
                        backup(&mut tree, &cursor.path, -z, lm_term);
                    }
                    depths.push(cursor.actions);
                    broke = true;
                    break;
                }
            }
            if broke {
                continue;
            }
            // Stopped at an unexpanded (or childless) node.
            if let Some(z) = cursor.cur.stm_terminal_z() {
                if z == 0.0 {
                    backup(&mut tree, &cursor.path, -cursor.cur.draw_score_stm(), lm_term);
                } else {
                    backup(&mut tree, &cursor.path, -z, lm_term);
                }
                depths.push(cursor.actions);
                continue;
            }
            let out = eval(std::slice::from_ref(&cursor.cur));
            let pv = out.priors.into_iter().next().unwrap_or_default();
            let v = out.values.into_iter().next().unwrap_or(0.0);
            let m = out.mls.into_iter().next().unwrap_or(None);
            let (leaf_v, leaf_m) = expand_from_eval(&mut tree, &cursor, params, &pv, v, m);
            // Quiescence (sequential style: draws carry lm_term).
            let leaf_v = quiescence_run(
                &mut tree,
                &mut cursor,
                params,
                lm_term,
                leaf_v,
                eval,
                &mut extensions,
                &history_set,
                has_history,
            );
            if let Some(v) = leaf_v {
                backup(&mut tree, &cursor.path, -v, leaf_m);
            }
            depths.push(cursor.actions);
        }
    }

    finish(
        &mut tree,
        root_id,
        params,
        &legal0,
        reused,
        n_sims,
        extensions,
        early_fired,
        depths,
        root_state,
        tt_opt,
    )
}

// ---------------------------------------------------------------------------
// Batched driver (mcts.py _search_batched)
// ---------------------------------------------------------------------------

struct Pending<P> {
    cursor: Cursor<P>,
    immediate: bool,
}

fn search_batched<P: Position, F: FnMut(&[P]) -> EvalOut>(
    tree: &mut Tree,
    root_state: &P,
    params: &SearchParams,
    n_sims: usize,
    history_set: &HashSet<u64>,
    has_history: bool,
    lm_term: Option<f64>,
    early_fired: &mut bool,
    depths: &mut Vec<usize>,
    eval: &mut F,
) -> u64 {
    let root_id = tree.root;
    let mut extensions: u64 = 0;
    let mut sims_done = 0usize;
    while sims_done < n_sims {
        if params.futile_stop && root_decided(tree, root_id) {
            *early_fired = true;
            break;
        }
        let batch_n = std::cmp::min(params.leaf_batch.max(1), n_sims - sims_done);
        let mut pending: Vec<Pending<P>> = Vec::new();
        let mut pending_ids: HashSet<NodeId> = HashSet::new();
        let mut attempts = 0usize;
        while pending.len() < batch_n && attempts < batch_n * 3 + 1 {
            attempts += 1;
            let mut cursor = Cursor::new(root_state, root_id);
            let mut fell_back = false;
            // Selection with WU pending counts (mcts.py:795-856).
            while tree.node(cursor.node).expanded && !tree.node(cursor.node).children.is_empty() {
                let is_root = cursor.node == root_id;
                let ci =
                    best_child(&tree, cursor.node, params.c_puct, true, params.fpu(), is_root, params.ml_cfg());
                let ci = match ci {
                    Some(i) => i,
                    None => break,
                };
                let (ba, bc) = tree.node(cursor.node).children[ci];
                cursor.path.push(bc);
                o_add(tree, &[bc]);
                cursor.step(ba);
                cursor.node = bc;
                if cursor.cur.repetition_draw() {
                    let draw = cursor.cur.draw_score_stm();
                    let path = cursor.path.clone();
                    o_remove(tree, &path[1..]);
                    backup(tree, &path, -draw, None); // batched: draws carry NO M (:838)
                    depths.push(cursor.actions);
                    sims_done += 1;
                    fell_back = true;
                    break;
                }
                cursor.push_key();
                if let Some(z) = cursor.cur.stm_terminal_z() {
                    let path = cursor.path.clone();
                    o_remove(tree, &path[1..]);
                    if z == 0.0 {
                        let draw = cursor.cur.draw_score_stm();
                        backup(tree, &path, -draw, None); // (:849)
                    } else {
                        backup(tree, &path, -z, lm_term); // (:851)
                    }
                    depths.push(cursor.actions);
                    sims_done += 1;
                    fell_back = true;
                    break;
                }
            }
            if fell_back {
                continue;
            }
            if let Some(z) = cursor.cur.stm_terminal_z() {
                let path = cursor.path.clone();
                o_remove(tree, &path[1..]);
                if z == 0.0 {
                    let draw = cursor.cur.draw_score_stm();
                    backup(tree, &path, -draw, None); // (:863)
                } else {
                    backup(tree, &path, -z, lm_term); // (:865)
                }
                depths.push(cursor.actions);
                sims_done += 1;
                continue;
            }
            if pending_ids.contains(&cursor.node) {
                let path = cursor.path.clone();
                o_remove(tree, &path[1..]);
                if attempts >= batch_n * 3 {
                    break; // Flush unique pending leaves before retrying.
                }
                continue;
            }
            pending_ids.insert(cursor.node);
            pending.push(Pending { cursor, immediate: false });
            sims_done += 1;
        }
        // One shared forward for the batch (mcts.py:888-898).
        let batched_idx: Vec<usize> =
            pending.iter().enumerate().filter(|(_, p)| !p.immediate).map(|(i, _)| i).collect();
        let (priors_b, values_b, mls_b) = if !batched_idx.is_empty() {
            let states: Vec<P> = batched_idx.iter().map(|&i| pending[i].cursor.cur.clone()).collect();
            let out = eval(&states);
            let mls = if params.ml_on { out.mls } else { vec![None; states.len()] };
            (out.priors, out.values, mls)
        } else {
            (Vec::new(), Vec::new(), Vec::new())
        };
        let mut bi = 0usize;
        for p in pending.iter_mut() {
            let path = p.cursor.path.clone();
            o_remove(tree, &path[1..]);
            let (pv, leaf_v0, leaf_m0) = if p.immediate {
                let out = eval(std::slice::from_ref(&p.cursor.cur));
                let pv = out.priors.into_iter().next().unwrap_or_default();
                let v = out.values.into_iter().next().unwrap_or(0.0);
                let m = if params.ml_on {
                    out.mls.into_iter().next().unwrap_or(None)
                } else {
                    None
                };
                (pv, v, m)
            } else {
                let pv = priors_b.get(bi).cloned().unwrap_or_default();
                let v = values_b.get(bi).copied().unwrap_or(0.0);
                let m = mls_b.get(bi).copied().flatten();
                bi += 1;
                (pv, v, m)
            };
            let (leaf_v, leaf_m) =
                expand_from_eval(tree, &p.cursor, params, &pv, leaf_v0, leaf_m0);
            // Quiescence (batched style: draws carry None, mcts.py:961-972).
            let leaf_v = quiescence_run(
                tree,
                &mut p.cursor,
                params,
                None,
                leaf_v,
                eval,
                &mut extensions,
                history_set,
                has_history,
            );
            if let Some(v) = leaf_v {
                let path = p.cursor.path.clone();
                backup(tree, &path, -v, leaf_m);
            }
            depths.push(p.cursor.actions);
        }
    }
    extensions
}

// ---------------------------------------------------------------------------
// Finalize: visits, stats, TT child store (mcts.py:628-715)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn finish<P: Position>(
    tree: &mut Tree,
    root_id: NodeId,
    params: &SearchParams,
    legal: &[Action],
    reused: u64,
    n_sims: usize,
    extensions: u64,
    early_fired: bool,
    depths: Vec<usize>,
    root_state: &P,
    tt: Option<&mut TtStore>,
) -> SearchOut {
    let (dense, counts, tot) = finalize_visits(tree, root_id, params.prune_singletons);
    let (breadth, visit_entropy) = breadth_entropy(&counts, tot);
    let kl = kl_vs_raw(tree, root_id, legal, &counts, tot);
    let rq = root_q(tree, root_id, &counts, tot);
    let stats = Stats {
        root_visits: tot as u64,
        reused,
        new_sims: n_sims as u64,
        extensions,
        early_stop: early_fired,
        breadth,
        visit_entropy,
        kl,
        root_q: rq,
    };
    if let Some(store) = tt {
        // Store expanded children with visits under own keys (mcts.py:709).
        let kids: Vec<(Action, NodeId)> = tree.arena[root_id].children.clone();
        for (a, c) in kids {
            if tree.arena[c].expanded && tree.arena[c].n > 0 {
                let mut probe = root_state.clone();
                probe.apply(a);
                let key = probe.tt_key();
                store.insert_child(key, tree, c);
            }
        }
    }
    SearchOut { visits: dense, tree: tree.clone(), stats, depths }
}

// ---------------------------------------------------------------------------
// Dead-code guard: quiescence() is superseded by quiescence_run(); removed.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{kl_vs_raw, normalize_expand_priors, select_fpu_score, select_score, select_wu_score};

    // Tiny scripted game: path of actions; depth = path length.
    // legal(depth): 0 -> [10,20,30]; 1 -> [id-based pair]; 2 -> [single]; >=3 terminal.
    #[derive(Clone, Debug)]
    struct Mock {
        path: Vec<Action>,
    }

    impl Mock {
        fn depth(&self) -> usize {
            self.path.len()
        }
    }

    impl Position for Mock {
        fn rep_key(&self) -> u64 {
            let mut h: u64 = 0x9e37;
            for a in &self.path {
                h = h.wrapping_mul(0x100000001b3).wrapping_add(u64::from(*a) + 1);
            }
            h
        }
        fn legal_actions(&self) -> Vec<Action> {
            match self.depth() {
                0 => vec![10, 20, 30],
                1 => {
                    let b = self.path[0];
                    vec![b + 1, b + 2]
                }
                2 => vec![self.path[0] + self.path[1]],
                _ => vec![],
            }
        }
        fn apply(&mut self, a: Action) {
            self.path.push(a);
        }
        fn stm_terminal_z(&self) -> Option<f64> {
            if self.depth() >= 3 {
                // Decisive for even last action, draw for odd.
                if self.path[2] % 2 == 0 {
                    Some(-1.0)
                } else {
                    Some(0.0)
                }
            } else {
                None
            }
        }
        fn in_check(&self) -> bool {
            self.depth() == 1 && self.path[0] == 10
        }
        fn piece_count(&self) -> u32 {
            32u32.saturating_sub(self.depth() as u32)
        }
        fn white_pawns(&self) -> u32 {
            8
        }
        fn black_pawns(&self) -> u32 {
            8
        }
        fn white_to_move(&self) -> bool {
            self.depth() % 2 == 0
        }
        fn has_promo_threat(&self) -> bool {
            false
        }
        fn forcing_action(&self, a: Action) -> bool {
            a == 10
        }
        fn draw_score_stm(&self) -> f64 {
            0.0
        }
    }

    /// Scripted eval: uniform-ish priors tilted to the first legal action,
    /// value = +0.2 * (3 - depth) in leaf view... any deterministic fn works.
    fn mock_eval(states: &[Mock]) -> EvalOut {
        let mut priors = Vec::new();
        let mut values = Vec::new();
        let mut mls = Vec::new();
        for s in states {
            let legal = s.legal_actions();
            let pv: Vec<(Action, f64)> = legal
                .iter()
                .enumerate()
                .map(|(i, &a)| (a, if i == 0 { 0.6 } else { 0.4 / (legal.len() - 1).max(1) as f64 }))
                .collect();
            priors.push(pv);
            values.push(0.1 * (3 - s.depth().min(3)) as f64);
            mls.push(Some(0.5));
        }
        EvalOut { priors, values, mls }
    }

    #[test]
    fn puct_first_descent_picks_policy_max_not_lowest_index() {
        // total=0 => sqrt(max(1,0))=1 for all; highest prior must win.
        let s_hi = select_score(0.0, 0.7, 0, 0, 1.414);
        let s_lo = select_score(0.0, 0.2, 0, 0, 1.414);
        assert!(s_hi > s_lo);
    }

    #[test]
    fn wu_equals_select_when_o_zero() {
        for (n, total) in [(0u32, 0i64), (3, 10), (7, 25)] {
            let a = select_score(0.3, 0.5, n, total as u64, 1.414);
            let b = select_wu_score(0.3, 0.5, n, 0, total, 1.414);
            assert!((a - b).abs() < 1e-12, "n={n} total={total}");
        }
        // Pending counts suppress re-selection.
        let free = select_wu_score(0.3, 0.5, 0, 0, 4, 1.414);
        let held = select_wu_score(0.3, 0.5, 0, 2, 4, 1.414);
        assert!(held < free);
    }

    #[test]
    fn fpu_unvisited_reads_parent_minus_reduction() {
        let s = select_fpu_score(0.0, 0.5, 0, -0.2, 5, 1.414, 0.1);
        let u = 1.414 * 0.5 * (5f64).sqrt();
        assert!(((s - (-0.2 - 0.1 + u))).abs() < 1e-12);
        // Visited children ignore FPU.
        let v = select_fpu_score(0.4, 0.5, 3, -0.2, 5, 1.414, 0.1);
        assert!((v - (0.4 + 1.414 * 0.5 * (5f64).sqrt() / 4.0)).abs() < 1e-12);
    }

    #[test]
    fn ml_bonus_gate_clamp_sign() {
        // Indecisive parent => 0.
        assert_eq!(crate::ml_bonus(Some(0.4), Some(0.5), 0.5, 0.003, 0.07, 0.8), 0.0);
        // Decisive win (Qp>thr): shorter (childM<parentM) => positive.
        let b = crate::ml_bonus(Some(0.4), Some(0.5), 0.9, 0.003, 0.07, 0.8);
        assert!((b - 0.003 * 0.07).abs() < 1e-12, "{b}");
        // Decisive loss: longer => positive (delay).
        let b2 = crate::ml_bonus(Some(0.6), Some(0.5), -0.9, 0.003, 0.07, 0.8);
        assert!((b2 - 0.003 * 0.07).abs() < 1e-12, "{b2}");
        // Missing M => 0.
        assert_eq!(crate::ml_bonus(None, Some(0.5), 0.9, 0.003, 0.07, 0.8), 0.0);
    }

    #[test]
    fn backup_flips_signs_and_averages_m_without_flip() {
        let mut t = Tree::new();
        let c = t.alloc(Node::fresh(0.5, 0.5));
        t.arena[0].children.push((10, c));
        t.arena[0].expanded = true;
        let path = vec![0, c];
        backup(&mut t, &path, -0.6, Some(0.4));
        assert_eq!(t.arena[c].n, 1);
        assert!((t.arena[c].w - -0.6).abs() < 1e-12);
        assert!((t.arena[0].w - 0.6).abs() < 1e-12);
        assert_eq!(t.arena[c].m, Some(0.4));
        assert!((t.arena[0].m.unwrap() - (0.4 + 1.0 / 300.0)).abs() < 1e-12);
        backup(&mut t, &path, -0.2, Some(0.6));
        // Running mean with divisor = post-increment n.
        assert!((t.arena[c].m.unwrap() - 0.5).abs() < 1e-12);
        // leaf_m None leaves m untouched.
        backup(&mut t, &path, -0.2, None);
        assert!((t.arena[c].m.unwrap() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn root_decided_threshold() {
        let mut t = Tree::new();
        let c = t.alloc(Node::fresh(0.5, 0.5));
        t.arena[0].children.push((10, c));
        t.arena[0].expanded = true;
        assert!(!root_decided(&t, 0));
        for _ in 0..3 {
            backup(&mut t, &[0, c], -1.0, None);
        }
        assert!(!root_decided(&t, 0));
    }

    #[test]
    fn dirichlet_remix_uses_raw_not_compounded() {
        let mut t = Tree::new();
        let legal = vec![10u16, 20];
        expand_node(&mut t, 0, &[(10, 0.8), (20, 0.2)]);
        mix_root_noise(&mut t, 0, &legal, &[0.5, 0.5], 0.25);
        let p1 = t.arena[t.arena[0].find(10).unwrap()].prior;
        assert!((p1 - (0.75 * 0.8 + 0.25 * 0.5)).abs() < 1e-12);
        // Second remix from RAW again (audit round 2: no 0.75^k decay).
        remix_root_noise(&mut t, 0, &legal, &[0.5, 0.5], 0.25);
        let p2 = t.arena[t.arena[0].find(10).unwrap()].prior;
        assert!((p2 - p1).abs() < 1e-12);
        // Drop now-illegal children.
        remix_root_noise(&mut t, 0, &[10], &[1.0], 0.25);
        assert!(t.arena[0].find(20).is_none());
    }

    #[test]
    fn normalize_expand_masks_renorms_boosts() {
        let pv = normalize_expand_priors(&[(10, 0.2), (99, 5.0)], &[10, 20], &[true, false], 0.25);
        // 99 illegal dropped; 20 zero-net => uniform fallback... net has 10=0.2
        // so no fallback: 10 boosted.
        assert_eq!(pv.len(), 2);
        let sum: f64 = pv.iter().map(|(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-12);
        assert!(pv[0].1 > pv[1].1); // boosted 10 first
        // All-zero net => uniform.
        let pv2 = normalize_expand_priors(&[], &[10, 20], &[false, false], 0.25);
        assert!((pv2[0].1 - 0.5).abs() < 1e-12);
    }

    #[test]
    fn sequential_search_visits_sum_and_depths() {
        let root = Mock { path: vec![] };
        let params = SearchParams::default();
        let mut eval_calls = 0usize;
        let mut eval = |states: &[Mock]| {
            eval_calls += states.len();
            mock_eval(states)
        };
        let out = search(&root, Tree::new(), &params, 40, &[], None, 0.0, None, &mut eval);
        let tot: f32 = out.visits.iter().sum();
        assert!((tot - 1.0).abs() < 1e-6, "{tot}");
        assert_eq!(out.depths.len(), 40);
        assert_eq!(out.stats.new_sims, 40);
        assert_eq!(out.stats.reused, 0);
        assert_eq!(out.stats.root_visits, 40);
        assert!(!out.stats.early_stop);
        // Quiescence: descents through action 10 (in-check leaf) extend.
        assert!(out.stats.extensions > 0, "expected check extensions");
        // Breadth over 3 root moves, entropy finite, KL finite, root_q finite.
        assert!((1..=3).contains(&out.stats.breadth));
        assert!(out.stats.visit_entropy.is_finite());
        assert!(out.stats.kl.is_finite());
        assert!(out.stats.root_q.is_finite());
    }

    #[test]
    fn batched_single_shared_forward_and_o_drained() {
        let root = Mock { path: vec![] };
        let params = SearchParams { leaf_batch: 8, ..SearchParams::default() };
        let mut batches: Vec<usize> = Vec::new();
        let mut eval = |states: &[Mock]| {
            batches.push(states.len());
            mock_eval(states)
        };
        let out = search(&root, Tree::new(), &params, 8, &[], None, 0.0, None, &mut eval);
        // Root eval (1) + one shared batch forward (<=8) + quiescence singles.
        assert!(batches.iter().any(|&b| b > 1 && b <= 8), "{batches:?}");
        let tot: f32 = out.visits.iter().sum();
        assert!((tot - 1.0).abs() < 1e-6);
        // Nominal 8 sims + gather fallbacks (loop/terminal descents back up
        // immediately yet the batch still fills to 8 pendings — same as
        // Python, so visits can exceed n_sims by the fallback count).
        assert!(out.stats.root_visits >= 8, "{}", out.stats.root_visits);
        assert_eq!(out.depths.len() as u64, out.stats.root_visits);
        // All WU pending counts drained after the search: 0 on normal
        // paths, -k with k = immediate-fallback descents through the node
        // (Python removes O at gather AND at resolve — replicated exactly,
        // so marks are never left positive). See Node::o.
        fn check_o(t: &Tree, id: NodeId) {
            let o = t.arena[id].o;
            assert!(o <= 0, "node {id} o={o}");
            for (_, c) in t.arena[id].children.clone() {
                check_o(t, c);
            }
        }
        check_o(&out.tree, out.tree.root);
    }

    #[test]
    fn repetition_causes_draw_backup_with_no_extra_eval() {
        #[derive(Clone)]
        struct LoopMock(Mock);
        impl Position for LoopMock {
            fn repetition_draw(&self) -> bool { self.0.path.len() >= 1 }
            fn rep_key(&self) -> u64 {
                7 // mock explicitly reports a threefold terminal
            }
            fn legal_actions(&self) -> Vec<Action> {
                self.0.legal_actions()
            }
            fn apply(&mut self, a: Action) {
                self.0.apply(a);
            }
            fn stm_terminal_z(&self) -> Option<f64> {
                None
            }
            fn in_check(&self) -> bool {
                false
            }
            fn piece_count(&self) -> u32 {
                32
            }
            fn white_pawns(&self) -> u32 {
                8
            }
            fn black_pawns(&self) -> u32 {
                8
            }
            fn white_to_move(&self) -> bool {
                true
            }
            fn has_promo_threat(&self) -> bool {
                false
            }
            fn forcing_action(&self, _: Action) -> bool {
                false
            }
            fn draw_score_stm(&self) -> f64 {
                0.0
            }
        }
        let root = LoopMock(Mock { path: vec![] });
        let params = SearchParams { quiescence_depth: 0, ..SearchParams::default() };
        let mut calls = 0usize;
        let mut eval = |states: &[LoopMock]| {
            calls += states.len();
            EvalOut {
                priors: states
                    .iter()
                    .map(|s| s.legal_actions().into_iter().map(|a| (a, 1.0)).collect())
                    .collect(),
                values: vec![0.5; states.len()],
                mls: vec![None; states.len()],
            }
        };
        // history contains the root key => first descent loops immediately.
        let out = search(&root, Tree::new(), &params, 5, &[7], None, 0.0, None, &mut eval);
        assert_eq!(calls, 1, "only the root expansion evaluates");
        assert_eq!(out.stats.root_visits, 5);
    }

    #[test]
    fn futile_stop_fires_and_reports() {
        #[derive(Clone)]
        struct MateMock {
            d: usize,
        }
        impl Position for MateMock {
            fn rep_key(&self) -> u64 {
                self.d as u64 + 100
            }
            fn legal_actions(&self) -> Vec<Action> {
                if self.d == 0 {
                    vec![1, 2]
                } else {
                    vec![]
                }
            }
            fn apply(&mut self, _: Action) {
                self.d += 1;
            }
            fn stm_terminal_z(&self) -> Option<f64> {
                if self.d >= 1 {
                    Some(-1.0)
                } else {
                    None
                }
            }
            fn in_check(&self) -> bool {
                false
            }
            fn piece_count(&self) -> u32 {
                32
            }
            fn white_pawns(&self) -> u32 {
                8
            }
            fn black_pawns(&self) -> u32 {
                8
            }
            fn white_to_move(&self) -> bool {
                true
            }
            fn has_promo_threat(&self) -> bool {
                false
            }
            fn forcing_action(&self, _: Action) -> bool {
                false
            }
            fn draw_score_stm(&self) -> f64 {
                0.0
            }
        }
        let root = MateMock { d: 0 };
        let params =
            SearchParams { futile_stop: true, quiescence_depth: 0, ..SearchParams::default() };
        let mut eval = |states: &[MateMock]| EvalOut {
            priors: states
                .iter()
                .map(|s| s.legal_actions().into_iter().map(|a| (a, 1.0)).collect())
                .collect(),
            values: vec![0.0; states.len()],
            mls: vec![None; states.len()],
        };
        let out = search(&root, Tree::new(), &params, 400, &[], None, 0.0, None, &mut eval);
        assert!(!out.stats.early_stop);
        assert_eq!(out.stats.root_visits, 400);
    }

    #[test]
    fn singleton_prune_and_fallback() {
        let mut t = Tree::new();
        expand_node(&mut t, 0, &[(5, 0.5), (6, 0.5)]);
        // All singletons => fallback to unpruned.
        let c5 = t.arena[0].find(5).unwrap();
        let c6 = t.arena[0].find(6).unwrap();
        backup(&mut t, &[0, c5], -0.5, None);
        backup(&mut t, &[0, c6], -0.5, None);
        let (dense, _, tot) = finalize_visits(&t, 0, true);
        assert!((tot - 2.0).abs() < 1e-12);
        assert!((dense[5] - 0.5).abs() < 1e-6);
        // One singleton among heavies => pruned.
        for _ in 0..5 {
            backup(&mut t, &[0, c5], -0.5, None);
        }
        let (dense, _, _) = finalize_visits(&t, 0, true);
        assert!((dense[6] - 1.0/7.0).abs() < 1e-6);
        assert!((dense[5] - 6.0/7.0).abs() < 1e-6);
    }

    #[test]
    fn tt_promote_prune_evict_and_child_store() {
        let root = Mock { path: vec![] };
        let params = SearchParams::default();
        let mut eval = |states: &[Mock]| mock_eval(states);
        let mut tt = TtStore::new();
        let out = search(&root, Tree::new(), &params, 20, &[], None, 0.0, Some(&mut tt), &mut eval);
        assert!(tt.len() >= 1);
        // Promote the most-visited child and re-search: reused > 0.
        let counts = crate::raw_visits(&out.tree, out.tree.root);
        let best = counts.into_iter().max_by_key(|(_, n)| *n).unwrap().0;
        let mut probe = root.clone();
        probe.apply(best);
        let promoted = tt.promote(probe.rep_key()).expect("child stored");
        let mut eval2 = |states: &[Mock]| mock_eval(states);
        let out2 = search(&root, promoted, &params, 10, &[], None, 0.0, Some(&mut tt), &mut eval2);
        assert!(out2.stats.reused > 0, "{}", out2.stats.reused);
        // prune_to keeps only the promoted line.
        tt.prune_to(probe.rep_key());
        assert_eq!(tt.len(), 1);
        // Eviction bound: insert_root past cap keeps <= 20001.
        for k in 0..20005u64 {
            let t = Tree::new();
            tt.insert_root(k + 1000, &t);
        }
        assert!(tt.len() <= 20001, "{}", tt.len());
    }

    #[test]
    fn empty_legal_returns_zeros_without_eval() {
        let root = Mock { path: vec![1, 2, 3] }; // depth 3 => no legal
        let params = SearchParams::default();
        let mut calls = 0usize;
        let mut eval = |states: &[Mock]| {
            calls += states.len();
            mock_eval(states)
        };
        let out = search(&root, Tree::new(), &params, 10, &[], None, 0.0, None, &mut eval);
        assert_eq!(calls, 0);
        assert!(out.visits.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn kl_and_stats_helpers_sane() {
        let mut t = Tree::new();
        expand_node(&mut t, 0, &[(10, 0.8), (20, 0.2)]);
        let c = t.arena[0].find(10).unwrap();
        backup(&mut t, &[0, c], -0.5, None);
        let (_, counts, tot) = finalize_visits(&t, 0, false);
        let kl = kl_vs_raw(&t, 0, &[10, 20], &counts, tot);
        assert!(kl.is_finite() && kl >= 0.0);
        let (b, e) = crate::breadth_entropy(&counts, tot);
        assert_eq!(b, 1);
        assert!((e - 0.0).abs() < 1e-12);
        let q = root_q(&t, 0, &counts, tot);
        // Caller-negated convention: backup(-0.5) lands w=-0.5 on the
        // child (root sees +0.5); root_q reads the CHILD view => -0.5.
        assert!((q - -0.5).abs() < 1e-12, "{q}");
    }
}
