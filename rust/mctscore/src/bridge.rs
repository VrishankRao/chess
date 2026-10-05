//! V21.1 E1 PyO3 control plane over the dependency-free core.
//!
//! Ownership split (per /tmp/v21_bridge.md batch-queue design):
//! - Rust owns the search TREE (arena, selection, expansion, backup,
//!   quiescence, TT bookkeeping) via `crate::search`.
//! - Python owns BOARDS (all rules queries run against the live
//!   `chess_zero.game.State` object through the [`Position`] impl below,
//!   so movegen/adjudication/contempt semantics are Python's own),
//!   RNG (Dirichlet vectors are drawn from `numpy.random` exactly as
//!   `mcts.search` draws them), and the NN (MPS weights/forward stay
//!   100% Python-side).
//! - Eval vectors cross ONLY through the one `evaluate_fn` callback, one
//!   call per leaf-batch (batch-queue protocol, `max_batch == leaf_batch`).
//!   NO NN in Rust, now or ever.
//!
//! TT-carryover fix (V21.1 audit F1-F9): multi-ply games persist a
//! per-game [`GameStore`] (Rust `TtStore` + eval-cache map) across
//! `search` calls — promote at entry, `insert_root`/`insert_child` refresh
//! at exit (F1), `prune_to` at every `prune_tt` site (F2). Single-position
//! parity harnesses keep passing plain dicts (fresh throwaway store per
//! call, zeros preserved). Dirichlet draws happen iff fresh-expand OR
//! `eps > 0` with `None` passed otherwise (F4, RNG/drop parity). Python
//! `tt`/`eval_cache` dicts stay empty on the Rust path (F2/F6): the store
//! is the source of truth.
//!
//! Offline use only (parity harness); never called by the live loop.
//! `--rust-tree` stays default OFF; best.pt is read-only.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyIterator};
use std::cell::RefCell;
use std::collections::HashMap;

use crate::search::{EvalOut, Position, SearchParams};
use crate::{Action, Tree, TtStore};

// ---------------------------------------------------------------------------
// Python-owned board behind the core `Position` trait
// ---------------------------------------------------------------------------

/// Hash a Python rep key to the u64 the core uses for its history/path/TT
/// sets.
///
/// F5 key domain: `State.rep_key()` is python-chess `transposition_key()`
/// (a tuple of ints: board + turn + castling + EP, no clocks; >= 1.10
/// returns tuples, older returns ints — both hashable). The bridge calls
/// `__hash__` and reinterprets the `isize` as u64 two's-complement
/// (`h as u64`). Tuples of ints hash deterministically IN-PROCESS (int
/// hashes are identity; `PYTHONHASHSEED` randomizes str/bytes only, and
/// keys contain no strs), so a game-leg is self-consistent across its
/// searches; hashes are NOT stable across processes (never persist them
/// to disk). History, path, and TT keys all cross through THIS function,
/// so loop detection and TT promotion agree by construction. 64-bit
/// collisions are negligible (birthday bound ~2^32 entries; the TT caps
/// at 20k, the eval cache at one game's positions). The eval cache keys
/// `(rep_key_hash, rep_count)` — the count joins because rep planes 14-15
/// depend on it (B6) — while TT/history use the bare hash.
fn py_hash_u64(obj: &Bound<'_, PyAny>) -> PyResult<u64> {
    let h: isize = obj.call_method0("__hash__")?.extract()?;
    Ok(h as u64)
}

// ---------------------------------------------------------------------------
// Per-game persistent store (F1/F2/F6): one `GameStore` per game, threaded
// through `search` as `tt` (and `eval_cache`) on the Rust path. Python
// `tt`/`eval_cache` dicts stay empty there (never written); single-search
// harnesses keep passing dicts and get a fresh throwaway store per call.
// ---------------------------------------------------------------------------

/// Per-game Rust-side store: TT across moves + eval-cache across moves.
///
/// Lifetime: create ONE per game (`GameStore()`), pass as `tt=` (and
/// `eval_cache=`) to every `search` of that game, `prune_to(next_rep_key)`
/// after every applied move (mirroring `mcts.prune_tt`), `clear()` on
/// game/agent reset. Never shared across games or processes (one driver
/// per self-play worker). `__len__` reports TT entries; `cache_len`
/// reports eval-cache entries (telemetry only).
#[pyclass]
pub struct GameStore {
    tt: TtStore,
    cache: HashMap<(u64, u64), (Vec<f64>, f64)>,
}

#[pymethods]
impl GameStore {
    #[new]
    fn new() -> Self {
        GameStore { tt: TtStore::new(), cache: HashMap::new() }
    }

    /// Keep ONLY the promoted child's subtree for `next_rep_key`
    /// (mirrors `mcts.prune_tt`). `next_rep_key` is the live
    /// `State.rep_key()` object (tuple); it is hashed with `py_hash_u64`
    /// (F5), so callers never convert keys themselves.
    fn prune_to(&mut self, py: Python<'_>, next_key: Py<PyAny>) -> PyResult<()> {
        let h = py_hash_u64(&next_key.bind(py))?;
        self.tt.prune_to(h);
        Ok(())
    }

    /// Clear TT + eval cache (game/agent reset).
    fn clear(&mut self) {
        self.tt.map.clear();
        self.tt.order.clear();
        self.cache.clear();
    }

    fn __len__(&self) -> usize {
        self.tt.len()
    }

    /// TT entry count (telemetry).
    fn tt_len(&self) -> usize {
        self.tt.len()
    }

    /// Eval-cache entry count (telemetry).
    fn cache_len(&self) -> usize {
        self.cache.len()
    }
}

/// A `chess_zero.game.State` handle. States are immutable in Python
/// (`apply` returns a new object), so `Clone` just clones the handle and
/// `apply` swaps the inner object. All rules answers come from Python.
struct PyState {
    obj: Py<PyAny>,
    /// `chess_zero.selfplay.draw_leaf_value` callable (bound once per search).
    dlv: Py<PyAny>,
    contempt: f64,
    asymmetric: bool,
    edge_scale: f64,
    /// Forcing flags aligned with `legal_moves()` order, computed once per
    /// expanded position (first `forcing_action` call fills it).
    forcing_cache: RefCell<Option<Vec<bool>>>,
}

impl Clone for PyState {
    fn clone(&self) -> Self {
        Python::with_gil(|py| PyState {
            obj: self.obj.clone_ref(py),
            dlv: self.dlv.clone_ref(py),
            contempt: self.contempt,
            asymmetric: self.asymmetric,
            edge_scale: self.edge_scale,
            forcing_cache: RefCell::new(None),
        })
    }
}

impl PyState {
    fn with_gil<T>(&self, f: impl FnOnce(Python<'_>, &Bound<'_, PyAny>) -> PyResult<T>) -> PyResult<T> {
        Python::with_gil(|py| {
            let bound = self.obj.bind(py);
            f(py, &bound)
        })
    }

    fn board_pieces_len(&self, piece_type: u8, color_white: bool) -> PyResult<u32> {
        self.with_gil(|_, bound| {
            let board = bound.getattr("board")?;
            let coll = board.call_method1("pieces", (piece_type, color_white))?;
            let n: usize = coll.call_method0("__len__")?.extract()?;
            Ok(n as u32)
        })
    }

    /// Forcing flags for every legal move, in `legal_moves()` order.
    /// Mirrors `mcts.search._boost_forcing` exactly: capture/check via the
    /// fast queries on the absolute move, promotion, or pawn push to
    /// 6th/7th rank (White to rank idx 5/6, Black to 2/1).
    fn forcing_flags(&self) -> PyResult<Vec<bool>> {
        if let Some(cached) = self.forcing_cache.borrow().clone() {
            return Ok(cached);
        }
        let flags = self.with_gil(|_py, bound| {
            let moves: Vec<Bound<'_, PyAny>> =
                bound.call_method0("_legal_chess_moves")?.extract()?;
            let board = bound.getattr("board")?;
            let turn_white: bool = board.getattr("turn")?.extract()?;
            let mut out = Vec::with_capacity(moves.len());
            for mv in &moves {
                let cap: bool =
                    bound.call_method1("is_capture_fast", (mv,))?.extract()?;
                let chk: bool =
                    bound.call_method1("gives_check_fast", (mv,))?.extract()?;
                let promoted = !mv.getattr("promotion")?.is_none();
                let mut flag = cap || chk || promoted;
                if !flag {
                    let fr: i64 = mv.getattr("from_square")?.extract()?;
                    let to: i64 = mv.getattr("to_square")?.extract()?;
                    let pc = board.call_method1("piece_at", (fr,))?;
                    if !pc.is_none() {
                        let pt: i64 = pc.getattr("piece_type")?.extract()?;
                        if pt == 1 {
                            let rto = (to >> 3) & 7;
                            flag = if turn_white {
                                rto == 5 || rto == 6
                            } else {
                                rto == 2 || rto == 1
                            };
                        }
                    }
                }
                out.push(flag);
            }
            Ok(out)
        })?;
        *self.forcing_cache.borrow_mut() = Some(flags.clone());
        Ok(flags)
    }
}

impl crate::search::Position for PyState {
    fn repetition_draw(&self) -> bool {
        self.with_gil(|_, st| st.call_method0("rep_count")?.extract::<u32>()).expect("rep_count") >= 2
    }
    fn tt_key(&self) -> u64 {
        self.with_gil(|_, bound| py_hash_u64(&bound.call_method0("key")?))
            .expect("State.key must be hashable")
    }
    fn rep_key(&self) -> u64 {
        // Hashed (see py_hash_u64): python-chess returns tuples here.
        self.with_gil(|_, bound| {
            let rk = bound.call_method0("rep_key")?;
            py_hash_u64(&rk)
        })
        .expect("State.rep_key() must be hashable")
    }

    fn legal_actions(&self) -> Vec<Action> {
        self.with_gil(|_, bound| bound.call_method0("legal_moves")?.extract::<Vec<Action>>())
            .expect("State.legal_moves() must return action ints")
    }

    fn apply(&mut self, a: Action) {
        Python::with_gil(|py| {
            let next = self
                .obj
                .bind(py)
                .call_method1("apply", (a,))
                .expect("State.apply(legal) must succeed")
                .unbind();
            self.obj = next;
            *self.forcing_cache.borrow_mut() = None;
        })
    }

    fn stm_terminal_z(&self) -> Option<f64> {
        let (done, z): (bool, f64) = self
            .with_gil(|_, bound| bound.call_method0("is_terminal")?.extract::<(bool, f64)>())
            .expect("State.is_terminal() must return (bool, float)");
        if done {
            Some(z)
        } else {
            None
        }
    }

    fn in_check(&self) -> bool {
        self.with_gil(|_, bound| bound.call_method0("in_check_fast")?.extract::<bool>())
            .expect("State.in_check_fast() must return bool")
    }

    fn piece_count(&self) -> u32 {
        self.with_gil(|_, bound| {
            let board = bound.getattr("board")?;
            let pm = board.call_method0("piece_map")?;
            let dict = pm.downcast::<PyDict>()?;
            Ok(dict.len() as u32)
        })
        .expect("board.piece_map() must return a dict")
    }

    fn white_pawns(&self) -> u32 {
        self.board_pieces_len(1, true).expect("board.pieces() must be sized")
    }

    fn black_pawns(&self) -> u32 {
        self.board_pieces_len(1, false).expect("board.pieces() must be sized")
    }

    fn white_to_move(&self) -> bool {
        self.with_gil(|_, bound| bound.getattr("board")?.getattr("turn")?.extract::<bool>())
            .expect("board.turn must be a bool")
    }

    fn has_promo_threat(&self) -> bool {
        // Either side a pawn one step from queening (White rank idx 6,
        // Black rank idx 1); mirrors mcts._has_promo_threat.
        self.with_gil(|_, bound| {
            let board = bound.getattr("board")?;
            for (color, rank) in [(true, 6i64), (false, 1i64)] {
                let set = board.call_method1("pieces", (1u8, color))?;
                for sq in PyIterator::from_bound_object(&set)? {
                    let sq: i64 = sq?.extract()?;
                    if (sq >> 3) & 7 == rank {
                        return Ok(true);
                    }
                }
            }
            Ok(false)
        })
        .expect("promo-threat scan must succeed")
    }

    fn forcing_action(&self, a: Action) -> bool {
        let flags = self.forcing_flags().expect("forcing snapshot must succeed");
        let legal = self.legal_actions();
        legal.iter().position(|&x| x == a).map(|i| flags[i]).unwrap_or(false)
    }

    fn draw_score_stm(&self) -> f64 {
        self.with_gil(|py, bound| {
            let board = bound.getattr("board")?;
            self.dlv
                .bind(py)
                .call1((board, self.contempt, self.asymmetric, self.edge_scale))?
                .extract::<f64>()
        })
        .expect("draw_leaf_value must return float")
    }
}

// ---------------------------------------------------------------------------
// Eval plumbing: batch-queue crossing + optional caches (F6 per-game)
// ---------------------------------------------------------------------------

/// First error raised while talking to Python. The core `EvalFn` cannot
/// return `Result`, so the closure records the failure, returns a dummy,
/// and the `#[pyfunction]` below converts it into a raised exception.
struct EvalCtx {
    evaluate_fn: Py<PyAny>,
    ml_fn: Option<Py<PyAny>>,
    ml_on: bool,
    /// F6: on the `GameStore` path this map is TAKEN from the store at
    /// search entry and RESTORED at exit, so hits persist across the
    /// game's moves exactly like Python's per-game `eval_cache` dict
    /// (B6 key `(rep_key_hash, rep_count)`, misses share one forward).
    /// On the legacy dict path it is a fresh per-search map (single
    /// searches have no cross-move positions anyway). The Python
    /// `eval_cache` dict itself stays empty on the Rust path (never
    /// written; non-None merely enables the store's cache).
    cache: HashMap<(u64, u64), (Vec<f64>, f64)>,
    use_cache: bool,
    error: Option<String>,
}

fn tolist_f64(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    if let Ok(v) = obj.call_method0("tolist")?.extract::<Vec<f64>>() {
        return Ok(v);
    }
    // Tolerate (B,1)/(1,) shapes: flatten one level.
    let nested: Vec<Vec<f64>> = obj.call_method0("tolist")?.extract()?;
    Ok(nested.into_iter().flat_map(|r| r.into_iter()).collect())
}

fn eval_batch(py: Python<'_>, ctx: &mut EvalCtx, states: &[PyState]) -> EvalOut {
    let fail = |ctx: &mut EvalCtx, msg: String| {
        if ctx.error.is_none() {
            ctx.error = Some(msg);
        }
        EvalOut {
            priors: vec![Vec::new(); states.len()],
            values: vec![0.0; states.len()],
            mls: vec![None; states.len()],
        }
    };
    // B6-style cache key (rep_key, rep_count), mirroring mcts.search.
    let mut keys: Vec<(u64, u64)> = Vec::with_capacity(states.len());
    if ctx.use_cache {
        for s in states {
            let key: PyResult<(u64, u64)> = (|| {
                let bound = s.obj.bind(py);
                let rk = py_hash_u64(&bound.call_method0("key")?)?;
                let rc: u64 = bound.call_method0("rep_count")?.extract()?;
                Ok((rk, rc))
            })();
            match key {
                Ok(k) => keys.push(k),
                Err(e) => return fail(ctx, format!("rep key failed: {e}")),
            }
        }
    }
    let mut rows_p: Vec<Option<Vec<f64>>> = vec![None; states.len()];
    let mut rows_v: Vec<Option<f64>> = vec![None; states.len()];
    let mut miss: Vec<usize> = Vec::new();
    if ctx.use_cache {
        for (i, k) in keys.iter().enumerate() {
            match ctx.cache.get(k) {
                Some((p, v)) => {
                    rows_p[i] = Some(p.clone());
                    rows_v[i] = Some(*v);
                }
                None => miss.push(i),
            }
        }
    } else {
        miss = (0..states.len()).collect();
    }
    if !miss.is_empty() {
        let objs: Vec<Py<PyAny>> = miss.iter().map(|&i| states[i].obj.clone_ref(py)).collect();
        let out = ctx.evaluate_fn.bind(py).call1((objs,));
        let out = match out {
            Ok(o) => o,
            Err(e) => return fail(ctx, format!("evaluate_fn failed: {e}")),
        };
        let (priors_o, values_o): (Bound<'_, PyAny>, Bound<'_, PyAny>) = match out.extract() {
            Ok(t) => t,
            Err(e) => return fail(ctx, format!("evaluate_fn must return (priors, values): {e}")),
        };
        let priors_nested: Vec<Vec<f64>> = match priors_o.call_method0("tolist") {
            Ok(t) => match t.extract::<Vec<Vec<f64>>>() {
                Ok(n) => n,
                Err(_) => match t.extract::<Vec<f64>>() {
                    Ok(f) => vec![f],
                    Err(e2) => {
                        return fail(ctx, format!("priors tolist failed: {e2}"))
                    }
                },
            },
            Err(e) => return fail(ctx, format!("priors tolist failed: {e}")),
        };
        let values_flat: Vec<f64> = match tolist_f64(&values_o) {
            Ok(v) => v,
            Err(e) => return fail(ctx, format!("values tolist failed: {e}")),
        };
        if priors_nested.len() != miss.len() || values_flat.len() != miss.len() {
            return fail(
                ctx,
                format!(
                    "evaluate_fn shape mismatch: {} priors / {} values for {} states",
                    priors_nested.len(),
                    values_flat.len(),
                    miss.len()
                ),
            );
        }
        for (j, &i) in miss.iter().enumerate() {
            rows_p[i] = Some(priors_nested[j].clone());
            rows_v[i] = Some(values_flat[j]);
            if ctx.use_cache {
                if ctx.cache.len() >= 2048 { ctx.cache.clear(); }
                ctx.cache.insert(keys[i], (priors_nested[j].clone(), values_flat[j]));
            }
        }
    }
    // ML side channel: one shared forward per batch (solo on immediate
    // leaves, since the core calls the closure per leaf there too).
    let mut mls: Vec<Option<f64>> = vec![None; states.len()];
    if ctx.ml_on {
        if let Some(ref ml) = ctx.ml_fn {
            let ml = ml.clone_ref(py);
            let objs: Vec<Py<PyAny>> =
                states.iter().map(|s| s.obj.clone_ref(py)).collect();
            match ml.bind(py).call1((objs,)) {
                Ok(o) => match tolist_f64(&o) {
                    Ok(v) => {
                        for (i, x) in v.into_iter().take(states.len()).enumerate() {
                            mls[i] = if x.is_finite() { Some(x) } else { None };
                        }
                    }
                    Err(e) => return fail(ctx, format!("ml_fn tolist failed: {e}")),
                },
                Err(e) => return fail(ctx, format!("ml_fn failed: {e}")),
            }
        }
    }
    let mut priors: Vec<Vec<(Action, f64)>> = Vec::with_capacity(states.len());
    let mut values: Vec<f64> = Vec::with_capacity(states.len());
    for i in 0..states.len() {
        let row = rows_p[i].clone().unwrap_or_default();
        // All 4096 pairs: illegal actions are dropped by the core's
        // legal-list masking, exactly like mcts.search's mask step.
        let pairs: Vec<(Action, f64)> = row
            .into_iter()
            .take(crate::N_ACTIONS)
            .enumerate()
            .map(|(a, p)| (a as Action, p))
            .collect();
        priors.push(pairs);
        values.push(rows_v[i].unwrap_or(0.0));
    }
    EvalOut { priors, values, mls }
}

// ---------------------------------------------------------------------------
// `mctscore.search` — same contract as `mcts.search` (same defaults)
// ---------------------------------------------------------------------------

/// Run MCTS from `state` with the Rust tree. Returns
/// `(pi[4096] list of float32, stats dict)`. See module docs for the
/// ownership split; defaults mirror `mcts.search` exactly.
///
/// `tt`: None (no carry), a dict (legacy single-search: validated, then a
/// fresh throwaway store per call — the dict stays empty), or a
/// `GameStore` (per-game persistence F1: promote at entry, refresh at
/// exit; `prune_to` after each move, F2). `eval_cache`: None (off), a
/// dict (legacy flag: per-search map), or a `GameStore` (F6 per-game map;
/// passing the SAME store as `tt` shares one object — the dict case then
/// stays empty and merely enables the store's cache).
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (state, evaluate_fn, n_sims, *,
                    c_puct = 1.414,
                    dirichlet_alpha = 0.3, dirichlet_eps = 0.25,
                    depths = None, tt = None, history = None,
                    stats = None,
                    contempt = 0.0, asymmetric_contempt = false,
                    quiescence_depth = 2, forcing_bonus = 0.25,
                    contempt_edge_scale = 0.0,
                    leaf_batch = 1, virtual_loss = 1.0,
                    fpu_reduction = 0.0, prune_singletons = false,
                    eval_cache = None,
                    ml_fn = None, ml_slope = 0.0,
                    ml_cap = 0.07, ml_thr = 0.8,
                    futile_stop = false))]
fn search(
    py: Python<'_>,
    state: Py<PyAny>,
    evaluate_fn: Py<PyAny>,
    n_sims: usize,
    c_puct: f64,
    dirichlet_alpha: f64,
    dirichlet_eps: f64,
    depths: Option<Py<PyAny>>,
    tt: Option<Py<PyAny>>,
    history: Option<Vec<Py<PyAny>>>,
    stats: Option<Py<PyAny>>,
    contempt: f64,
    asymmetric_contempt: bool,
    quiescence_depth: usize,
    forcing_bonus: f64,
    contempt_edge_scale: f64,
    leaf_batch: usize,
    virtual_loss: f64,
    fpu_reduction: f64,
    prune_singletons: bool,
    eval_cache: Option<Py<PyAny>>,
    ml_fn: Option<Py<PyAny>>,
    ml_slope: f64,
    ml_cap: f64,
    ml_thr: f64,
    futile_stop: bool,
) -> PyResult<(Vec<f32>, Py<PyDict>)> {
    fn is_store(py: Python<'_>, obj: &Py<PyAny>) -> bool {
        obj.bind(py).downcast::<GameStore>().is_ok()
    }
    let tt_is_store = match &tt {
        Some(t) => is_store(py, t),
        None => false,
    };
    let ec_is_store = match &eval_cache {
        Some(c) => is_store(py, c),
        None => false,
    };
    // Signature parity: non-store values must be dicts (loud TypeError).
    if let Some(ref t) = tt {
        if !tt_is_store && t.bind(py).downcast::<PyDict>().is_err() {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "mctscore.search: tt must be a dict, GameStore, or None",
            ));
        }
    }
    if let Some(ref c) = eval_cache {
        if !ec_is_store && c.bind(py).downcast::<PyDict>().is_err() {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "mctscore.search: eval_cache must be a dict, GameStore, or None",
            ));
        }
    }
    let use_cache = eval_cache.is_some();
    let same_store = match (&tt, &eval_cache) {
        (Some(a), Some(b)) if tt_is_store && ec_is_store => {
            a.bind(py).as_ptr() == b.bind(py).as_ptr()
        }
        _ => false,
    };
    let dlv: Py<PyAny> = py
        .import_bound("chess_zero.selfplay")?
        .getattr("draw_leaf_value")?
        .unbind();
    let root = PyState {
        obj: state,
        dlv,
        contempt,
        asymmetric: asymmetric_contempt,
        edge_scale: contempt_edge_scale,
        forcing_cache: RefCell::new(None),
    };
    let legal0 = root.legal_actions();
    let stats_dict = PyDict::new_bound(py);
    if legal0.is_empty() {
        // Mirrors mcts.search: zeros, no eval, stats untouched (all-zero).
        return Ok((vec![0.0f32; crate::N_ACTIONS], stats_dict.unbind()));
    }
    let history: Vec<u64> = match history {
        Some(h) => h.iter().map(|o| py_hash_u64(&o.bind(py))).collect::<PyResult<_>>()?,
        None => Vec::new(),
    };
    // F1: take persisted TT/cache out of the store(s). No store borrows
    // are held during the search itself (the core calls back into Python
    // for evals); everything is restored afterwards.
    let root_key = root.tt_key();
    let mut local_tt = TtStore::new();
    let mut persisted_tt = false;
    // Cache handling: (taken_map, restore_target). Target is TtStore's own
    // cell, the separate eval_cache cell, or nowhere (legacy per-search).
    let mut local_cache: HashMap<(u64, u64), (Vec<f64>, f64)> = HashMap::new();
    // 0 = nowhere/legacy, 1 = tt store, 2 = separate eval_cache store.
    let mut cache_target: u8 = 0;
    if tt_is_store {
        let cell = tt
            .as_ref()
            .unwrap()
            .bind(py)
            .downcast::<GameStore>()
            .map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err("tt GameStore downcast failed")
            })?;
        local_tt = std::mem::take(&mut cell.borrow_mut().tt);
        persisted_tt = true;
        if use_cache && (!ec_is_store || same_store) {
            // Same object (or dict flag): the tt store owns the cache.
            local_cache = std::mem::take(&mut cell.borrow_mut().cache);
            cache_target = 1;
        }
    }
    if use_cache && ec_is_store && !same_store && !(tt_is_store && cache_target == 1) {
        // Separate eval_cache store (covers tt-not-store + ec-store, and
        // the rare two-different-stores case).
        if !tt_is_store || cache_target == 0 {
            let cell = eval_cache
                .as_ref()
                .unwrap()
                .bind(py)
                .downcast::<GameStore>()
                .map_err(|_| {
                    pyo3::exceptions::PyTypeError::new_err(
                        "eval_cache GameStore downcast failed",
                    )
                })?;
            // When tt is also a (different) store its cache is NOT used;
            // the eval_cache store owns caching.
            if !tt_is_store {
                local_cache = std::mem::take(&mut cell.borrow_mut().cache);
                cache_target = 2;
            } else if cache_target == 0 {
                local_cache = std::mem::take(&mut cell.borrow_mut().cache);
                cache_target = 2;
            }
        }
    }
    // Promote the stored subtree for this position (F1). Fresh when the
    // store has no entry or the entry is unexpanded (Python discards
    // unexpanded stored roots and expands fresh).
    let promoted: Option<Tree> = local_tt.promote(root_key);
    let (start_tree, was_expanded) = match promoted {
        Some(t) if t.node(t.root).expanded => (t, true),
        _ => (Tree::new(), false),
    };
    // F4 RNG/drop parity: draw fresh noise iff this search expands fresh
    // (even when eps == 0 — the draw still advances numpy RNG exactly as
    // mcts.search does) OR eps > 0 on a reuse (remix from raw). Otherwise
    // pass None: no RNG consumed, no remix, no illegal-child drop —
    // matching the `elif dirichlet_eps > 0` gate in mcts.search.
    let need_noise = !was_expanded || dirichlet_eps > 0.0;
    let noise_opt: Option<Vec<f64>> = if need_noise {
        Some(
            py.import_bound("numpy")?
                .getattr("random")?
                .call_method1("dirichlet", (vec![dirichlet_alpha; legal0.len()],))?
                .call_method0("tolist")?
                .extract()?,
        )
    } else {
        None
    };
    let ml_on = ml_fn.is_some() && ml_slope != 0.0;
    let params = SearchParams {
        c_puct,
        leaf_batch,
        virtual_loss,
        fpu_reduction,
        futile_stop,
        quiescence_depth,
        forcing_bonus,
        ml_slope,
        ml_cap,
        ml_thr,
        ml_on,
        prune_singletons,
    };
    let mut ctx = EvalCtx {
        evaluate_fn,
        ml_fn,
        ml_on,
        cache: std::mem::take(&mut local_cache),
        use_cache,
        error: None,
    };
    let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut eval = |batch: &[PyState]| eval_batch(py, &mut ctx, batch);
        crate::search::search(
            &root,
            start_tree,
            &params,
            n_sims,
            &history,
            noise_opt.as_deref(),
            dirichlet_eps,
            Some(&mut local_tt),
            &mut eval,
        )
    })).map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("Rust search failed; discarded partial search state"))?;
    if let Some(msg) = ctx.error {
        // Best-effort restore before raising (keeps the store usable).
        if persisted_tt {
            if let Ok(cell) = tt
                .as_ref()
                .unwrap()
                .bind(py)
                .downcast::<GameStore>()
            {
                cell.borrow_mut().tt = std::mem::take(&mut local_tt);
            }
        }
        if cache_target == 1 {
            if let Ok(cell) = tt
                .as_ref()
                .unwrap()
                .bind(py)
                .downcast::<GameStore>()
            {
                cell.borrow_mut().cache = std::mem::take(&mut ctx.cache);
            }
        } else if cache_target == 2 {
            if let Ok(cell) = eval_cache
                .as_ref()
                .unwrap()
                .bind(py)
                .downcast::<GameStore>()
            {
                cell.borrow_mut().cache = std::mem::take(&mut ctx.cache);
            }
        }
        return Err(pyo3::exceptions::PyRuntimeError::new_err(msg));
    }
    // F1 post-search refresh: `local_tt` already holds the refreshed root
    // + expanded children (inserted by the core); write it back so the
    // next move promotes. Cache likewise (F6 per-game).
    if persisted_tt {
        let cell = tt
            .as_ref()
            .unwrap()
            .bind(py)
            .downcast::<GameStore>()
            .map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err("tt GameStore downcast failed")
            })?;
        cell.borrow_mut().tt = local_tt;
    }
    if cache_target == 1 {
        let cell = tt
            .as_ref()
            .unwrap()
            .bind(py)
            .downcast::<GameStore>()
            .map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err("tt GameStore downcast failed")
            })?;
        cell.borrow_mut().cache = std::mem::take(&mut ctx.cache);
    } else if cache_target == 2 {
        let cell = eval_cache
            .as_ref()
            .unwrap()
            .bind(py)
            .downcast::<GameStore>()
            .map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err(
                    "eval_cache GameStore downcast failed",
                )
            })?;
        cell.borrow_mut().cache = std::mem::take(&mut ctx.cache);
    }
    // Tail in float32, mirroring mcts.search line-for-line: counts are f32,
    // tot is the f32 sum, pi is the f32 quotient (no f64 double-rounding).
    let raw = crate::raw_visits(&out.tree, out.tree.root);
    let mut counts: Vec<(Action, f32)> =
        raw.iter().map(|(a, n)| (*a, *n as f32)).collect();
    if false && prune_singletons {
        for (_, v) in counts.iter_mut() {
            if *v <= 1.0 {
                *v = 0.0;
            }
        }
        if counts.iter().all(|(_, v)| *v == 0.0) {
            counts = raw.iter().map(|(a, n)| (*a, *n as f32)).collect();
        }
    }
    let tot: f32 = counts.iter().map(|(_, v)| *v).sum();
    let mut pi = vec![0.0f32; crate::N_ACTIONS];
    if tot > 0.0 {
        for (a, v) in &counts {
            pi[*a as usize] = *v / tot;
        }
    }
    // Visit-weighted mean child Q from the f32 counts in children order
    // (same operands, same order as mcts.search).
    let mut rq = 0.0f64;
    if tot > 0.0 {
        let mut acc = 0.0f64;
        for (a, v) in &counts {
            if *v > 0.0 {
                if let Some(cid) = out.tree.node(out.tree.root).find(*a) {
                    acc += f64::from(*v) * out.tree.node(cid).q();
                }
            }
        }
        rq = acc / f64::from(tot);
        if !rq.is_finite() {
            rq = 0.0;
        }
    }
    let breadth = counts.iter().filter(|(_, v)| *v > 0.0).count();
    stats_dict.set_item("root_visits", tot as u64)?;
    stats_dict.set_item("reused", out.stats.reused)?;
    stats_dict.set_item("new_sims", out.stats.new_sims)?;
    stats_dict.set_item("extensions", out.stats.extensions)?;
    stats_dict.set_item("early_stop", if out.stats.early_stop { 1 } else { 0 })?;
    stats_dict.set_item("breadth", breadth)?;
    stats_dict.set_item("visit_entropy", out.stats.visit_entropy)?;
    stats_dict.set_item("kl", out.stats.kl)?;
    stats_dict.set_item("root_q", rq)?;
    if let Some(d) = depths {
        let bound = d.bind(py);
        for dep in &out.depths {
            bound.call_method1("append", (*dep,))?;
        }
    }
    if let Some(s) = stats {
        if let Ok(target) = s.bind(py).downcast::<PyDict>() {
            for (k, v) in stats_dict.iter() {
                target.set_item(k, v)?;
            }
        }
    }
    Ok((pi, stats_dict.unbind()))
}

#[pymodule]
fn mctscore(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(search, m)?)?;
    m.add_class::<GameStore>()?;
    // Stale-wheel visibility, mirroring chesscore.
    m.add("AUDIT_SEMANTICS", 2)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
