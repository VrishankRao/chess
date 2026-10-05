"""Config defaults. PDF ranges (blocks 4-6, channels 64/128, action 4096)
held through v13; v14 deliberately breaks the sims 30-50 range (400 sims,
user-directed: the 1500+ goal outranks the knob ranges).
"""
from dataclasses import dataclass


@dataclass
class Config:
    blocks: int = 4
    channels: int = 64
    input_planes: int = 13
    action_size: int = 4096

    sims: int = 40
    c_puct: float = 1.414
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25
    temp_moves: int = 20

    move_cap: int = 300  # plies; sole training termination guarantee
    adjudicate_margin: float = 3.0

    # v3 anti-draw/opening-diversity controls (self-play only, not arena)
    opening_random_moves: int = 0  # unrecorded uniform-random opening plies
    resign_threshold: float | None = None  # stm value below this starts streak
    resign_moves: int = 3

    # v4: LR decay + gating
    lr_milestones: tuple = (25, 40)  # iterations at which lr *= lr_gamma
    lr_gamma: float = 0.1
    gate_games: int = 20  # challenger-vs-incumbent head-to-head per iter
    gate_threshold: float = 0.55  # min score to promote (draws count half)
    gate_opening_moves: int = 6  # random opening plies per gate game

    # v5: material grounding + contempt (self-play targets only)
    contempt: float = 0.0  # penalty on stm for allowing an adjudicated draw
    no_progress_plies: int = 100  # dead-game adjudication (mirror 50-move)
    aux_w: float = 0.1  # weight of the material-head MSE in the loss

    # v6: pure unit-count aux + empirical values from results
    aux_unit_counts: bool = False  # True: aux target counts units, not pawns
    empirical_values: bool = False  # True: adjudication values from results
    values_every: int = 5  # refit cadence (iters), needs 2000+ samples
    eg_frac: float = 0.0  # replay share drawn from endgame deque (<=10 men)
    eg_capacity: int = 20000
    ml_w: float = 0.05  # moves-left head loss weight

    # v7: contempt that bites + policy stays alive + arena measures openly
    adjudicate_min_ply: int = 0  # no-progress adjudication floor (book-blind)
    asymmetric_contempt: bool = False  # draws pay material-ahead side +c
    ent_w: float = 0.0  # policy-entropy bonus weight in the loss
    arena_noise: float = 0.0  # dirichlet eps for arena measurement games
    arena_temp_moves: int = 0  # sample (not argmax) first K arena plies
    gate_incumbent_games: int = 64
    gate_alpha: float = 0.05  # fixed-checkpoint paired test, not run-wide error rate
    gate_book_file: str = ""  # reserved evaluation opening suite
    gate_min_pairs: int = 1
    challenger_selfplay_frac: float = 0.0  # opt-in experiment
    gate_every: int = 1  # full promotion evaluation cadence
    weak_diagnostics_every: int = 1  # additional random/punisher diagnostics
    sample_reuse: float = 1.5
    augment: bool = False
    sf_depth_suite: bool = False  # extra depth1/depth3/elo rungs every 5 iters

    # v8: repetition-visible value head + paper-faithful temperature
    adaptive_contempt: bool = False  # steer contempt to hold ~40% draws

    # v8.1: deterministic-diverse arena (openings, then argmax, no noise)
    arena_opening_moves: int = 0  # random opening plies per arena game

    # v8.2: sparring (tactical punishment in training) + PGN evidence
    sparring_frac: float = 0.0  # share of self-play games vs greedy sparring
    pgn_losses: bool = False  # save arena losses as PGN for autopsy

    # v8.3: tactical solidity first (recaptures into the weights)
    tactical_override: bool = False  # force + teach sound captures in selfplay
    tac_threshold: float = 0.09  # min net 2-ply gain (own value units)
    resign_playout_frac: float = 0.0  # share of games ignoring resign (AZ 10%)

    # v9: veto hangs (teach not-hanging into the weights) + punisher mix
    blunder_veto: bool = False  # replace hanging MCTS picks (2-ply net)

    # v10: quiescence + forcing-first ordering (search depth where it pays)
    quiescence_depth: int = 2  # extra plies answering checks (0 = off)
    forcing_bonus: float = 0.25  # prior scale for checks/captures

    # v11: edge-scaled contempt magnitude (0.0 = legacy fixed magnitude)
    contempt_edge_scale: float = 0.0  # pawns for full contempt

    # v13: finishing school (mate-signal games). Share of pure self-play
    # games run as FULL playouts: no resign, no material adjudication —
    # games end only by mate, rules draws, or the ply cap. The winner
    # must demonstrate the mate; the value head finally sees true mate
    # terminals. Sparring/arena/gate stay truncated (measurement + punishment).
    full_playout_frac: float = 0.0

    # v15: mate-DISTANCE finishing. Mates and shuffles both score z=1.0,
    # so nothing preferred mating; the moves-left head (trained since
    # v6, loss-only) breaks the tie: when clearly ahead, play the
    # searched move minimizing predicted remaining length. Pure RL —
    # spends an existing head, no external data.
    mate_finish: bool = False

    # v16: curated ECO openings (book knowledge, user-approved pragmatic
    # deviation — AZ-pure would use scale). book_path "" = off,
    # "BUILTIN" = packaged lines. book_plies = plies taken unrecorded
    # (like opening_random_moves) from a random line per game.
    book_path: str = ""
    book_plies: int = 0

    # v16.1: batched-leaf MCTS (inference throughput, zero math change).
    # Leaves share one forward (leaf_batch=8) with child-only virtual loss
    # so concurrent descents diverge. 1 = legacy sequential path.
    leaf_batch: int = 1
    virtual_loss: float = 1.0

    # v15.5: king-safety veto (TEETH for the safety head). The head
    # predicts; this guard acts: after tac/veto, among near-best visited
    # moves it rejects the pick if a sibling keeps our king clearly safer
    # (own-safety gap > safety_drop_thr). Self-gating: an untrained head
    # outputs ~0.5 clustered, gaps never reach the threshold, so the veto
    # sleeps until the head actually learns (no iter gate needed).
    safety_veto: bool = False
    safety_drop_thr: float = 0.15

    # v18: SE trunk ratio (0 = plain residual blocks, yesterday).
    # se_ratio=4 on 64ch -> 16 SE channels (Lc0/KataGo doctrine, ~3%).
    se_ratio: int = 0
    # v18: fast/full playout split (KataGo cap randomization). fast_frac
    # of non-mate-signal games run cheap (fast_sims, no noise) for VALUE
    # data; their policy weight is 0 (policy trains on full games only).
    fast_frac: float = 0.0
    fast_sims: int = 100
    # v18: opp-reply aux head loss weight (64-way from-square, lite).
    reply_w: float = 0.05

    # v19: aux-head loss weights as first-class knobs (yesterday = the
    # train_step/model defaults the loop never overrode, so old configs
    # train bit-identically while the loop now passes them explicitly).
    # V19 tunes reply/own and late-culls mob/margin/safety/check (C5).
    own_w: float = 0.1
    margin_w: float = 0.05
    mob_w: float = 0.05
    safe_w: float = 0.05
    soft_w: float = 0.3
    check_w: float = 0.01
    smooth_eps: float = 0.0  # label smoothing over legal moves (train)
    # v19 value-head schedule (C3): value_w0 before lr_drop_games,
    # value_w1 after. Yesterday = 1.0 always.
    value_w0: float = 0.5
    value_w1: float = 1.0
    # v19 search (mcts.search accepts both; 0.0/False = yesterday
    # bit-exact, guarded by the param default in B2/B3).
    fpu_reduction: float = 0.0
    prune_singletons: bool = False
    # v19 D3 in-tree MLH selection bonus (Lc0 moves-left lite). Slope 0 =
    # off = yesterday bit-exact. V19 sets 0.003/0.07/0.8.
    ml_slope: float = 0.0
    ml_cap: float = 0.07
    ml_thr: float = 0.8
    # v19 lineage scheduling (persisted counters; defaults = yesterday).
    progress_check_every: int = 3
    progress_check_games: int = 12
    # v19 game-budget LR (C3). lr_drop_games == 0 selects the milestone
    # schedule = yesterday; V19 sets 7000.
    lr_max: float = 3e-4
    lr_min: float = 3e-5
    lr_drop_games: int = 0
    lr_warmup_steps: int = 1000
    # v19 SPRT gate (C6): elo hypotheses + error rates.
    sprt_elo0: float = 0.0
    sprt_elo1: float = 30.0
    sprt_alpha: float = 0.10
    sprt_beta: float = 0.10
    mirror_gate: bool = False  # V19 master switch: mirrored SPRT gate (C6/C7)
    # v19 PFSP-lite pool (population R1): fraction of self-play-chunk
    # games played vs frozen champion snapshots (max-min robustness)
    # instead of pure self-play. pool_paths set per iter by the loop
    # (last-4 champs); empty = all self (yesterday). Sparring chunks
    # (punisher) are separate and unchanged.
    pool_frac: float = 0.0
    pool_paths: tuple = ()
    # v19 KL anchor + rehearsal mix (C8). Zero fractions = off = yesterday.
    kl_w0: float = 0.0
    kl_games: int = 5000
    rehearsal_frac: float = 0.0
    rehearsal_games: str = "data_sl/games.jsonl"
    teacher_path: str = "checkpoints_warm/warmstart.pt"
    # v19 second sweep (D11/D12/D13): forced-book fraction (train split),
    # external-anchor games appended to lineage checks only (0 = off).
    opening_book_frac: float = 0.0
    # v19 D12: empirical train-split book for forced openings (False =
    # yesterday: only book_path/book_plies drive openings).
    book_empirical: bool = False
    sf_anchor_games: int = 0
    sf_anchor_rung: str = "sf-elo1350"

    # live adjudication table, refit from results (audit round 2: the
    # self-play worker path reads this; run_training refreshes it per iter)
    adjudicate_values: dict | None = None

    # V20 batch-B appends (Agent B owns; Agent A/C share EXACT field names
    # score_mean/score_stdev/root_q/av_logits/td_lambda — contract in the
    # V20 spec. All defaults = yesterday-off so V19 configs train
    # bit-identically; only V20_CONFIG turns them on. NEVER edit V19 above.)
    v20: bool = False  # master switch: position LR + endgame panel AND
    endgame_frac: float = 0.0  # D4 share of self-play games from endgames
    endgame_file: str = "data_sl/endgames.jsonl"  # D4 harvest output
    endgame_panel_pairs: int = 8  # D4 mirrored pairs in promotion gate
    endgame_panel_sims: int = 400  # D4 fixed sims for endgame pairs
    endgame_panel_min: float = 0.50  # D4 min score to pass endgame panel
    endgame_panel_workers: int = 0  # parallel fan-out for the endgame
    # panel (mirrors _panel_gate's play_match_parallel workers). 0/1 =
    # sequential (default OFF = yesterday bit-exact); >1 = ProcessPool
    # fan-out, one job per mirrored pair, with per-game explicit seeds
    # (seed_base + pair_idx*2 + leg) so WLD is bit-identical to
    # sequential. Live runs leave this 0 (unaffected even on re-import).
    playthrough_frac: float = 0.0  # D5 games ignoring resign (NOT full)
    smart_resign: bool = False  # D5 W+ML resign (replaces fixed rule)
    resign_w_thr: float = 0.02  # D5 resign only if W below this
    resign_ml_thr: float = 0.3  # D5 ... AND predicted ML fraction below this
    resign_consec: int = 3  # D5 consecutive failed checks to resign
    resign_every: int = 8  # D5 plies between resign checks
    tail_weight: float = 1.0  # D5 downweight past tail_ply (1.0 = off)
    tail_ply: int = 120  # D5 ply after which tail_weight applies
    tb_path: str = "data_tb"  # D6 Syzygy final home (was /tmp/syzygy)
    tb_ml_cap: int = 200  # D6 DTZ plies cap for the MLH rewrite
    deblunder_thr: float = 0.1  # D8 best_WL-chosen_WL gap triggering repair
    td_lambda: float = 0.5  # D2 short-horizon TD blend weight (A/C wire it)
    av_w: float = 0.1  # D3 action-value distillation weight (C builds it)
    score_w_max: float = 0.05  # D1 score-head ramp ceiling (A wires it)
    score_ramp_steps: int = 5000  # D1 0->max ramp horizon
    lr_drop_positions: int = 0  # D9 0 = game-counted (yesterday); V20 sets it

    buffer_size: int = 50000
    batch_size: int = 256
    lr: float = 1e-3
    l2: float = 1e-4
    grad_clip: float = 1.0

    # V20 Batch A (train core): AV softmax temperature/scale only.
    # score_w_max/score_ramp_steps/td_lambda/av_w live in the Batch-B block
    # above (identical values; single definition there — this block does
    # NOT duplicate them). V20_CONFIG preset + --v20 flag are Agent C/B's.
    # R2/R3/R4/R5 NOT touched by Batch A (outside R1 is forbidden).
    av_temp: float = 2.0
    av_scale: float = 100.0

    # V21 cumulative LLR (OR-path, future-only): Wald bound 2.94 ~= log(19)
    # for alpha=beta=0.05 (Fishtest convention). Append-only; V20_CONFIG
    # untouched (inherits defaults); read via getattr with 2.94 default.
    LLR_BOUND: float = 2.94
    EG_LLR_BOUND: float = 2.94

    # V21.1 Rust-tree loop seam (E4, flag only, default OFF): when True,
    # agent/selfplay search routes through mcts_bridge.RustTreeBackend
    # with identical params; False (default) is yesterday bit-exact.
    rust_tree: bool = False

    # V20.2 R2 rehearsal-AV table (append-only; empty = yesterday
    # bit-exact): path to build_av.py output jsonl ({"id","ply","av":{uci:
    # q8}}). The loop loads it ONCE at run start and fills real AV vectors
    # in the V20 rehearsal-tail overwrite (hit -> AV, miss -> zeros =
    # today). "" = OFF (default-off, loss_av == 0.0). Set via fullrun
    # --av-table. Warm-start already consumes --av-table (unchanged).
    av_table: str = ""

    @property
    def policy_channels(self) -> int:
        return 8

    @property
    def value_channels(self) -> int:
        return 8


TRAIN_CONFIG = Config(blocks=5, channels=128, sims=40)
LOCAL_CONFIG = Config(blocks=4, channels=64, sims=40)
TEST_CONFIG = Config(blocks=2, channels=32, sims=8, buffer_size=1024,
                     batch_size=16)
# v3: same net/sims/loss, but diverse openings + resign + tighter adjudication
V3_CONFIG = Config(blocks=5, channels=128, sims=40, opening_random_moves=6,
                   resign_threshold=-0.85, resign_moves=3,
                   adjudicate_margin=2.0)
# v5: v3 + material head, contempt, no-progress rule (fresh start)
V5_CONFIG = Config(blocks=5, channels=128, sims=40, opening_random_moves=6,
                   resign_threshold=-0.85, resign_moves=3,
                   adjudicate_margin=2.0, contempt=0.1,
                   no_progress_plies=100, aux_w=0.1)
# v6: pure unit-count aux + empirical adjudication values from results
V6_CONFIG = Config(blocks=5, channels=128, sims=40, opening_random_moves=6,
                   resign_threshold=-0.85, resign_moves=3,
                   adjudicate_margin=2.0, contempt=0.1,
                   no_progress_plies=100, aux_w=0.1, aux_unit_counts=True,
                   empirical_values=True, values_every=5, eg_frac=0.25)
# v7: asymmetric contempt + tight screws + entropy + phase plane + open arena
V7_CONFIG = Config(blocks=5, channels=128, input_planes=14, sims=40,
                   opening_random_moves=6, resign_threshold=-0.85,
                   resign_moves=3, adjudicate_margin=1.0, contempt=0.3,
                   no_progress_plies=60, adjudicate_min_ply=16,
                   asymmetric_contempt=True, aux_w=0.1, aux_unit_counts=True,
                   empirical_values=True, values_every=5, eg_frac=0.25,
                   ml_w=0.05, ent_w=0.005, arena_noise=0.05,
                   arena_temp_moves=6, sf_depth_suite=True)
# v8: v7 + repetition-visible value head + AZ-paper temperature (30) +
# adaptive contempt. Warm-starts from v7-best (14ch -> 16ch surgery).
V8_CONFIG = Config(blocks=5, channels=128, input_planes=16, sims=40,
                   temp_moves=30, opening_random_moves=6,
                   resign_threshold=-0.85, resign_moves=3,
                   adjudicate_margin=1.0, contempt=0.3,
                   no_progress_plies=60, adjudicate_min_ply=16,
                   asymmetric_contempt=True, aux_w=0.1, aux_unit_counts=True,
                   empirical_values=True, values_every=5, eg_frac=0.25,
                   ml_w=0.05, ent_w=0.005, arena_noise=0.05,
                   arena_temp_moves=6, sf_depth_suite=True,
                   adaptive_contempt=True)
# v8.1: v8 training, but the arena measures deterministically (argmax, no
# noise, no sampling) from diverse openings. Same blindness cure as the open
# arena, without the ±400-Elo opening-luck noise that made v8 unreadable.
V8_1_CONFIG = Config(blocks=5, channels=128, input_planes=16, sims=40,
                     temp_moves=30, opening_random_moves=6,
                     resign_threshold=-0.85, resign_moves=3,
                     adjudicate_margin=1.0, contempt=0.3,
                     no_progress_plies=60, adjudicate_min_ply=16,
                     asymmetric_contempt=True, aux_w=0.1, aux_unit_counts=True,
                     empirical_values=True, values_every=5, eg_frac=0.25,
                     ml_w=0.05, ent_w=0.005, arena_noise=0.0,
                     arena_temp_moves=0, arena_opening_moves=6,
                     sf_depth_suite=True,
                     adaptive_contempt=True)
# v8.2: v8.1 + sparring vs greedy (15%) + PGN loss autopsy + white-share log.
V8_2_CONFIG = Config(blocks=5, channels=128, input_planes=16, sims=40,
                     temp_moves=30, opening_random_moves=6,
                     resign_threshold=-0.85, resign_moves=3,
                     adjudicate_margin=1.0, contempt=0.3,
                     no_progress_plies=60, adjudicate_min_ply=16,
                     asymmetric_contempt=True, aux_w=0.1, aux_unit_counts=True,
                     empirical_values=True, values_every=5, eg_frac=0.25,
                     ml_w=0.05, ent_w=0.005, arena_noise=0.0,
                     arena_temp_moves=0, arena_opening_moves=6,
                     sf_depth_suite=True,
                     adaptive_contempt=True, sparring_frac=0.15,
                     pgn_losses=True)
# v8.3: v8.2 + tactical override (sound captures forced + taught) + 10%
# resign-playout (bad positions stay in the buffer).
V8_3_CONFIG = Config(blocks=5, channels=128, input_planes=16, sims=40,
                     temp_moves=30, opening_random_moves=6,
                     resign_threshold=-0.85, resign_moves=3,
                     adjudicate_margin=1.0, contempt=0.3,
                     no_progress_plies=60, adjudicate_min_ply=16,
                     asymmetric_contempt=True, aux_w=0.1, aux_unit_counts=True,
                     empirical_values=True, values_every=5, eg_frac=0.25,
                     ml_w=0.05, ent_w=0.005, arena_noise=0.0,
                     arena_temp_moves=0, arena_opening_moves=6,
                     sf_depth_suite=True,
                     adaptive_contempt=True, sparring_frac=0.15,
                     pgn_losses=True, tactical_override=True,
                     tac_threshold=0.09, resign_playout_frac=0.1)
# v8.4: v8.3 training + environment-integrity fixes (audit IMPROVEMENTS.md):
# rules-draws before adjudication, real repetition history (threefold ends
# games), PUCT sqrt(max(1,n)), rep planes identical train/inference,
# ADJUDICATE_VALUES stamped to workers, paired arena/gate openings,
# single-worker path mirrors parallel, temp unified to plies, dead L2
# removed, tactical scan on pure unit counts, optimizer+schedule resume.
V8_4_CONFIG = Config(blocks=5, channels=128, input_planes=16, sims=40,
                     temp_moves=30, opening_random_moves=6,
                     resign_threshold=-0.85, resign_moves=3,
                     adjudicate_margin=1.0, contempt=0.3,
                     no_progress_plies=60, adjudicate_min_ply=16,
                     asymmetric_contempt=True, aux_w=0.1, aux_unit_counts=True,
                     empirical_values=True, values_every=5, eg_frac=0.25,
                     ml_w=0.05, ent_w=0.005, arena_noise=0.0,
                     arena_temp_moves=0, arena_opening_moves=6,
                     sf_depth_suite=True,
                     adaptive_contempt=True, sparring_frac=0.15,
                     pgn_losses=True, tactical_override=True,
                     tac_threshold=0.09, resign_playout_frac=0.1)
# v8.5: v8.4 training + audit round 2: fitted values reach self-play
# (structural GAME_GLOBALS stamp), tactical scoring on the fitted table,
# real subtree reuse (child promotion + raw-prior remix), iter-seeded
# openings, opt/sched in champion files, decisive-only value refit,
# history-threaded UCI. Same numerics; all fixes are code.
V8_5_CONFIG = Config(blocks=5, channels=128, input_planes=16, sims=40,
                     temp_moves=30, opening_random_moves=6,
                     resign_threshold=-0.85, resign_moves=3,
                     adjudicate_margin=1.0, contempt=0.3,
                     no_progress_plies=60, adjudicate_min_ply=16,
                     asymmetric_contempt=True, aux_w=0.1, aux_unit_counts=True,
                     empirical_values=True, values_every=5, eg_frac=0.25,
                     ml_w=0.05, ent_w=0.005, arena_noise=0.0,
                     arena_temp_moves=0, arena_opening_moves=6,
                     sf_depth_suite=True,
                     adaptive_contempt=True, sparring_frac=0.15,
                     pgn_losses=True, tactical_override=True,
                     tac_threshold=0.09, resign_playout_frac=0.1)
# v8.6: v8.5 + audit round 4 (§6.1-§6.8):
# - tt pruned to promoted child every move (bounds memory, kills
#   history-merge hazard); effective visits tracked in search stats.
#   Nominal sims stay 40 (PDF 30-50 locks the nominal knob); effective
#   visits run higher with reuse — documented deviation, measurable via
#   tt=None arenas (rescore --no-reuse).
# - corroborated promotion: gate (noisy selection) AND arena
#   non-regression (deterministic measurement, offset seeds). Gate seed =
#   arena seed + 500k so the two tests use different openings.
# - distance-weighted decisive refit (w=1/(1+2*ml)); dead guard removed.
# - train steps scaled by buffer occupancy (no more 500 full-LR steps on
#   one iter's 18k positions after a cut).
V8_6_CONFIG = Config(blocks=5, channels=128, input_planes=16, sims=40,
                     temp_moves=30, opening_random_moves=6,
                     resign_threshold=-0.85, resign_moves=3,
                     adjudicate_margin=1.0, contempt=0.3,
                     no_progress_plies=60, adjudicate_min_ply=16,
                     asymmetric_contempt=True, aux_w=0.1, aux_unit_counts=True,
                     empirical_values=True, values_every=5, eg_frac=0.25,
                     ml_w=0.05, ent_w=0.005, arena_noise=0.0,
                     arena_temp_moves=0, arena_opening_moves=6,
                     sf_depth_suite=True,
                     adaptive_contempt=True, sparring_frac=0.15,
                     pgn_losses=True, tactical_override=True,
                     tac_threshold=0.09, resign_playout_frac=0.1)
# v8.7: v8.6 training + inference speedup phase 1 (§8.1): worker-side
# frozen inference (torch.jit.freeze in make_evaluate, CPU only, ~2x on
# batch-1 — VM-verified 6.0 -> 3.0 ms). Same numerics, same targets —
# ENVIRONMENT change, not an engine change: frozen inference is ~1e-5
# equivalent (argmax 256/256, battery identical) but not bit-identical,
# so pre/post-v8.7 arena numbers are not same-engine evidence.
# Kill switch CHESS_ZERO_NO_JIT=1 restores eager.
V8_7_CONFIG = Config(blocks=5, channels=128, input_planes=16, sims=40,
                     temp_moves=30, opening_random_moves=6,
                     resign_threshold=-0.85, resign_moves=3,
                     adjudicate_margin=1.0, contempt=0.3,
                     no_progress_plies=60, adjudicate_min_ply=16,
                     asymmetric_contempt=True, aux_w=0.1, aux_unit_counts=True,
                     empirical_values=True, values_every=5, eg_frac=0.25,
                     ml_w=0.05, ent_w=0.005, arena_noise=0.0,
                     arena_temp_moves=0, arena_opening_moves=6,
                     sf_depth_suite=True,
                     adaptive_contempt=True, sparring_frac=0.15,
                     pgn_losses=True, tactical_override=True,
                     tac_threshold=0.09, resign_playout_frac=0.1)
# v8.8: v8.7 training + inference speedup phase 3 (§8.3): batched GPU
# inference server for self-play/arena/gate workers (one server per
# phase, built from that phase's weights — staleness impossible by
# construction). Same model/search/targets, fp32 ULP-identical to CPU
# (GPU-vs-CPU 9.5e-07, L4 ladder 15.1k/31.1k/43.8k pos/s — verified
# pre-cut). New version number because wall-clock per phase changes what
# gets measured per unit time, not because the engine plays differently.
V8_8_CONFIG = Config(blocks=5, channels=128, input_planes=16, sims=40,
                     temp_moves=30, opening_random_moves=6,
                     resign_threshold=-0.85, resign_moves=3,
                     adjudicate_margin=1.0, contempt=0.3,
                     no_progress_plies=60, adjudicate_min_ply=16,
                     asymmetric_contempt=True, aux_w=0.1, aux_unit_counts=True,
                     empirical_values=True, values_every=5, eg_frac=0.25,
                     ml_w=0.05, ent_w=0.005, arena_noise=0.0,
                     arena_temp_moves=0, arena_opening_moves=6,
                     sf_depth_suite=True,
                     adaptive_contempt=True, sparring_frac=0.15,
                     pgn_losses=True, tactical_override=True,
                     tac_threshold=0.09, resign_playout_frac=0.1)
# v9: search contempt replaces label contempt (pure value targets) +
# blunder veto + greedy/punisher sparring mix (0.25) + 18-plane input
# (stm castling rights) + stm-oriented codec flip (one policy both colors;
# mirror asymmetry exec-confirmed at MAE 0.69). Training-algorithm change:
# new version number, warm-start surgery 16->18ch from v88 best.
V9_CONFIG = Config(blocks=5, channels=128, input_planes=18, sims=40,
                   temp_moves=30, opening_random_moves=6,
                   resign_threshold=-0.85, resign_moves=3,
                   adjudicate_margin=1.0, contempt=0.3,
                   no_progress_plies=60, adjudicate_min_ply=16,
                   asymmetric_contempt=True, aux_w=0.1, aux_unit_counts=True,
                   empirical_values=True, values_every=5, eg_frac=0.25,
                   ml_w=0.05, ent_w=0.005, arena_noise=0.0,
                   arena_temp_moves=0, arena_opening_moves=6,
                   sf_depth_suite=True,
                   adaptive_contempt=True, sparring_frac=0.25,
                   pgn_losses=True, tactical_override=True,
                   tac_threshold=0.09, resign_playout_frac=0.1,
                   blunder_veto=True)
# v10: planning package — ownership + margin + mobility aux heads (dense
# spatial supervision), check-evasion quiescence + forcing-first ordering
# (same 40-sim budget), veto fire-rate telemetry. Same 18 planes/codec as
# v9: warm start needs no surgery (verbatim restore).
V10_CONFIG = Config(blocks=5, channels=128, input_planes=18, sims=40,
                    temp_moves=30, opening_random_moves=6,
                    resign_threshold=-0.85, resign_moves=3,
                    adjudicate_margin=1.0, contempt=0.3,
                    no_progress_plies=60, adjudicate_min_ply=16,
                    asymmetric_contempt=True, aux_w=0.1, aux_unit_counts=True,
                    empirical_values=True, values_every=5, eg_frac=0.25,
                    ml_w=0.05, ent_w=0.005, arena_noise=0.0,
                    arena_temp_moves=0, arena_opening_moves=6,
                    sf_depth_suite=True,
                    adaptive_contempt=True, sparring_frac=0.25,
                    pgn_losses=True, tactical_override=True,
                    tac_threshold=0.09, resign_playout_frac=0.1,
                    blunder_veto=True,
                    quiescence_depth=2, forcing_bonus=0.25)

# v11: edge-scaled contempt magnitude (draw_leaf_value scales ±c by
# min(1, |edge|/3): won positions hate draws at full strength, level
# positions don't care). Fixes the v10-film failure: adaptive steering
# had relaxed contempt to its 0.1 floor on globally-rare draws, leaving
# nothing to avoid a draw in a specifically won endgame. Same 18 planes,
# heads and search as v10 otherwise.
V11_CONFIG = Config(blocks=5, channels=128, input_planes=18, sims=40,
                    temp_moves=30, opening_random_moves=6,
                    resign_threshold=-0.85, resign_moves=3,
                    adjudicate_margin=1.0, contempt=0.3,
                    no_progress_plies=60, adjudicate_min_ply=16,
                    asymmetric_contempt=True, aux_w=0.1, aux_unit_counts=True,
                    empirical_values=True, values_every=5, eg_frac=0.25,
                    ml_w=0.05, ent_w=0.005, arena_noise=0.0,
                    arena_temp_moves=0, arena_opening_moves=6,
                    sf_depth_suite=True,
                    adaptive_contempt=True, sparring_frac=0.25,
                    pgn_losses=True, tactical_override=True,
                    tac_threshold=0.09, resign_playout_frac=0.1,
                    blunder_veto=True,
                    quiescence_depth=2, forcing_bonus=0.25,
                    contempt_edge_scale=3.0)

# v12: planning & tactical hardening:
# - capture extensions in MCTS quiescence (tactical lines extended)
# - ownership map coordinate flip fix (proper spatial supervision for Black)
# - stm-relative margin head (consistent canonical polarity)
# - pawn-anchored empirical value fitting (kills deflation collapse)
# - opponent castling (18-19) + en-passant (20) planes: 18 -> 21ch.
#   Warm-start surgery zero-inits the 3 new channels (old 18ch checkpoints
#   load verbatim); growth applies ONLY to trunk_in.
# - Apple Silicon M1 / low-RAM adaptation
V12_CONFIG = Config(blocks=5, channels=128, input_planes=21, sims=40,
                    temp_moves=30, opening_random_moves=6,
                    resign_threshold=-0.85, resign_moves=3,
                    adjudicate_margin=1.0, contempt=0.3,
                    no_progress_plies=60, adjudicate_min_ply=16,
                    asymmetric_contempt=True, aux_w=0.1, aux_unit_counts=True,
                    empirical_values=True, values_every=5, eg_frac=0.25,
                    ml_w=0.05, ent_w=0.005, arena_noise=0.0,
                    arena_temp_moves=0, arena_opening_moves=6,
                    sf_depth_suite=True,
                    adaptive_contempt=True, sparring_frac=0.25,
                    pgn_losses=True, tactical_override=True,
                    tac_threshold=0.09, resign_playout_frac=0.1,
                    blunder_veto=True,
                    quiescence_depth=2, forcing_bonus=0.25,
                    contempt_edge_scale=3.0)

# MAC_CONFIG: Optimized for Apple Silicon M1 with 8GB RAM.
# Uses 4x64 blocks/channels (lightweight ~150MB net) and a bounded replay buffer
# (20,000 positions, ~420MB RAM) with batch 128 (fast on MPS/CPU) so training and
# self-play run smoothly without memory compression or disk swap thrashing.
# v12 planes (21ch: +opp castling + EP) — same growth surgery as V12.
MAC_CONFIG = Config(blocks=4, channels=64, input_planes=21, sims=40,
                    temp_moves=30, opening_random_moves=6,
                    resign_threshold=-0.85, resign_moves=3,
                    adjudicate_margin=1.0, contempt=0.3,
                    no_progress_plies=60, adjudicate_min_ply=16,
                    asymmetric_contempt=True, aux_w=0.1, aux_unit_counts=True,
                    empirical_values=True, values_every=5, eg_frac=0.25,
                    ml_w=0.05, ent_w=0.005, arena_noise=0.0,
                    arena_temp_moves=0, arena_opening_moves=6,
                    sf_depth_suite=False,
                    adaptive_contempt=True, sparring_frac=0.25,
                    pgn_losses=True, tactical_override=True,
                    tac_threshold=0.09, resign_playout_frac=0.1,
                    blunder_veto=True,
                    quiescence_depth=2, forcing_bonus=0.25,
                    contempt_edge_scale=3.0,
                    buffer_size=20000, batch_size=128)

# v13: finishing school on the M1 footprint (4x64, 21ch — warm-starts
# verbatim from MAC/v12-checkpoint lineage, no surgery):
# - full_playout_frac 0.15: ~3 of every 20 self-play games play to a
#   TRUE ending (mate / rules-draw / cap). Mate terminals + conversion
#   practice enter the buffer for the first time ever.
# - adjudicate_margin 1.0 -> 3.0: truncated games end only when truly
#   decided; +1..+3 positions must be PLAYED, not counted. Draws still
#   truncate on schedule (no-progress rule unchanged) so iter time holds.
# - terminal-reason counters (Tmate/Tresign/...) flow into history.json
#   as term/mate_rate: the honest compass, greedy arena secondary.
V13_CONFIG = Config(blocks=4, channels=64, input_planes=21, sims=40,
                    temp_moves=30, opening_random_moves=6,
                    resign_threshold=-0.85, resign_moves=3,
                    adjudicate_margin=3.0, contempt=0.3,
                    no_progress_plies=60, adjudicate_min_ply=16,
                    asymmetric_contempt=True, aux_w=0.1, aux_unit_counts=True,
                    empirical_values=True, values_every=5, eg_frac=0.25,
                    ml_w=0.05, ent_w=0.005, arena_noise=0.0,
                    arena_temp_moves=0, arena_opening_moves=6,
                    sf_depth_suite=False,
                    adaptive_contempt=True, sparring_frac=0.25,
                    pgn_losses=True, tactical_override=True,
                    tac_threshold=0.09, resign_playout_frac=0.1,
                    blunder_veto=True,
                    quiescence_depth=2, forcing_bonus=0.25,
                    contempt_edge_scale=3.0,
                    full_playout_frac=0.15,
                    buffer_size=20000, batch_size=128)

# v14: strength over PDF-faithfulness (explicit deviation, user-directed):
# sims 40 -> 400, breaking the proposal's locked 30-50 range. The goal
# (1500+) outranks the knob ranges: 40-sim search sees 1-2 plies and
# cannot resolve tactics or mates, so its targets cap the whole lineage.
# Measured on M1 CPU-frozen 4x64: 400 sims ~= 3.3 s/move, i.e. ~45-70
# min/iter at 20 games x6 workers — weekend-scale training, accepted.
# Same 4x64/21ch arch: warm-starts verbatim, no surgery. Play side moves
# to CPU too (MPS-eager loses 2-3x on this net shape) at 400 sims.
V14_CONFIG = Config(blocks=4, channels=64, input_planes=21, sims=400,
                    temp_moves=30, opening_random_moves=6,
                    resign_threshold=-0.85, resign_moves=3,
                    adjudicate_margin=3.0, contempt=0.3,
                    no_progress_plies=60, adjudicate_min_ply=16,
                    asymmetric_contempt=True, aux_w=0.1, aux_unit_counts=True,
                    empirical_values=True, values_every=5, eg_frac=0.25,
                    ml_w=0.05, ent_w=0.005, arena_noise=0.0,
                    arena_temp_moves=0, arena_opening_moves=6,
                    sf_depth_suite=False,
                    adaptive_contempt=True, sparring_frac=0.25,
                    pgn_losses=True, tactical_override=True,
                    tac_threshold=0.09, resign_playout_frac=0.1,
                    blunder_veto=True,
                    quiescence_depth=2, forcing_bonus=0.25,
                    contempt_edge_scale=3.0,
                    full_playout_frac=0.15,
                    buffer_size=20000, batch_size=128)

# v15: v14 + ML-guided finishing (mate_finish). Same arch/knobs/sims —
# warm-starts verbatim. The mover that is clearly ahead now minimizes
# predicted remaining length among searched moves (self-play, arena,
# gate, UCI alike), so conversion technique enters the targets.
V15_CONFIG = Config(blocks=4, channels=64, input_planes=21, sims=400,
                    temp_moves=30, opening_random_moves=6,
                    resign_threshold=-0.85, resign_moves=3,
                    adjudicate_margin=3.0, contempt=0.3,
                    no_progress_plies=60, adjudicate_min_ply=16,
                    asymmetric_contempt=True, aux_w=0.1, aux_unit_counts=True,
                    empirical_values=True, values_every=5, eg_frac=0.25,
                    ml_w=0.05, ent_w=0.005, arena_noise=0.0,
                    arena_temp_moves=0, arena_opening_moves=6,
                    sf_depth_suite=False,
                    adaptive_contempt=True, sparring_frac=0.25,
                    pgn_losses=True, tactical_override=True,
                    tac_threshold=0.09, resign_playout_frac=0.1,
                    blunder_veto=True,
                    quiescence_depth=2, forcing_bonus=0.25,
                    contempt_edge_scale=3.0,
                    full_playout_frac=0.15,
                    mate_finish=True,
                    buffer_size=20000, batch_size=128)

# v16: the SF-films cut (0-8 vs 1350 exposed self-destructive openings,
# 0/8 castling, tactics collapsing past guard range).
# - 6x128 capacity (PDF max), warm-started from the 4x64 lineage via
#   exact growth surgery (zero-pad channels, identity new blocks —
#   verified bit-exact, so training continues, not restarts).
# - King-safety aux head (safe_w 0.05): dense rules-derived supervision
#   for castling/king danger. New keys random-init (small head).
# - Curated ECO openings (book BUILTIN, 8 plies unrecorded) INSTEAD of
#   uniform-random plies: real positions, same diversity mechanism.
# - adjudicate_margin inf (AZ-faithful): decisive games play out to
#   mate/rules/cap; only draws truncate. Draws still end on schedule
#   so iter time holds.
# - Deployment book in uci --book (same file). Pragmatic deviations
#   (book knowledge for compute), user-approved, training stays pure.
V16_CONFIG = Config(blocks=6, channels=128, input_planes=21, sims=400,
                    temp_moves=30, opening_random_moves=0,
                    resign_threshold=-0.85, resign_moves=3,
                    adjudicate_margin=float("inf"), contempt=0.3,
                    no_progress_plies=60, adjudicate_min_ply=16,
                    asymmetric_contempt=True, aux_w=0.1, aux_unit_counts=True,
                    empirical_values=True, values_every=5, eg_frac=0.25,
                    ml_w=0.05, ent_w=0.005, arena_noise=0.0,
                    arena_temp_moves=0, arena_opening_moves=6,
                    sf_depth_suite=False,
                    adaptive_contempt=True, sparring_frac=0.25,
                    pgn_losses=True, tactical_override=True,
                    tac_threshold=0.09, resign_playout_frac=0.1,
                    blunder_veto=True,
                    quiescence_depth=2, forcing_bonus=0.25,
                    contempt_edge_scale=3.0,
                    full_playout_frac=0.15,
                     mate_finish=True,
                     book_path="BUILTIN", book_plies=8,
                     leaf_batch=8, virtual_loss=1.0,
                     buffer_size=20000, batch_size=128)

# v18: the final-bot cut (v17+v18 literature package, single target).
# - 6x64 SE trunk (PDF max blocks; SE = KataGo global-pooling lite),
#   30 input planes (v17 tactical: attacks/defends/hang/checkers/
#   king-danger/material/opp-bishops/checkerboard/legal-from).
# - WDL value head (Lc0 v0.21: Q=W-L replaces scalar tanh).
# - Opp-reply aux head (KataGo opp-policy lite, train-only signal).
# - Fast/full playout split (KataGo cap randomization: fast games feed
#   value+aux, policy trains on full games only).
# - v15.5 inheritance: mate_finish, leaf_batch 8, safety_veto, margin
#   3.0, random plies (training book OFF by default — empirical 200-line
#   book artifact ships alongside, enable by setting book_path).
# Trains from SL warm-start (fresh 6x64/30ch/SE weights), NOT from old
# lineage (arch differs in planes/heads/blocks — surgery loads them for
# measurement, but training starts warm).
V18_CONFIG = Config(blocks=6, channels=64, input_planes=30, se_ratio=4,
                    sims=400, temp_moves=30, opening_random_moves=6,
                    resign_threshold=-0.85, resign_moves=3,
                    adjudicate_margin=3.0, contempt=0.3,
                    no_progress_plies=60, adjudicate_min_ply=16,
                    asymmetric_contempt=True, aux_w=0.1, aux_unit_counts=True,
                    empirical_values=True, values_every=5, eg_frac=0.25,
                    ml_w=0.05, ent_w=0.005, arena_noise=0.0,
                    arena_temp_moves=0, arena_opening_moves=6,
                    sf_depth_suite=False,
                    adaptive_contempt=True, sparring_frac=0.25,
                    pgn_losses=True, tactical_override=True,
                    tac_threshold=0.09, resign_playout_frac=0.1,
                    blunder_veto=True,
                    quiescence_depth=2, forcing_bonus=0.25,
                    contempt_edge_scale=3.0,
                    full_playout_frac=0.15,
                    mate_finish=True,
                    leaf_batch=8, virtual_loss=1.0,
                    safety_veto=True, safety_drop_thr=0.15,
                    fast_frac=0.5, fast_sims=100, reply_w=0.05,
                    buffer_size=20000, batch_size=128)
# v19: the final-bot cut (V18 base + the ranked literature wins):
# - c_puct 1.6, FPU 0.5, singleton pruning (search);
# - ent 0.002 + label smoothing 0.01 + value schedule 0.5->1.0 (loss);
# - aux final weights (D24): material (aux_w) 0.01, own 0.12, reply 0.15,
#   plies (ml_w) 0.05, margin 0.02, mobility/safety (mob/safe) 0.01,
#   soft 0.3, check 0.01; mob/safety/margin/check late-cull to 0.005
#   (material already at floor, no cull needed);
# - game-budget LR (3e-4 warmup -> const -> 3e-5 @7000 games, milestones
#   retired); dirichlet 0.25->0.12 @3000 games; PANEL gate (D8: 8 inc +
#   6 ancestor + 4 punisher + 2 random, 60/55/55 + Wilson) with noise 0;
# - PFSP-lite pool (D9: 12 self / 5 pool / 3 punisher of 20) in the loop
#   sequential self-play path (parallel paths: Impl-B's file, documented);
# - forced book 30% train-split (D12 knob; selfplay application: Impl-B),
#   gate pairs from the holdout split (D11); SF anchor on lineage (D13);
# - KL anchor + 20% SL rehearsal decaying over 5000 games; PreciseBN-lite
#   before gate; sampling-ratio step cap. Trains from SL warm-start
#   (checkpoints_warm/warmstart.pt), NOT from old lineage.
V19_CONFIG = Config(blocks=6, channels=64, input_planes=30, se_ratio=4,
                    sims=400, temp_moves=30, opening_random_moves=6,
                    resign_threshold=-0.85, resign_moves=3,
                    adjudicate_margin=3.0, contempt=0.3,
                    no_progress_plies=60, adjudicate_min_ply=16,
                    asymmetric_contempt=True, aux_w=0.01, aux_unit_counts=True,
                    empirical_values=True, values_every=5, eg_frac=0.25,
                    ml_w=0.05, ent_w=0.002, arena_noise=0.0,
                    arena_temp_moves=0, arena_opening_moves=6,
                    sf_depth_suite=False,
                    adaptive_contempt=True, sparring_frac=0.25,
                    pgn_losses=True, tactical_override=True,
                    tac_threshold=0.09, resign_playout_frac=0.1,
                    blunder_veto=True,
                    quiescence_depth=2, forcing_bonus=0.25,
                    contempt_edge_scale=3.0,
                    full_playout_frac=0.15,
                    mate_finish=True,
                    leaf_batch=8, virtual_loss=1.0,
                    safety_veto=True, safety_drop_thr=0.15,
                    fast_frac=0.5, fast_sims=100, reply_w=0.15,
                    pool_frac=0.25,
                    own_w=0.12, margin_w=0.02, mob_w=0.01, safe_w=0.01,
                    c_puct=1.6, fpu_reduction=0.5, prune_singletons=True,
                    ml_slope=0.003, ml_cap=0.07, ml_thr=0.8,
                    smooth_eps=0.01, value_w0=0.5, value_w1=1.0,
                    soft_w=0.3, check_w=0.01,
                    kl_w0=0.2, kl_games=5000, rehearsal_frac=0.2,
                    rehearsal_games="data_sl/games.jsonl",
                    teacher_path="checkpoints_warm/warmstart.pt",
                    opening_book_frac=0.3,
                    book_empirical=True,
                    sf_anchor_games=10, sf_anchor_rung="sf-elo1350",
                    lr_max=3e-4, lr_min=3e-5, lr_drop_games=7000,
                    lr_warmup_steps=1000,
                    sprt_elo0=0.0, sprt_elo1=30.0,
                    sprt_alpha=0.10, sprt_beta=0.10,
                    mirror_gate=True,
                    buffer_size=20000, batch_size=128)
# v16 proved capacity wasn't the constraint (6x128 champ -280 to 4x64;
# book narrowed the distribution; safety head had no teeth). Back to the
# 4x64 trunk that learns (warm-start iter169), training distribution
# restored (random plies, NO book — deployment keeps its book), margin
# back to 3.0, and the v16 ideas worth keeping: leaf_batch=8 (proven
# 1.5x, math-neutral) + safety head WITH teeth (safety_veto: the guard
# rejects king-walks once the head learns; self-gating while random).
V15_5_CONFIG = Config(blocks=4, channels=64, input_planes=21, sims=400,
                    temp_moves=30, opening_random_moves=6,
                    resign_threshold=-0.85, resign_moves=3,
                    adjudicate_margin=3.0, contempt=0.3,
                    no_progress_plies=60, adjudicate_min_ply=16,
                    asymmetric_contempt=True, aux_w=0.1, aux_unit_counts=True,
                    empirical_values=True, values_every=5, eg_frac=0.25,
                    ml_w=0.05, ent_w=0.005, arena_noise=0.0,
                    arena_temp_moves=0, arena_opening_moves=6,
                    sf_depth_suite=False,
                    adaptive_contempt=True, sparring_frac=0.25,
                    pgn_losses=True, tactical_override=True,
                    tac_threshold=0.09, resign_playout_frac=0.1,
                    blunder_veto=True,
                    quiescence_depth=2, forcing_bonus=0.25,
                    contempt_edge_scale=3.0,
                    full_playout_frac=0.15,
                    mate_finish=True,
                    leaf_batch=8, virtual_loss=1.0,
                    safety_veto=True, safety_drop_thr=0.15,
                    buffer_size=20000, batch_size=128)
# V20 batch-B append (Agent B owns these knobs; Agent A wires score/av/TD
# losses, Agent C wires warmstart/uci/fullrun --v20 + tests/test_v20.py —
# all share EXACT names score_mean/score_stdev/root_q/av_logits/td_lambda).
# V19 base + D4 endgame starts/panel + D5 smart resign/playthrough/tail +
# D6 tb_path + D7 literature search defaults (FPU 0.4, c_puct 1.2, ml 0.9)
# + D8 deblunder thr + D9 position-counted LR. R3 fixed resign REPLACED
# (resign_threshold=None, smart_resign=True); R2 game LR REPLACED by
# position LR (lr_drop_positions); R5 ml_thr 0.8->0.9.
V20_CONFIG = Config(blocks=6, channels=64, input_planes=30, se_ratio=4,
                    sims=400, temp_moves=30, opening_random_moves=6,
                    resign_threshold=None, resign_moves=3,
                    adjudicate_margin=3.0, contempt=0.3,
                    no_progress_plies=60, adjudicate_min_ply=16,
                    asymmetric_contempt=True, aux_w=0.01, aux_unit_counts=True,
                    empirical_values=True, values_every=5, eg_frac=0.25,
                    ml_w=0.05, ent_w=0.002, arena_noise=0.0,
                    arena_temp_moves=0, arena_opening_moves=6,
                    sf_depth_suite=False,
                    adaptive_contempt=True, sparring_frac=0.25,
                    pgn_losses=True, tactical_override=True,
                    tac_threshold=0.09, resign_playout_frac=0.1,
                    blunder_veto=True,
                    quiescence_depth=2, forcing_bonus=0.25,
                    contempt_edge_scale=3.0,
                    full_playout_frac=0.15,
                    mate_finish=True,
                    leaf_batch=8, virtual_loss=1.0,
                    safety_veto=True, safety_drop_thr=0.15,
                    fast_frac=0.5, fast_sims=100, reply_w=0.15,
                    pool_frac=0.25,
                    own_w=0.12, margin_w=0.02, mob_w=0.01, safe_w=0.01,
                    c_puct=1.2, fpu_reduction=0.4, prune_singletons=True,
                    ml_slope=0.003, ml_cap=0.07, ml_thr=0.9,
                    smooth_eps=0.01, value_w0=0.5, value_w1=1.0,
                    soft_w=0.3, check_w=0.01,
                    kl_w0=0.2, kl_games=5000, rehearsal_frac=0.2,
                    rehearsal_games="data_sl/games.jsonl",
                    teacher_path="checkpoints_warm/warmstart.pt",
                    opening_book_frac=0.3,
                    book_empirical=True,
                    sf_anchor_games=10, sf_anchor_rung="sf-elo1350",
                    lr_max=3e-4, lr_min=3e-5, lr_drop_games=7000,
                    lr_warmup_steps=1000,
                    sprt_elo0=0.0, sprt_elo1=30.0,
                    sprt_alpha=0.10, sprt_beta=0.10,
                    mirror_gate=True,
                    v20=True, endgame_frac=0.09,
                    endgame_file="data_sl/endgames.jsonl",
                    endgame_panel_pairs=8, endgame_panel_sims=400,
                    endgame_panel_min=0.50,
                    playthrough_frac=0.05, smart_resign=True,
                    resign_w_thr=0.02, resign_ml_thr=0.3,
                    resign_consec=3, resign_every=8,
                    tail_weight=0.25, tail_ply=120,
                    tb_path="data_tb", tb_ml_cap=200,
                    deblunder_thr=0.1, td_lambda=0.5, av_w=0.1,
                    score_w_max=0.05, score_ramp_steps=5000,
                    lr_drop_positions=560000,  # ~7000 games x ~80 plies
                    buffer_size=20000, batch_size=128)


# New runs may opt into the clock-aware representation. Existing checkpoints
# retain their original input count; changing it is an explicit migration.
from dataclasses import replace as _replace
AUDITED_CONFIG = _replace(V20_CONFIG, input_planes=31, prune_singletons=False,
                         augment=True, gate_incumbent_games=64)
