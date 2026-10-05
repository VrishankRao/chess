//! chesscore: thin PyO3 glue over the proven `chess` crate's bitboard
//! movegen. We NEVER implement move rules by hand — the `chess` crate is
//! perft-verified; this file only translates between its types and the
//! Python hot path (square ints 0-63 A1=0, matching python-chess exactly;
//! promotion ints 0=none, 2=N, 3=B, 4=R, 5=Q, matching python-chess).
//!
//! Scope: movegen + make/unmake + check queries (the profiled cluster).
//! Terminals, adjudication, clocks, planes, TT keys all stay Python.

use chess::{Board, ChessMove, Color, MoveGen, Piece, Square};
use pyo3::prelude::*;
use std::str::FromStr;

fn sq(i: u8) -> Square {
    // SAFETY: all call sites mask indices to 0..64.
    unsafe { Square::new(i) }
}

fn promo_of(p: u8) -> Option<Piece> {
    match p {
        2 => Some(Piece::Knight),
        3 => Some(Piece::Bishop),
        4 => Some(Piece::Rook),
        5 => Some(Piece::Queen),
        _ => None,
    }
}

fn promo_to(p: Option<Piece>) -> u8 {
    match p {
        Some(Piece::Knight) => 2,
        Some(Piece::Bishop) => 3,
        Some(Piece::Rook) => 4,
        Some(Piece::Queen) => 5,
        _ => 0,
    }
}

#[pyclass]
struct Position {
    // Full history stack: push clones (~200B memcpy), pop restores.
    // No FEN round-trips on the hot path (FEN only at construction).
    stack: Vec<Board>,
}

#[pymethods]
impl Position {
    #[new]
    fn new(fen: &str) -> PyResult<Self> {
        Board::from_str(fen)
            .map(|b| Position { stack: vec![b] })
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(
                format!("bad FEN: {e}"),
            ))
    }

    fn fen(&self) -> String {
        self.stack.last().map(|b| b.to_string()).unwrap_or_default()
    }

    fn turn_is_white(&self) -> bool {
        self.stack.last().map(|b| b.side_to_move() == Color::White)
            .unwrap_or(true)
    }

    /// All legal moves as (from, to, promo) int triples.
    fn legal_moves(&self) -> Vec<(u8, u8, u8)> {
        match self.stack.last() {
            Some(b) => MoveGen::new_legal(b)
                .map(|m| {
                    (m.get_source().to_index() as u8,
                     m.get_dest().to_index() as u8,
                     promo_to(m.get_promotion()))
                })
                .collect(),
            None => Vec::new(),
        }
    }

    /// Push a move; returns False (no state change) if illegal.
    fn push(&mut self, from: u8, to: u8, promo: u8) -> bool {
        let (from, to) = (from & 63, to & 63);
        let mv = ChessMove::new(sq(from), sq(to), promo_of(promo));
        match self.stack.last() {
            Some(b) => {
                // Verify legality against the generator (cheap: movegen
                // already ran for legal_moves; this guards stale callers).
                let ok = MoveGen::new_legal(b).any(|m| m == mv);
                if !ok {
                    return false;
                }
                let mut nb = *b;
                b.make_move(mv, &mut nb);
                self.stack.push(nb);
                true
            }
            None => false,
        }
    }

    /// Clone + push without a legality scan (caller pushes legal moves
    /// only). None on empty stack.
    fn pushed(&self, from: u8, to: u8, promo: u8) -> Option<Position> {
        match self.stack.last() {
            Some(b) => {
                let mv = ChessMove::new(sq(from & 63), sq(to & 63),
                                        promo_of(promo));
                let mut nb = *b;
                b.make_move(mv, &mut nb);
                let mut stack = self.stack.clone();
                stack.push(nb);
                Some(Position { stack })
            }
            None => None,
        }
    }

    fn pop(&mut self) -> bool {
        if self.stack.len() > 1 {
            self.stack.pop();
            true
        } else {
            false
        }
    }

    fn in_check(&self) -> bool {
        self.stack.last().map(|b| b.checkers().popcnt() > 0).unwrap_or(false)
    }

    fn is_capture(&self, from: u8, to: u8) -> bool {
        match self.stack.last() {
            Some(b) => {
                let (f, t) = (sq(from & 63), sq(to & 63));
                // Occupied destination counts only vs the OPPONENT
                // (python-chess semantics; own-piece destinations are
                // illegal and never queried on legal moves).
                if b.piece_on(t).is_some() {
                    return b.color_on(t) != b.color_on(f);
                }
                // En passant: destination empty, mover is a pawn of the
                // side to move. NOTE: the `chess` crate normalizes the
                // recorded EP square to the CAPTURABLE PAWN's square
                // (not the FEN target square), so compare against one
                // rank forward of it. Movegen itself is unaffected
                // (perft-exact incl. EP lines).
                if b.piece_on(f) != Some(Piece::Pawn)
                    || b.color_on(f) != Some(b.side_to_move())
                {
                    return false;
                }
                match b.en_passant() {
                    Some(ep) => {
                        let ei = ep.to_index() as i16;
                        let ti = t.to_index() as i16;
                        ti == if b.side_to_move() == Color::White {
                            ei + 8
                        } else {
                            ei - 8
                        }
                    }
                    None => false,
                }
            }
            None => false,
        }
    }

    fn gives_check(&self, from: u8, to: u8, promo: u8) -> bool {
        match self.stack.last() {
            Some(b) => {
                let mv = ChessMove::new(sq(from & 63), sq(to & 63),
                                        promo_of(promo));
                if !MoveGen::new_legal(b).any(|m| m == mv) {
                    return false;
                }
                let mut nb = *b;
                b.make_move(mv, &mut nb);
                nb.checkers().popcnt() > 0
            }
            None => false,
        }
    }

    fn piece_at(&self, sqi: u8) -> u8 {
        // 0 empty, else python-chess piece_type ints (P1 N2 B3 R4 Q5 K6).
        match self.stack.last().and_then(|b| b.piece_on(sq(sqi & 63))) {
            Some(Piece::Pawn) => 1,
            Some(Piece::Knight) => 2,
            Some(Piece::Bishop) => 3,
            Some(Piece::Rook) => 4,
            Some(Piece::Queen) => 5,
            Some(Piece::King) => 6,
            None => 0,
        }
    }
}

/// Perft counter for validation (startpos depth4 = 197281,
/// kiwipete depth3 = 97862). Not on the hot path. Depth capped: perft is
/// exponential, and unbounded recursion is a stack-overflow DoS vector.
#[pyfunction]
fn perft(fen: &str, depth: u8) -> PyResult<u64> {
    if depth > 6 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "perft depth capped at 6",
        ));
    }
    let b = Board::from_str(fen).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("bad FEN: {e}"))
    })?;
    Ok(perft_rec(&b, depth))
}

fn perft_rec(b: &Board, depth: u8) -> u64 {
    if depth == 0 {
        return 1;
    }
    let mut n = 0u64;
    for m in MoveGen::new_legal(b) {
        let mut nb = *b;
        b.make_move(m, &mut nb);
        n += perft_rec(&nb, depth - 1);
    }
    n
}

#[pymodule]
fn chesscore(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Position>()?;
    m.add_function(wrap_pyfunction!(perft, m)?)?;
    // Stale-wheel visibility: Python logs this once per process.
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
