#![no_main]

use libfuzzer_sys::fuzz_target;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use std::ffi::CString;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Once;

static PY_INIT: Once = Once::new();
static MODULE_COUNTER: AtomicUsize = AtomicUsize::new(0);

const PY_SELFPLAY_CODE: &str = r#"
import chess
import torch
from collections import deque

_PROMOTION_PIECE_ORDER = {
    chess.QUEEN: 0,
    chess.ROOK: 1,
    chess.BISHOP: 2,
    chess.KNIGHT: 3,
}
_PROMOTION_OPTION_COUNT = len(_PROMOTION_PIECE_ORDER)


class SelfPlayGame:
    """Mutable game-state protocol consumed by PPO for Fog of War Chess."""

    def __init__(self, board: chess.Board = None, history_tensors: deque = None):
        self._board = board if board else chess.Board()

        if history_tensors is not None:
            self._history_tensors = history_tensors
        else:
            self._history_tensors = deque(maxlen=8)
            self._save_history_state()

    def _save_history_state(self) -> None:
        current_player = self.turn
        visible_squares = self._compute_visibility(self._board, current_player)

        slice_tensor = torch.zeros((64, 13), dtype=torch.float32)

        for sq in chess.SQUARES:
            view_sq = sq if current_player == chess.WHITE else chess.square_mirror(sq)

            if sq not in visible_squares:
                slice_tensor[view_sq, 12] = 1.0
            else:
                piece = self._board.piece_at(sq)
                if piece is not None:
                    is_own_piece = (piece.color == current_player)
                    piece_idx = piece.piece_type - 1
                    if not is_own_piece:
                        piece_idx += 6
                    slice_tensor[view_sq, piece_idx] = 1.0

        self._history_tensors.append(slice_tensor)

    @property
    def turn(self) -> bool:
        return self._board.turn

    @property
    def board(self):
        mat = []
        for rank in range(7, -1, -1):
            row = []
            for file in range(8):
                piece = self._board.piece_at(chess.square(file, rank))
                if not piece:
                    row.append(0)
                else:
                    val = piece.piece_type
                    row.append(val if piece.color == chess.WHITE else -val)
            mat.append(row)
        return mat

    def castle_eligible(self):
        return (
            self._board.has_kingside_castling_rights(self.turn),
            self._board.has_queenside_castling_rights(self.turn),
        )

    def clone(self):
        cloned_history = deque([t.clone() for t in self._history_tensors], maxlen=8)
        return SelfPlayGame(board=self._board.copy(), history_tensors=cloned_history)

    def current_player(self):
        return self.turn

    def legal_actions(self):
        return list(self._board.pseudo_legal_moves)

    def apply(self, action):
        self._board.push(action)
        self._save_history_state()

    def is_terminal(self) -> bool:
        return (self._board.king(chess.WHITE) is None) or (
            self._board.king(chess.BLACK) is None
        )

    def terminal_value(self, player):
        white_king = self._board.king(chess.WHITE)
        black_king = self._board.king(chess.BLACK)
        if white_king is None:
            return 1.0 if player == chess.BLACK else -1.0
        if black_king is None:
            return 1.0 if player == chess.WHITE else -1.0
        return 0.0

    def _compute_visibility(self, board: chess.Board, player: bool):
        visible = set()
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece and piece.color == player:
                visible.add(sq)
                visible.update(board.attacks(sq))

                if piece.piece_type == chess.PAWN:
                    forward_offset = 8 if player == chess.WHITE else -8
                    fwd_sq = sq + forward_offset
                    if 0 <= fwd_sq < 64:
                        visible.add(fwd_sq)
                        rank = chess.square_rank(sq)
                        if (player == chess.WHITE and rank == 1) or (
                            player == chess.BLACK and rank == 6
                        ):
                            if board.piece_at(fwd_sq) is None:
                                visible.add(sq + 2 * forward_offset)
        return visible

    def encode(self):
        history_list = list(self._history_tensors)
        while len(history_list) < 8:
            history_list.insert(0, history_list[0])

        tensor = torch.cat(history_list, dim=1)

        castle_eligibility = self.castle_eligible()

        castling_channel = torch.zeros((64, 2), dtype=torch.float32)
        castling_channel[0] += castle_eligibility[0]
        castling_channel[1] += castle_eligibility[1]

        return torch.cat([tensor, castling_channel], dim=1)

    def action_index(self, action):
        from_sq = action.from_square
        to_sq = action.to_square

        if not self.turn:
            from_sq = chess.square_mirror(from_sq)
            to_sq = chess.square_mirror(to_sq)

        if action.promotion is not None:
            promotion_idx = _PROMOTION_PIECE_ORDER.get(action.promotion)
            if promotion_idx is None:
                raise ValueError(f"Unsupported promotion piece: {action.promotion}")
            return 4096 + from_sq * 4 + promotion_idx

        return from_sq * 64 + to_sq
"#;

fn ensure_python() {
    PY_INIT.call_once(pyo3::prepare_freethreaded_python);
}

fn python_selfplay_module<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyModule>> {
    let code = CString::new(PY_SELFPLAY_CODE).expect("Python code contains no NUL bytes");
    let filename = CString::new("selfplay_fuzz.py").expect("valid filename");
    let module_name = format!(
        "selfplay_fuzz_{}",
        MODULE_COUNTER.fetch_add(1, Ordering::Relaxed)
    );
    let module = CString::new(module_name).expect("valid module name");
    PyModule::from_code(py, code.as_c_str(), filename.as_c_str(), module.as_c_str())
}

fn flatten_2d(matrix: Vec<Vec<f32>>) -> Vec<f32> {
    let mut flat = Vec::new();
    for row in matrix {
        flat.extend_from_slice(&row);
    }
    flat
}

fn py_encode_flat(game: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    let tensor = game.call_method0("encode")?;
    let list = tensor.call_method0("tolist")?;
    let rows: Vec<Vec<f32>> = list.extract()?;
    Ok(flatten_2d(rows))
}

fn compare_state(py_game: &Bound<'_, PyAny>, rust_game: &fow_rl::SelfPlayGame) -> PyResult<()> {
    let py_turn: bool = py_game.getattr("turn")?.extract()?;
    let rust_turn = fow_rl::fuzzing::is_white_to_move(rust_game);
    if rust_turn != py_turn {
        return Err(PyErr::new::<pyo3::exceptions::PyAssertionError, _>(
            "turn mismatch",
        ));
    }

    let py_board: Vec<Vec<i8>> = py_game.getattr("board")?.extract()?;
    if fow_rl::fuzzing::board_matrix(rust_game) != py_board {
        return Err(PyErr::new::<pyo3::exceptions::PyAssertionError, _>(
            "board mismatch",
        ));
    }

    let py_castle: (bool, bool) = py_game.call_method0("castle_eligible")?.extract()?;
    if fow_rl::fuzzing::castle_eligible(rust_game) != py_castle {
        return Err(PyErr::new::<pyo3::exceptions::PyAssertionError, _>(
            "castling mismatch",
        ));
    }

    let py_encoded = py_encode_flat(py_game)?;
    if fow_rl::fuzzing::encode_flat(rust_game) != py_encoded {
        return Err(PyErr::new::<pyo3::exceptions::PyAssertionError, _>(
            "encode mismatch",
        ));
    }

    Ok(())
}

fuzz_target!(|data: &[u8]| {
    ensure_python();

    Python::with_gil(|py| {
        let chess_mod = match PyModule::import(py, "chess") {
            Ok(module) => module,
            Err(_) => return,
        };
        if PyModule::import(py, "torch").is_err() {
            return;
        }

        let module = match python_selfplay_module(py) {
            Ok(module) => module,
            Err(_) => return,
        };
        let cls = match module.getattr("SelfPlayGame") {
            Ok(cls) => cls,
            Err(_) => return,
        };
        let py_game = match cls.call0() {
            Ok(game) => game,
            Err(_) => return,
        };

        let move_type = match chess_mod.getattr("Move") {
            Ok(ty) => ty,
            Err(_) => return,
        };
        let from_uci = match move_type.getattr("from_uci") {
            Ok(func) => func,
            Err(_) => return,
        };

        let mut rust_game = fow_rl::fuzzing::new_default_game();

        if compare_state(&py_game, &rust_game).is_err() {
            panic!("state mismatch at start");
        }

        let max_steps = data.len().min(64);
        for (step, byte) in data.iter().take(max_steps).enumerate() {
            let rust_moves = fow_rl::fuzzing::legal_actions(&rust_game);
            if rust_moves.is_empty() {
                break;
            }

            let choice = (*byte as usize) % rust_moves.len();
            let mv = rust_moves[choice];
            let uci = mv.to_string();

            let py_move = match from_uci.call1((uci.as_str(),)) {
                Ok(mv) => mv.unbind(),
                Err(_) => break,
            };

            let py_idx: usize = match py_game.call_method1(
                "action_index",
                (py_move.clone_ref(py),),
            ) {
                Ok(value) => match value.extract() {
                    Ok(idx) => idx,
                    Err(_) => break,
                },
                Err(_) => break,
            };
            let rust_idx = match fow_rl::fuzzing::action_index(&rust_game, mv) {
                Ok(idx) => idx,
                Err(_) => break,
            };
            if rust_idx != py_idx {
                panic!("action_index mismatch at step {step}");
            }

            if py_game.call_method1("apply", (py_move,)).is_err() {
                break;
            }
            fow_rl::fuzzing::apply(&mut rust_game, mv);

            if compare_state(&py_game, &rust_game).is_err() {
                panic!("state mismatch at step {step}");
            }
        }
    });
});
