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

_PROMOTION_PIECE_ORDER = {
    chess.QUEEN: 0,
    chess.ROOK: 1,
    chess.BISHOP: 2,
    chess.KNIGHT: 3,
}
_PROMOTION_OPTION_COUNT = len(_PROMOTION_PIECE_ORDER)


class SelfPlayGame:
    """Mutable game-state tracker managing hidden states for Fog of War Chess."""

    def __init__(self, board: chess.Board = None):
        self._board = board if board else chess.Board()

        self._mental_pieces = {True: [None] * 64, False: [None] * 64}
        self._delta_t = {
            True: torch.full((64,), 100.0, dtype=torch.float32),
            False: torch.full((64,), 100.0, dtype=torch.float32),
        }

        self._update_mental_boards()

    def _update_mental_boards(self) -> None:
        current_player = self.turn
        visible_squares = self._compute_visibility(self._board, current_player)

        for sq in chess.SQUARES:
            if sq in visible_squares:
                self._mental_pieces[current_player][sq] = self._board.piece_at(sq)
                self._delta_t[current_player][sq] = 0.0
            else:
                self._delta_t[current_player][sq] += 1.0

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
        cloned = SelfPlayGame(board=self._board.copy())
        cloned._mental_pieces = {
            True: list(self._mental_pieces[True]),
            False: list(self._mental_pieces[False]),
        }
        cloned._delta_t = {
            True: self._delta_t[True].clone(),
            False: self._delta_t[False].clone(),
        }
        return cloned

    def current_player(self):
        return self.turn

    def legal_actions(self):
        return list(self._board.pseudo_legal_moves)

    def apply(self, action):
        self._board.push(action)
        self._update_mental_boards()

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

        ep_sq = board.ep_square
        if ep_sq is not None:
            captured_sq = ep_sq - 8 if player == chess.WHITE else ep_sq + 8
            if 0 <= captured_sq < 64:
                ep_piece = board.piece_at(captured_sq)
                if ep_piece and ep_piece.piece_type == chess.PAWN and ep_piece.color != player:
                    visible.add(captured_sq)
        return visible

    def encode(self, oracle: bool = False):
        current_player = self.turn

        if oracle:
            visible_squares = set(chess.SQUARES)
        else:
            visible_squares = self._compute_visibility(self._board, current_player)

        encoded = torch.zeros((64, 35), dtype=torch.float32)

        max_counts = {
            chess.PAWN: 8.0,
            chess.KNIGHT: 2.0,
            chess.BISHOP: 2.0,
            chess.ROOK: 2.0,
            chess.QUEEN: 1.0,
        }
        current_counts = {
            chess.PAWN: 0,
            chess.KNIGHT: 0,
            chess.BISHOP: 0,
            chess.ROOK: 0,
            chess.QUEEN: 0,
        }
        enemy_color = not current_player

        for piece in self._board.piece_map().values():
            if piece.color == enemy_color and piece.piece_type != chess.KING:
                current_counts[piece.piece_type] += 1

        cpv = [
            (max_counts[pt] - current_counts[pt]) / max_counts[pt]
            for pt in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
        ]

        encoded[:, 30:35] = torch.tensor(cpv, dtype=torch.float32)

        for sq in chess.SQUARES:
            view_sq = sq if current_player == chess.WHITE else chess.square_mirror(sq)

            if sq not in visible_squares:
                encoded[view_sq, 13] = 1.0
            else:
                piece = self._board.piece_at(sq)
                if piece is None:
                    encoded[view_sq, 12] = 1.0
                else:
                    p_idx = piece.piece_type - 1 if piece.color == current_player else piece.piece_type + 5
                    encoded[view_sq, p_idx] = 1.0

            old_piece = self._mental_pieces[current_player][sq]
            if old_piece is None:
                encoded[view_sq, 14 + 12] = 1.0
            else:
                p_idx = old_piece.piece_type - 1 if old_piece.color == current_player else old_piece.piece_type + 5
                encoded[view_sq, 14 + p_idx] = 1.0

            dt = self._delta_t[current_player][sq]
            encoded[view_sq, 27] = 1.0 / (1.0 + dt)

        castle_eligibility = self.castle_eligible()
        encoded[0, 28] = float(castle_eligibility[0])
        encoded[1, 29] = float(castle_eligibility[1])

        return encoded

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
    PY_INIT.call_once(Python::initialize);
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

fn py_legal_actions_uci(game: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    let moves = game.call_method0("legal_actions")?;
    let mut uci_list = Vec::new();
    for mv in moves.try_iter()? {
        let mv = mv?;
        let uci: String = mv.call_method0("uci")?.extract()?;
        uci_list.push(uci);
    }
    uci_list.sort();
    Ok(uci_list)
}

fn rust_legal_actions_uci(game: &fow_rl::SelfPlayGame) -> Vec<String> {
    let mut uci_list: Vec<String> = fow_rl::fuzzing::legal_actions(game)
        .iter()
        .map(|mv| mv.to_string())
        .collect();
    uci_list.sort();
    uci_list
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

    let py_actions = py_legal_actions_uci(py_game)?;
    let rust_actions = rust_legal_actions_uci(rust_game);
    if py_actions != rust_actions {
        return Err(PyErr::new::<pyo3::exceptions::PyAssertionError, _>(
            "legal actions mismatch",
        ));
    }

    Ok(())
}

fuzz_target!(|data: &[u8]| {
    ensure_python();

    Python::attach(|py| {
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
