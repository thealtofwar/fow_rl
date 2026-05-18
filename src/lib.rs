use std::collections::{HashSet, VecDeque};
use std::str::FromStr;

use chess::{
    get_bishop_moves, get_king_moves, get_knight_moves, get_pawn_attacks, get_rook_moves, Board,
    BitBoard, ChessMove, Color, File, MoveGen, Piece, Rank, Square, EMPTY,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

const HISTORY_LEN: usize = 8;
const SLICE_CHANNELS: usize = 13;
const CASTLING_CHANNELS: usize = 2;
const BOARD_SQUARES: usize = 64;
const PROMOTION_BASE: usize = 4096;

const FILES: [File; 8] = [
    File::A,
    File::B,
    File::C,
    File::D,
    File::E,
    File::F,
    File::G,
    File::H,
];

const RANKS: [Rank; 8] = [
    Rank::First,
    Rank::Second,
    Rank::Third,
    Rank::Fourth,
    Rank::Fifth,
    Rank::Sixth,
    Rank::Seventh,
    Rank::Eighth,
];

fn square_from_index(index: usize) -> Square {
    let file = FILES[index % 8];
    let rank = RANKS[index / 8];
    Square::make_square(rank, file)
}

fn square_index(square: Square) -> usize {
    square.to_index() as usize
}

fn mirror_index(index: usize) -> usize {
    index ^ 56
}

fn mirror_square(square: Square) -> Square {
    square_from_index(mirror_index(square_index(square)))
}

fn piece_base_index(piece: Piece) -> usize {
    match piece {
        Piece::Pawn => 0,
        Piece::Knight => 1,
        Piece::Bishop => 2,
        Piece::Rook => 3,
        Piece::Queen => 4,
        Piece::King => 5,
    }
}

fn piece_value(piece: Piece) -> i8 {
    match piece {
        Piece::Pawn => 1,
        Piece::Knight => 2,
        Piece::Bishop => 3,
        Piece::Rook => 4,
        Piece::Queen => 5,
        Piece::King => 6,
    }
}

fn promotion_index(piece: Piece) -> Option<usize> {
    match piece {
        Piece::Queen => Some(0),
        Piece::Rook => Some(1),
        Piece::Bishop => Some(2),
        Piece::Knight => Some(3),
        _ => None,
    }
}

fn reshape_rows(data: Vec<f32>, cols: usize) -> Vec<Vec<f32>> {
    data.chunks(cols).map(|chunk| chunk.to_vec()).collect()
}

fn fen_from_py(board_obj: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(fen) = board_obj.extract::<String>() {
        return Ok(fen);
    }

    if let Ok(fen_attr) = board_obj.getattr("fen") {
        if fen_attr.is_callable() {
            return fen_attr.call0()?.extract::<String>();
        }
        if let Ok(fen) = fen_attr.extract::<String>() {
            return Ok(fen);
        }
    }

    Err(PyValueError::new_err(
        "board must be a python-chess Board or FEN string",
    ))
}

fn flatten_matrix(matrix: Vec<Vec<f32>>) -> PyResult<Vec<f32>> {
    if matrix.len() != BOARD_SQUARES {
        return Err(PyValueError::new_err(format!(
            "history_tensors entries must have {} rows",
            BOARD_SQUARES
        )));
    }

    let mut flat = Vec::with_capacity(BOARD_SQUARES * SLICE_CHANNELS);
    for row in matrix {
        if row.len() != SLICE_CHANNELS {
            return Err(PyValueError::new_err(format!(
                "history_tensors entries must have {} columns",
                SLICE_CHANNELS
            )));
        }
        flat.extend_from_slice(&row);
    }

    Ok(flat)
}

fn validate_flat(flat: Vec<f32>) -> PyResult<Vec<f32>> {
    if flat.len() != BOARD_SQUARES * SLICE_CHANNELS {
        return Err(PyValueError::new_err(format!(
            "history_tensors entries must have length {}",
            BOARD_SQUARES * SLICE_CHANNELS
        )));
    }
    Ok(flat)
}

fn tensor_to_flat(tensor_obj: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    if let Ok(matrix) = tensor_obj.extract::<Vec<Vec<f32>>>() {
        return flatten_matrix(matrix);
    }
    if let Ok(flat) = tensor_obj.extract::<Vec<f32>>() {
        return validate_flat(flat);
    }

    if let Ok(tolist) = tensor_obj.getattr("tolist") {
        if tolist.is_callable() {
            let list_obj = tolist.call0()?;
            if let Ok(matrix) = list_obj.extract::<Vec<Vec<f32>>>() {
                return flatten_matrix(matrix);
            }
            if let Ok(flat) = list_obj.extract::<Vec<f32>>() {
                return validate_flat(flat);
            }
        }
    }

    Err(PyValueError::new_err(
        "history_tensors entries must be 64x13 tensors or lists",
    ))
}

fn history_from_py(history_obj: &Bound<'_, PyAny>) -> PyResult<VecDeque<Vec<f32>>> {
    let mut deque = VecDeque::new();

    for item in history_obj.try_iter()? {
        let item = item?;
        let flat = tensor_to_flat(&item)?;
        deque.push_back(flat);
    }

    while deque.len() > HISTORY_LEN {
        deque.pop_front();
    }

    Ok(deque)
}

fn uci_from_py(action_obj: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(uci) = action_obj.extract::<String>() {
        return Ok(uci);
    }

    if let Ok(uci_attr) = action_obj.getattr("uci") {
        if uci_attr.is_callable() {
            return uci_attr.call0()?.extract::<String>();
        }
        if let Ok(uci) = uci_attr.extract::<String>() {
            return Ok(uci);
        }
    }

    Err(PyValueError::new_err(
        "action must be a python-chess Move or UCI string",
    ))
}

#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct SelfPlayGame {
    board: Board,
    history_tensors: VecDeque<Vec<f32>>,
}

impl SelfPlayGame {
    fn new_internal(board: Board, history_tensors: Option<VecDeque<Vec<f32>>>) -> Self {
        let mut game = SelfPlayGame {
            board,
            history_tensors: history_tensors.unwrap_or_default(),
        };

        if game.history_tensors.is_empty() {
            game.save_history_state();
        }

        game
    }

    fn side_to_move(&self) -> Color {
        self.board.side_to_move()
    }

    fn push_history(&mut self, slice: Vec<f32>) {
        if self.history_tensors.len() == HISTORY_LEN {
            self.history_tensors.pop_front();
        }
        self.history_tensors.push_back(slice);
    }

    fn save_history_state(&mut self) {
        let current_player = self.side_to_move();
        let visible_squares = self.compute_visibility(current_player);

        let mut slice_tensor = vec![0.0; BOARD_SQUARES * SLICE_CHANNELS];

        for index in 0..BOARD_SQUARES {
            let view_index = if current_player == Color::White {
                index
            } else {
                mirror_index(index)
            };

            if !visible_squares.contains(&index) {
                slice_tensor[view_index * SLICE_CHANNELS + 12] = 1.0;
                continue;
            }

            let square = square_from_index(index);
            if let Some(piece) = self.board.piece_on(square) {
                let is_own_piece = self.board.color_on(square) == Some(current_player);
                let mut piece_idx = piece_base_index(piece);
                if !is_own_piece {
                    piece_idx += 6;
                }
                slice_tensor[view_index * SLICE_CHANNELS + piece_idx] = 1.0;
            }
        }

        self.push_history(slice_tensor);
    }

    fn board_matrix(&self) -> Vec<Vec<i8>> {
        let mut mat = Vec::with_capacity(8);
        for rank in (0..8).rev() {
            let mut row = Vec::with_capacity(8);
            for file in 0..8 {
                let square = Square::make_square(RANKS[rank], FILES[file]);
                if let Some(piece) = self.board.piece_on(square) {
                    let mut val = piece_value(piece);
                    if self.board.color_on(square) == Some(Color::Black) {
                        val = -val;
                    }
                    row.push(val);
                } else {
                    row.push(0);
                }
            }
            mat.push(row);
        }
        mat
    }

    fn castle_eligible_internal(&self) -> (bool, bool) {
        let rights = self.board.castle_rights(self.side_to_move());
        (rights.has_kingside(), rights.has_queenside())
    }

    fn legal_actions_internal(&self) -> Vec<ChessMove> {
        MoveGen::new_legal(&self.board).collect()
    }

    fn apply_internal(&mut self, action: ChessMove) {
        self.board = self.board.make_move_new(action);
        self.save_history_state();
    }

    fn is_terminal_internal(&self) -> bool {
        !self.king_present(Color::White) || !self.king_present(Color::Black)
    }

    fn terminal_value_internal(&self, player: Color) -> f32 {
        if !self.king_present(Color::White) {
            return if player == Color::Black { 1.0 } else { -1.0 };
        }
        if !self.king_present(Color::Black) {
            return if player == Color::White { 1.0 } else { -1.0 };
        }
        0.0
    }

    fn king_present(&self, color: Color) -> bool {
        let kings = *self.board.pieces(Piece::King) & *self.board.color_combined(color);
        kings != EMPTY
    }

    fn piece_attacks(&self, square: Square, piece: Piece, color: Color) -> BitBoard {
        let blockers = *self.board.combined();
        match piece {
            Piece::Pawn => get_pawn_attacks(square, color, !EMPTY),
            Piece::Knight => get_knight_moves(square),
            Piece::Bishop => get_bishop_moves(square, blockers),
            Piece::Rook => get_rook_moves(square, blockers),
            Piece::Queen => get_bishop_moves(square, blockers) | get_rook_moves(square, blockers),
            Piece::King => get_king_moves(square),
        }
    }

    fn compute_visibility(&self, player: Color) -> HashSet<usize> {
        let mut visible = HashSet::new();

        for index in 0..BOARD_SQUARES {
            let square = square_from_index(index);
            let piece = self.board.piece_on(square);
            let color = self.board.color_on(square);

            if piece.is_some() && color == Some(player) {
                visible.insert(index);
                let attacks = self.piece_attacks(square, piece.unwrap(), player);
                for target in attacks {
                    visible.insert(square_index(target));
                }

                if piece == Some(Piece::Pawn) {
                    let forward_offset: i32 = if player == Color::White { 8 } else { -8 };
                    let fwd_index = index as i32 + forward_offset;
                    if (0..BOARD_SQUARES as i32).contains(&fwd_index) {
                        visible.insert(fwd_index as usize);

                        let rank = index / 8;
                        let start_rank = if player == Color::White { 1 } else { 6 };
                        if rank == start_rank {
                            let fwd_square = square_from_index(fwd_index as usize);
                            if self.board.piece_on(fwd_square).is_none() {
                                let dbl_index = index as i32 + 2 * forward_offset;
                                if (0..BOARD_SQUARES as i32).contains(&dbl_index) {
                                    visible.insert(dbl_index as usize);
                                }
                            }
                        }
                    }
                }
            }
        }

        visible
    }

    fn encode_flat(&self) -> Vec<f32> {
        let mut history_list: Vec<Vec<f32>> = if self.history_tensors.is_empty() {
            vec![vec![0.0; BOARD_SQUARES * SLICE_CHANNELS]]
        } else {
            self.history_tensors.iter().cloned().collect()
        };

        while history_list.len() < HISTORY_LEN {
            let first = history_list[0].clone();
            history_list.insert(0, first);
        }

        let history_channels = SLICE_CHANNELS * HISTORY_LEN;
        let mut tensor = vec![0.0; BOARD_SQUARES * history_channels];

        for square in 0..BOARD_SQUARES {
            for (idx, slice) in history_list.iter().enumerate() {
                let dst_base = square * history_channels + idx * SLICE_CHANNELS;
                let src_base = square * SLICE_CHANNELS;
                tensor[dst_base..dst_base + SLICE_CHANNELS]
                    .copy_from_slice(&slice[src_base..src_base + SLICE_CHANNELS]);
            }
        }

        let mut castling_channel = vec![0.0; BOARD_SQUARES * CASTLING_CHANNELS];
        let (king_side, queen_side) = self.castle_eligible_internal();
        let king_val = if king_side { 1.0 } else { 0.0 };
        let queen_val = if queen_side { 1.0 } else { 0.0 };

        if BOARD_SQUARES >= 2 {
            castling_channel[0] += king_val;
            castling_channel[1] += king_val;
            castling_channel[2] += queen_val;
            castling_channel[3] += queen_val;
        }

        let mut encoded = vec![0.0; BOARD_SQUARES * (history_channels + CASTLING_CHANNELS)];
        for square in 0..BOARD_SQUARES {
            let dst_base = square * (history_channels + CASTLING_CHANNELS);
            let src_base = square * history_channels;
            encoded[dst_base..dst_base + history_channels]
                .copy_from_slice(&tensor[src_base..src_base + history_channels]);

            let castling_base = square * CASTLING_CHANNELS;
            encoded[dst_base + history_channels..dst_base + history_channels + CASTLING_CHANNELS]
                .copy_from_slice(&castling_channel[castling_base..castling_base + CASTLING_CHANNELS]);
        }

        encoded
    }

    fn action_index_internal(&self, action: ChessMove) -> Result<usize, String> {
        let mut from_sq = action.get_source();
        let mut to_sq = action.get_dest();

        if self.side_to_move() == Color::Black {
            from_sq = mirror_square(from_sq);
            to_sq = mirror_square(to_sq);
        }

        if let Some(promotion) = action.get_promotion() {
            let promotion_idx = promotion_index(promotion)
                .ok_or_else(|| format!("Unsupported promotion piece: {:?}", promotion))?;
            return Ok(PROMOTION_BASE + square_index(from_sq) * 4 + promotion_idx);
        }

        Ok(square_index(from_sq) * BOARD_SQUARES + square_index(to_sq))
    }
}

#[pymethods]
impl SelfPlayGame {
    #[new]
    #[pyo3(signature = (board=None, history_tensors=None))]
    fn new(
        py: Python,
        board: Option<Py<PyAny>>,
        history_tensors: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let board = match board {
            Some(obj) => {
                let bound = obj.bind(py);
                let fen = fen_from_py(&bound)?;
                Board::from_str(&fen).map_err(|_| PyValueError::new_err("Invalid FEN string"))?
            }
            None => Board::default(),
        };

        let history_deque = match history_tensors {
            Some(obj) => {
                let bound = obj.bind(py);
                Some(history_from_py(&bound)?)
            }
            None => None,
        };

        Ok(SelfPlayGame::new_internal(board, history_deque))
    }

    #[getter]
    fn turn(&self) -> bool {
        self.side_to_move() == Color::White
    }

    #[getter]
    fn board(&self) -> Vec<Vec<i8>> {
        self.board_matrix()
    }

    fn castle_eligible(&self) -> (bool, bool) {
        self.castle_eligible_internal()
    }

    fn clone(&self) -> SelfPlayGame {
        Clone::clone(self)
    }

    fn current_player(&self) -> bool {
        self.side_to_move() == Color::White
    }

    #[pyo3(name = "legal_actions")]
    fn legal_actions_py(&self, py: Python) -> PyResult<Vec<Py<PyAny>>> {
        let chess_mod = PyModule::import(py, "chess")
            .map_err(|_| PyValueError::new_err("python-chess is required for legal_actions()"))?;
        let move_type = chess_mod.getattr("Move")?;
        let from_uci = move_type.getattr("from_uci")?;

        let mut moves = Vec::new();
        for mv in self.legal_actions_internal() {
            let obj = from_uci.call1((mv.to_string(),))?;
            moves.push(obj.unbind());
        }

        Ok(moves)
    }

    fn apply(&mut self, py: Python, action: Py<PyAny>) -> PyResult<()> {
        let bound = action.bind(py);
        let uci = uci_from_py(&bound)?;
        let action = ChessMove::from_str(&uci)
            .map_err(|_| PyValueError::new_err("Invalid UCI move"))?;
        self.apply_internal(action);
        Ok(())
    }

    fn is_terminal(&self) -> bool {
        self.is_terminal_internal()
    }

    fn terminal_value(&self, player_is_white: bool) -> f32 {
        let player = if player_is_white { Color::White } else { Color::Black };
        self.terminal_value_internal(player)
    }

    fn encode(&self, py: Python) -> PyResult<Py<PyAny>> {
        let flat = self.encode_flat();
        let cols = SLICE_CHANNELS * HISTORY_LEN + CASTLING_CHANNELS;
        let rows = reshape_rows(flat, cols);

        let torch = PyModule::import(py, "torch")
            .map_err(|_| PyValueError::new_err("torch is required for encode()"))?;
        let tensor_fn = torch.getattr("tensor")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("dtype", torch.getattr("float32")?)?;
        let tensor = tensor_fn.call((rows,), Some(&kwargs))?;
        Ok(tensor.unbind())
    }

    fn action_index(&self, py: Python, action: Py<PyAny>) -> PyResult<usize> {
        let bound = action.bind(py);
        let uci = uci_from_py(&bound)?;
        let action = ChessMove::from_str(&uci)
            .map_err(|_| PyValueError::new_err("Invalid UCI move"))?;
        self.action_index_internal(action)
            .map_err(PyValueError::new_err)
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn fow_rl(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SelfPlayGame>()?;
    Ok(())
}

#[cfg(feature = "fuzzing")]
pub mod fuzzing {
    use super::*;

    pub fn new_default_game() -> SelfPlayGame {
        SelfPlayGame::new_internal(Board::default(), None)
    }

    pub fn apply(game: &mut SelfPlayGame, action: ChessMove) {
        game.apply_internal(action);
    }

    pub fn legal_actions(game: &SelfPlayGame) -> Vec<ChessMove> {
        game.legal_actions_internal()
    }

    pub fn board_matrix(game: &SelfPlayGame) -> Vec<Vec<i8>> {
        game.board_matrix()
    }

    pub fn castle_eligible(game: &SelfPlayGame) -> (bool, bool) {
        game.castle_eligible_internal()
    }

    pub fn encode_flat(game: &SelfPlayGame) -> Vec<f32> {
        game.encode_flat()
    }

    pub fn action_index(game: &SelfPlayGame, action: ChessMove) -> Result<usize, String> {
        game.action_index_internal(action)
    }

    pub fn is_white_to_move(game: &SelfPlayGame) -> bool {
        game.side_to_move() == Color::White
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::PyModule;
    use std::ffi::CString;
    use std::sync::Once;

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

    static PY_INIT: Once = Once::new();

    fn with_python<F, R>(f: F) -> R
    where
        F: for<'py> FnOnce(Python<'py>) -> R,
    {
        PY_INIT.call_once(Python::initialize);
        Python::attach(f)
    }

    fn python_selfplay_module<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyModule>> {
        let code = CString::new(PY_SELFPLAY_CODE).expect("Python code contains no NUL bytes");
        let filename = CString::new("selfplay_ref.py").expect("valid filename");
        let module = CString::new("selfplay_ref").expect("valid module name");
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

    #[test]
    fn regression_default_state() -> PyResult<()> {
        with_python(|py| {
            let chess_mod = match PyModule::import(py, "chess") {
                Ok(module) => module,
                Err(_) => return Ok(()),
            };
            if PyModule::import(py, "torch").is_err() {
                return Ok(());
            }

            let module = python_selfplay_module(py)?;
            let cls = module.getattr("SelfPlayGame")?;
            let py_game = cls.call0()?;
            let rust_game = SelfPlayGame::new_internal(Board::default(), None);

            let py_turn: bool = py_game.getattr("turn")?.extract()?;
            assert_eq!(rust_game.side_to_move() == Color::White, py_turn);

            let py_board: Vec<Vec<i8>> = py_game.getattr("board")?.extract()?;
            assert_eq!(rust_game.board_matrix(), py_board);

            let py_castle: (bool, bool) = py_game.call_method0("castle_eligible")?.extract()?;
            assert_eq!(rust_game.castle_eligible_internal(), py_castle);

            let py_encoded = py_encode_flat(&py_game)?;
            assert_eq!(rust_game.encode_flat(), py_encoded);

            let mv = chess_mod
                .getattr("Move")?
                .getattr("from_uci")?
                .call1(("e2e4",))?;
            let py_idx: usize = py_game.call_method1("action_index", (mv,))?.extract()?;
            let rust_idx = rust_game
                .action_index_internal(ChessMove::from_str("e2e4").unwrap())
                .unwrap();
            assert_eq!(rust_idx, py_idx);

            Ok(())
        })
    }

    #[test]
    fn regression_after_moves() -> PyResult<()> {
        with_python(|py| {
            let chess_mod = match PyModule::import(py, "chess") {
                Ok(module) => module,
                Err(_) => return Ok(()),
            };
            if PyModule::import(py, "torch").is_err() {
                return Ok(());
            }

            let module = python_selfplay_module(py)?;
            let cls = module.getattr("SelfPlayGame")?;
            let py_game = cls.call0()?;
            let mut rust_game = SelfPlayGame::new_internal(Board::default(), None);

            for uci in ["e2e4", "e7e5", "g1f3"] {
                let mv = chess_mod
                    .getattr("Move")?
                    .getattr("from_uci")?
                    .call1((uci,))?;
                py_game.call_method1("apply", (mv,))?;

                let rust_mv = ChessMove::from_str(uci).unwrap();
                rust_game.apply_internal(rust_mv);
            }

            let py_turn: bool = py_game.getattr("turn")?.extract()?;
            assert_eq!(rust_game.side_to_move() == Color::White, py_turn);

            let py_board: Vec<Vec<i8>> = py_game.getattr("board")?.extract()?;
            assert_eq!(rust_game.board_matrix(), py_board);

            let py_castle: (bool, bool) = py_game.call_method0("castle_eligible")?.extract()?;
            assert_eq!(rust_game.castle_eligible_internal(), py_castle);

            let py_encoded = py_encode_flat(&py_game)?;
            assert_eq!(rust_game.encode_flat(), py_encoded);

            Ok(())
        })
    }

    #[test]
    fn regression_promotion_action_index() -> PyResult<()> {
        with_python(|py| {
            let chess_mod = match PyModule::import(py, "chess") {
                Ok(module) => module,
                Err(_) => return Ok(()),
            };
            if PyModule::import(py, "torch").is_err() {
                return Ok(());
            }

            let module = python_selfplay_module(py)?;
            let cls = module.getattr("SelfPlayGame")?;

            let fen = "8/P7/8/8/8/8/8/K6k w - - 0 1";
            let py_board = chess_mod.getattr("Board")?.call1((fen,))?;
            let py_game = cls.call1((py_board,))?;

            let rust_board = Board::from_str(fen).unwrap();
            let rust_game = SelfPlayGame::new_internal(rust_board, None);

            let mv = chess_mod
                .getattr("Move")?
                .getattr("from_uci")?
                .call1(("a7a8q",))?;
            let py_idx: usize = py_game.call_method1("action_index", (mv,))?.extract()?;

            let rust_idx = rust_game
                .action_index_internal(ChessMove::from_str("a7a8q").unwrap())
                .unwrap();
            assert_eq!(rust_idx, py_idx);

            Ok(())
        })
    }
}
