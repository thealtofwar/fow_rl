use std::str::FromStr;

use chess::{
    get_bishop_moves, get_king_moves, get_knight_moves, get_pawn_attacks, get_pawn_quiets,
    get_rook_moves, Board, BitBoard, ChessMove, Color, File, Piece, Rank, Square, EMPTY,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

const BOARD_SQUARES: usize = 64;
const ENCODE_CHANNELS: usize = 35;
const OBS_EMPTY_CHANNEL: usize = 12;
const OBS_FOG_CHANNEL: usize = 13;
const MEMORY_OFFSET: usize = 14;
const MEMORY_EMPTY_CHANNEL: usize = 12;
const RECENCY_CHANNEL: usize = 27;
const CASTLING_KS_CHANNEL: usize = 28;
const CASTLING_QS_CHANNEL: usize = 29;
const CPV_OFFSET: usize = 30;
const CPV_LEN: usize = 5;
const INITIAL_DELTA_T: f32 = 100.0;
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

fn is_promotion_square(square: Square, color: Color) -> bool {
    match color {
        Color::White => square.get_rank() == Rank::Eighth,
        Color::Black => square.get_rank() == Rank::First,
    }
}

fn push_promotion_moves(moves: &mut Vec<ChessMove>, source: Square, dest: Square) {
    moves.push(ChessMove::new(source, dest, Some(Piece::Queen)));
    moves.push(ChessMove::new(source, dest, Some(Piece::Rook)));
    moves.push(ChessMove::new(source, dest, Some(Piece::Bishop)));
    moves.push(ChessMove::new(source, dest, Some(Piece::Knight)));
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct SeenPiece {
    piece: Piece,
    color: Color,
}

fn color_index(color: Color) -> usize {
    match color {
        Color::White => 0,
        Color::Black => 1,
    }
}

fn opposite_color(color: Color) -> Color {
    match color {
        Color::White => Color::Black,
        Color::Black => Color::White,
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
    mental_pieces: [[Option<SeenPiece>; BOARD_SQUARES]; 2],
    delta_t: [[f32; BOARD_SQUARES]; 2],
    captured_king: Option<Color>,
}

impl SelfPlayGame {
    fn new_internal(board: Board) -> Self {
        let mut game = SelfPlayGame {
            board,
            mental_pieces: [[None; BOARD_SQUARES]; 2],
            delta_t: [[INITIAL_DELTA_T; BOARD_SQUARES]; 2],
            captured_king: None,
        };

        game.update_mental_boards();
        game
    }

    fn side_to_move(&self) -> Color {
        self.board.side_to_move()
    }

    fn update_mental_boards(&mut self) {
        let current_player = self.side_to_move();
        let player_idx = color_index(current_player);
        let visible_squares = self.compute_visibility_bb(current_player);

        for index in 0..BOARD_SQUARES {
            let square = square_from_index(index);
            let square_bb = BitBoard::from_square(square);
            let is_visible = (visible_squares & square_bb) != EMPTY;
            if is_visible {
                let piece = self.board.piece_on(square);
                let color = self.board.color_on(square);
                self.mental_pieces[player_idx][index] = match (piece, color) {
                    (Some(piece), Some(color)) => Some(SeenPiece { piece, color }),
                    _ => None,
                };
                self.delta_t[player_idx][index] = 0.0;
            } else {
                self.delta_t[player_idx][index] += 1.0;
            }
        }
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
        let color = self.side_to_move();
        let own = *self.board.color_combined(color);
        let enemy = *self.board.color_combined(opposite_color(color));
        let occupied = *self.board.combined();
        let mut moves = Vec::new();

        let pawns = *self.board.pieces(Piece::Pawn) & own;
        let en_passant = self.board.en_passant();
        let en_passant_target = en_passant.map(|ep_sq| {
            let forward_offset = if color == Color::White { 8 } else { -8 };
            let target_index = square_index(ep_sq) as i32 + forward_offset;
            square_from_index(target_index as usize)
        });
        
        for source in pawns {
            let quiets = get_pawn_quiets(source, color, occupied);
            for dest in quiets {
                if is_promotion_square(dest, color) {
                    push_promotion_moves(&mut moves, source, dest);
                } else {
                    moves.push(ChessMove::new(source, dest, None));
                }
            }

            let captures = get_pawn_attacks(source, color, enemy);
            for dest in captures {
                if is_promotion_square(dest, color) {
                    push_promotion_moves(&mut moves, source, dest);
                } else {
                    moves.push(ChessMove::new(source, dest, None));
                }
            }

            if let Some(target_sq) = en_passant_target {
                let target_bb = BitBoard::from_square(target_sq);
                if (get_pawn_attacks(source, color, target_bb) & target_bb) != EMPTY {
                    moves.push(ChessMove::new(source, target_sq, None));
                }
            }
        }

        let knights = *self.board.pieces(Piece::Knight) & own;
        for source in knights {
            let destinations = get_knight_moves(source) & !own;
            for dest in destinations {
                moves.push(ChessMove::new(source, dest, None));
            }
        }

        let bishops = *self.board.pieces(Piece::Bishop) & own;
        for source in bishops {
            let destinations = get_bishop_moves(source, occupied) & !own;
            for dest in destinations {
                moves.push(ChessMove::new(source, dest, None));
            }
        }

        let rooks = *self.board.pieces(Piece::Rook) & own;
        for source in rooks {
            let destinations = get_rook_moves(source, occupied) & !own;
            for dest in destinations {
                moves.push(ChessMove::new(source, dest, None));
            }
        }

        let queens = *self.board.pieces(Piece::Queen) & own;
        for source in queens {
            let destinations = (get_bishop_moves(source, occupied) | get_rook_moves(source, occupied)) & !own;
            for dest in destinations {
                moves.push(ChessMove::new(source, dest, None));
            }
        }

        let kings = *self.board.pieces(Piece::King) & own;
        for source in kings {
            let destinations = get_king_moves(source) & !own;
            for dest in destinations {
                moves.push(ChessMove::new(source, dest, None));
            }
            self.add_castling_moves(source, color, occupied, &mut moves);
        }

        moves
    }

    fn add_castling_moves(
        &self,
        king_square: Square,
        color: Color,
        occupied: BitBoard,
        moves: &mut Vec<ChessMove>,
    ) {
        let rights = self.board.castle_rights(color);
        match color {
            Color::White => {
                if king_square != Square::E1 {
                    return;
                }

                if rights.has_kingside()
                    && self.board.piece_on(Square::H1) == Some(Piece::Rook)
                    && self.board.color_on(Square::H1) == Some(Color::White)
                {
                    let path =
                        BitBoard::from_square(Square::F1) | BitBoard::from_square(Square::G1);
                    if (occupied & path) == EMPTY {
                        moves.push(ChessMove::new(Square::E1, Square::G1, None));
                    }
                }

                if rights.has_queenside()
                    && self.board.piece_on(Square::A1) == Some(Piece::Rook)
                    && self.board.color_on(Square::A1) == Some(Color::White)
                {
                    let path = BitBoard::from_square(Square::B1)
                        | BitBoard::from_square(Square::C1)
                        | BitBoard::from_square(Square::D1);
                    if (occupied & path) == EMPTY {
                        moves.push(ChessMove::new(Square::E1, Square::C1, None));
                    }
                }
            }
            Color::Black => {
                if king_square != Square::E8 {
                    return;
                }

                if rights.has_kingside()
                    && self.board.piece_on(Square::H8) == Some(Piece::Rook)
                    && self.board.color_on(Square::H8) == Some(Color::Black)
                {
                    let path =
                        BitBoard::from_square(Square::F8) | BitBoard::from_square(Square::G8);
                    if (occupied & path) == EMPTY {
                        moves.push(ChessMove::new(Square::E8, Square::G8, None));
                    }
                }

                if rights.has_queenside()
                    && self.board.piece_on(Square::A8) == Some(Piece::Rook)
                    && self.board.color_on(Square::A8) == Some(Color::Black)
                {
                    let path = BitBoard::from_square(Square::B8)
                        | BitBoard::from_square(Square::C8)
                        | BitBoard::from_square(Square::D8);
                    if (occupied & path) == EMPTY {
                        moves.push(ChessMove::new(Square::E8, Square::C8, None));
                    }
                }
            }
        }
    }

    fn apply_internal(&mut self, action: ChessMove) {
        if self.board.piece_on(action.get_dest()) == Some(Piece::King) {
            self.captured_king = self.board.color_on(action.get_dest());
        } else {
            self.board = self.board.make_move_new(action);
        }
        self.update_mental_boards();
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
        if self.captured_king == Some(color) {
            return false;
        }
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

fn compute_visibility_bb(&self, player: Color) -> BitBoard {
        let mut visible = EMPTY;

        for index in 0..BOARD_SQUARES {
            let square = square_from_index(index);
            let piece = self.board.piece_on(square);
            let color = self.board.color_on(square);

            if piece.is_some() && color == Some(player) {
                visible |= BitBoard::from_square(square);
                let attacks = self.piece_attacks(square, piece.unwrap(), player);
                visible |= attacks;

                if piece == Some(Piece::Pawn) {
                    let forward_offset: i32 = if player == Color::White { 8 } else { -8 };
                    let fwd_index = index as i32 + forward_offset;
                    if (0..BOARD_SQUARES as i32).contains(&fwd_index) {
                        let fwd_square = square_from_index(fwd_index as usize);
                        visible |= BitBoard::from_square(fwd_square);

                        let rank = index / 8;
                        let start_rank = if player == Color::White { 1 } else { 6 };
                        if rank == start_rank {
                            if self.board.piece_on(fwd_square).is_none() {
                                let dbl_index = index as i32 + 2 * forward_offset;
                                if (0..BOARD_SQUARES as i32).contains(&dbl_index) {
                                    let dbl_square = square_from_index(dbl_index as usize);
                                    visible |= BitBoard::from_square(dbl_square);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Fix: Explicitly search for our pawns that attack the en passant square.
        // We iterate all squares to avoid any BitBoard iterator issues, and use 
        // `self.piece_attacks` to guarantee we get the correct attack pattern.
        if let Some(ep_square) = self.board.en_passant() {
            let forward_offset = if player == Color::White { 8 } else { -8 };
            let target_index = square_index(ep_square) as i32 + forward_offset;
            let target_bb = BitBoard::from_square(square_from_index(target_index as usize));

            let mut can_en_passant = false;

            for index in 0..BOARD_SQUARES {
                let square = square_from_index(index);
                if self.board.piece_on(square) == Some(Piece::Pawn)
                    && self.board.color_on(square) == Some(player)
                {
                    // Utilize the verified piece_attacks function
                    let attacks = self.piece_attacks(square, Piece::Pawn, player);
                    if (attacks & target_bb) != EMPTY {
                        can_en_passant = true;
                        break;
                    }
                }
            }

            if can_en_passant {
                if self.board.piece_on(ep_square) == Some(Piece::Pawn)
                    && self.board.color_on(ep_square) == Some(opposite_color(player))
                {
                    visible |= BitBoard::from_square(ep_square);
                }
            }
        }

        visible
    }
    
    
    fn captured_piece_vector(&self, current_player: Color) -> [f32; CPV_LEN] {
        let enemy = opposite_color(current_player);
        let counts = [
            (Piece::Pawn, 8.0_f32),
            (Piece::Knight, 2.0_f32),
            (Piece::Bishop, 2.0_f32),
            (Piece::Rook, 2.0_f32),
            (Piece::Queen, 1.0_f32),
        ];

        let mut cpv = [0.0_f32; CPV_LEN];
        for (idx, (piece, max_count)) in counts.iter().enumerate() {
            let current = (*self.board.pieces(*piece) & *self.board.color_combined(enemy))
                .popcnt() as f32;
            cpv[idx] = (max_count - current) / max_count;
        }

        cpv
    }

    #[cfg(any(test, feature = "fuzzing"))]
    pub(crate) fn encode_flat(&self) -> Vec<f32> {
        self.encode_flat_with_oracle(false)
    }

    fn encode_flat_with_oracle(&self, oracle: bool) -> Vec<f32> {
        let current_player = self.side_to_move();
        let player_idx = color_index(current_player);
        let visible_squares = if oracle {
            None
        } else {
            Some(self.compute_visibility_bb(current_player))
        };

        let mut encoded = vec![0.0; BOARD_SQUARES * ENCODE_CHANNELS];

        let cpv = self.captured_piece_vector(current_player);
        for square in 0..BOARD_SQUARES {
            let base = square * ENCODE_CHANNELS + CPV_OFFSET;
            encoded[base..base + CPV_LEN].copy_from_slice(&cpv);
        }

        for index in 0..BOARD_SQUARES {
            let view_index = if current_player == Color::White {
                index
            } else {
                mirror_index(index)
            };
            let base = view_index * ENCODE_CHANNELS;
            let square = square_from_index(index);
            let square_bb = BitBoard::from_square(square);

            let is_visible = visible_squares
                .as_ref()
                .map(|bb| (*bb & square_bb) != EMPTY)
                .unwrap_or(true);

            if !is_visible {
                encoded[base + OBS_FOG_CHANNEL] = 1.0;
            } else if let Some(piece) = self.board.piece_on(square) {
                let is_own_piece = self.board.color_on(square) == Some(current_player);
                let mut piece_idx = piece_base_index(piece);
                if !is_own_piece {
                    piece_idx += 6;
                }
                encoded[base + piece_idx] = 1.0;
            } else {
                encoded[base + OBS_EMPTY_CHANNEL] = 1.0;
            }

            match self.mental_pieces[player_idx][index] {
                Some(seen) => {
                    let mut piece_idx = piece_base_index(seen.piece);
                    if seen.color != current_player {
                        piece_idx += 6;
                    }
                    encoded[base + MEMORY_OFFSET + piece_idx] = 1.0;
                }
                None => {
                    encoded[base + MEMORY_OFFSET + MEMORY_EMPTY_CHANNEL] = 1.0;
                }
            }

            let dt = self.delta_t[player_idx][index];
            encoded[base + RECENCY_CHANNEL] = 1.0 / (1.0 + dt);
        }

        let (king_side, queen_side) = self.castle_eligible_internal();
        encoded[CASTLING_KS_CHANNEL] = if king_side { 1.0 } else { 0.0 };
        encoded[ENCODE_CHANNELS + CASTLING_QS_CHANNEL] = if queen_side { 1.0 } else { 0.0 };

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
    #[pyo3(signature = (board=None))]
    fn new(py: Python, board: Option<Py<PyAny>>) -> PyResult<Self> {
        let board = match board {
            Some(obj) => {
                let bound = obj.bind(py);
                let fen = fen_from_py(&bound)?;
                Board::from_str(&fen).map_err(|_| PyValueError::new_err("Invalid FEN string"))?
            }
            None => Board::default(),
        };

        Ok(SelfPlayGame::new_internal(board))
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

    fn legal_actions_uci(&self) -> Vec<String> {
        self.legal_actions_internal()
            .iter()
            .map(|mv| mv.to_string())
            .collect()
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

    #[pyo3(signature = (oracle = false))]
    fn encode(&self, py: Python, oracle: bool) -> PyResult<Py<PyAny>> {
        let flat = self.encode_flat_with_oracle(oracle);
        let cols = ENCODE_CHANNELS;
        let rows = reshape_rows(flat, cols);

        let torch = PyModule::import(py, "torch")
            .map_err(|_| PyValueError::new_err("torch is required for encode()"))?;
        let tensor_fn = torch.getattr("tensor")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("dtype", torch.getattr("float32")?)?;
        let tensor = tensor_fn.call((rows,), Some(&kwargs))?;
        Ok(tensor.unbind())
    }

    #[pyo3(signature = (oracle = false))]
    fn encode_flattened(&self, oracle: bool) -> Vec<f32> {
        self.encode_flat_with_oracle(oracle)
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
        SelfPlayGame::new_internal(Board::default())
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

    fn rust_legal_actions_uci(game: &SelfPlayGame) -> Vec<String> {
        let mut uci_list: Vec<String> = game
            .legal_actions_internal()
            .iter()
            .map(|mv| mv.to_string())
            .collect();
        uci_list.sort();
        uci_list
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
            let rust_game = SelfPlayGame::new_internal(Board::default());

            let py_turn: bool = py_game.getattr("turn")?.extract()?;
            assert_eq!(rust_game.side_to_move() == Color::White, py_turn);

            let py_board: Vec<Vec<i8>> = py_game.getattr("board")?.extract()?;
            assert_eq!(rust_game.board_matrix(), py_board);

            let py_castle: (bool, bool) = py_game.call_method0("castle_eligible")?.extract()?;
            assert_eq!(rust_game.castle_eligible_internal(), py_castle);

            let py_encoded = py_encode_flat(&py_game)?;
            assert_eq!(rust_game.encode_flat(), py_encoded);

            let py_actions = py_legal_actions_uci(&py_game)?;
            let rust_actions = rust_legal_actions_uci(&rust_game);
            assert_eq!(rust_actions, py_actions);

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
            let mut rust_game = SelfPlayGame::new_internal(Board::default());

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

            let py_actions = py_legal_actions_uci(&py_game)?;
            let rust_actions = rust_legal_actions_uci(&rust_game);
            assert_eq!(rust_actions, py_actions);

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
            let rust_game = SelfPlayGame::new_internal(rust_board);

            let mv = chess_mod
                .getattr("Move")?
                .getattr("from_uci")?
                .call1(("a7a8q",))?;
            let py_idx: usize = py_game.call_method1("action_index", (mv,))?.extract()?;

            let rust_idx = rust_game
                .action_index_internal(ChessMove::from_str("a7a8q").unwrap())
                .unwrap();
            assert_eq!(rust_idx, py_idx);

            let py_actions = py_legal_actions_uci(&py_game)?;
            let rust_actions = rust_legal_actions_uci(&rust_game);
            assert_eq!(rust_actions, py_actions);

            Ok(())
        })
    }

    #[test]
    fn en_passant_pawn_visibility() {
        let fen = "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1";
        let board = Board::from_str(fen).expect("valid en passant FEN");
        let game = SelfPlayGame::new_internal(board);

        let visible = game.compute_visibility_bb(Color::White);

        println!("visibility: {visible}");

        let pawn_square = Square::D5;
        assert!((visible & BitBoard::from_square(pawn_square)) != EMPTY);

        let mut attacks = EMPTY;
        for index in 0..BOARD_SQUARES {
            let square = square_from_index(index);
            if game.board.color_on(square) == Some(Color::White) {
                if let Some(piece) = game.board.piece_on(square) {
                    attacks |= game.piece_attacks(square, piece, Color::White);
                }
            }
        }
        assert!((attacks & BitBoard::from_square(pawn_square)) == EMPTY);
    }

    #[test]
    fn test_pseudo_legal_capturing_king() {
        // White Queen on a2, Black King on e8. Black pawn on a7.
        let fen = "4k3/p7/8/8/8/8/Q7/4K3 w - - 0 1";
        let mut board = Board::from_str(fen).unwrap();
        // White moves Queen to e6 (giving check)
        board = board.make_move_new(ChessMove::from_str("a2e6").unwrap());
        // Black ignores check and moves pawn
        board = board.make_move_new(ChessMove::from_str("a7a6").unwrap());
        
        let mut game = SelfPlayGame::new_internal(board);
        let moves = rust_legal_actions_uci(&game);
        assert!(moves.contains(&"e6e8".to_string()));
        
        // Ensure apply_internal doesn't panic
        game.apply_internal(ChessMove::from_str("e6e8").unwrap());
        assert!(!game.king_present(Color::Black));
    }

    #[test]
    fn test_pseudo_legal_castling_through_check() {
        // Black rooks on d8 and f8 attack the castling paths for white. Black king on e8.
        let fen = "3rkr2/8/8/8/8/8/8/R3K2R w KQ - 0 1";
        let board = Board::from_str(fen).unwrap();
        let mut game = SelfPlayGame::new_internal(board);
        let moves = rust_legal_actions_uci(&game);
        assert!(moves.contains(&"e1g1".to_string()));
        assert!(moves.contains(&"e1c1".to_string()));
        
        // Ensure apply_internal doesn't panic
        game.apply_internal(ChessMove::from_str("e1g1").unwrap());
    }

    #[test]
    fn test_pseudo_legal_moving_into_check() {
        // Black rook on f2. White King on e1.
        let fen = "4k3/8/8/8/8/8/5r2/4K3 w - - 0 1";
        let board = Board::from_str(fen).unwrap();
        let mut game = SelfPlayGame::new_internal(board);
        let moves = rust_legal_actions_uci(&game);
        assert!(moves.contains(&"e1f1".to_string()));
        assert!(moves.contains(&"e1f2".to_string()));
        
        // Ensure apply_internal doesn't panic
        game.apply_internal(ChessMove::from_str("e1f1").unwrap());
    }

    #[test]
    fn test_pseudo_legal_unpinning_piece() {
        // White rook on e2. Black King on e8. Black Knight on e7.
        let fen = "4k3/4n3/8/8/8/8/4R3/4K3 b - - 0 1";
        let board = Board::from_str(fen).unwrap();
        let mut game = SelfPlayGame::new_internal(board);
        let moves = rust_legal_actions_uci(&game);
        // Moving knight to c6 unpins it, exposing the king to the rook.
        assert!(moves.contains(&"e7c6".to_string()));
        
        // Ensure apply_internal doesn't panic
        game.apply_internal(ChessMove::from_str("e7c6").unwrap());
    }
}
