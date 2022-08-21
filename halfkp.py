import chess
from enum import Enum
from enum import IntFlag
import numpy as np
import tensorflow as tf

SQUARE_NB = 64


class PieceSquare(IntFlag):
    NONE     =  0,
    W_PAWN   =  1,
    B_PAWN   =  1 * SQUARE_NB + 1
    W_KNIGHT =  2 * SQUARE_NB + 1
    B_KNIGHT =  3 * SQUARE_NB + 1
    W_BISHOP =  4 * SQUARE_NB + 1
    B_BISHOP =  5 * SQUARE_NB + 1
    W_ROOK   =  6 * SQUARE_NB + 1
    B_ROOK   =  7 * SQUARE_NB + 1
    W_QUEEN  =  8 * SQUARE_NB + 1
    B_QUEEN  =  9 * SQUARE_NB + 1
    W_KING   = 10 * SQUARE_NB + 1
    END      = W_KING # pieces without kings (pawns included)
    B_KING   = 11 * SQUARE_NB + 1
    END2     = 12 * SQUARE_NB + 1

    @staticmethod
    def from_piece(p: chess.Piece, is_white_pov: bool):
        return {
            chess.WHITE: {
                chess.PAWN:     PieceSquare.W_PAWN,
                chess.KNIGHT:   PieceSquare.W_KNIGHT,
                chess.BISHOP:   PieceSquare.W_BISHOP,
                chess.ROOK:     PieceSquare.W_ROOK,
                chess.QUEEN:    PieceSquare.W_QUEEN,
                chess.KING:     PieceSquare.W_KING
            },
            chess.BLACK: {
                chess.PAWN:     PieceSquare.B_PAWN,
                chess.KNIGHT:   PieceSquare.B_KNIGHT,
                chess.BISHOP:   PieceSquare.B_BISHOP,
                chess.ROOK:     PieceSquare.B_ROOK,
                chess.QUEEN:    PieceSquare.B_QUEEN,
                chess.KING:     PieceSquare.B_KING
            }
        }[p.color == is_white_pov][p.piece_type]


def orient(is_white_pov: bool, sq: int):
    # return (63 * (not is_white_pov)) ^ sq if is_white_pov else (63 * (not is_white_pov)) ^ sq ^ 7# Binary XOR to invert bits.
    return (56 * (not is_white_pov)) ^ sq


def make_halfkp_index(is_white_pov: bool, king_sq: int, sq: int, p: chess.Piece):
    return orient(is_white_pov, sq) + PieceSquare.from_piece(p, is_white_pov) + PieceSquare.END * king_sq

def make_nnue_index(is_white_pov: bool, sq:int, p:chess.Piece):
    # print(sq, p)
    # print(f"piecesquare: {PieceSquare.from_piece(p, is_white_pov)} orient: {orient(is_white_pov, sq)} total: {PieceSquare.from_piece(p, is_white_pov) + orient(is_white_pov, sq)}")
    return PieceSquare.from_piece(p, is_white_pov) + orient(is_white_pov, sq)

# Returns SparseTensors
def get_nnue_indeces(board: chess.Board):
    indices = []
    values = []
    turn = board.turn
    for sq, p in board.piece_map().items():
        indices.append([make_nnue_index(turn, sq, p)])
        values.append(1.0)
    return tf.sparse.reorder(tf.sparse.SparseTensor(
            indices=indices, values=values, dense_shape=[768]))

def get_dense_indeces(board: chess.Board):
    indices = []
    values = []
    turn = board.turn
    for sq, p in board.piece_map().items():
        indices.append([make_nnue_index(turn, sq, p)])
        values.append(1.0)
    return tf.sparse.to_dense(tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=[768])).numpy()
