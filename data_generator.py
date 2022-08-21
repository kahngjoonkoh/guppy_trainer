import chess
import chess.pgn
import random

from halfkp import get_nnue_indeces, get_dense_indeces

# GAMES_PER_EPOCH = 1_000

# def centipawn_to_wincp(cp_score):
#   return 2 / (1 + np.math.exp(-0.004 * cp_score)) - 1

def get_eval(engine: chess.engine.SimpleEngine, board: chess.Board):
    try:
        info = engine.analyse(board, chess.engine.Limit(time=0.02, depth=20))
        # return info["score"].pov(board.turn).score(mate_score=10000)
        return info["score"].pov(not board.turn).score(mate_score=10000)
    except:
        return False


def generate(pgn_file_path: str, engine_file_path: str):
    engine = chess.engine.SimpleEngine.popen_uci(engine_file_path)
    with open(pgn_file_path) as pgn:
        # from random position in the file
        pgn.seek(0, 2)
        pgn.seek(random.randint(0, pgn.tell()))
        
        game_counter = 1
        total = 1
        while True:
            game = chess.pgn.read_game(pgn)

            if not game:
                pgn.seek(0)
                continue
    
            board = game.board()
            move_counter = 1
            for node in game.mainline():
                board.push(node.move)
                cp = get_eval(engine, board)
                # log(f"{game.headers['Date']} {game.headers['White']} {game.headers['Black']} {game.headers['Result']}")
                log(f"Game: {game_counter} Move: {move_counter} Total: {total}")
                log(f"{board.fen()}\n{board}\n{cp}\n")
                # log(game.headers)
                if not cp:
                    break
                
                try:
                    X = get_nnue_indeces(board)
                except:
                    print(f'Failrue to get halfkp: {board}')
                    continue
                move_counter += 1
                total += 1
                yield X, cp
            game_counter += 1

def load(pgn_file_path: str, engine_file_path: str, n: int):
    engine = chess.engine.SimpleEngine.popen_uci(engine_file_path)
    Xs = []
    Ys = []
    with open(pgn_file_path) as pgn:
        # from random position in the file
        pgn.seek(0, 2)
        pgn.seek(random.randint(0, pgn.tell()))
        
        game_counter = 1
        total = 1
        while True:
            game = chess.pgn.read_game(pgn)

            if not game:
                pgn.seek(0)
                continue
    
            board = game.board()
            move_counter = 1
            for node in game.mainline():
                board.push(node.move)
                cp = get_eval(engine, board)
                # log(f"{game.headers['Date']} {game.headers['White']} {game.headers['Black']} {game.headers['Result']}")
                log(f"Game: {game_counter} Move: {move_counter} Total: {total}")
                log(f"{board.fen()}\n{board}\n{cp}\n")
                # log(game.headers)
                if not cp:
                    break
                
                try:
                    X = get_nnue_indeces(board)
                except:
                    print(f'Failrue to get halfkp: {board}')
                    continue
                move_counter += 1
                total += 1
                Xs.append(X)
                Ys.append(cp/100)
            game_counter += 1
            if move_counter >= n:
                break;
    
    return Xs, Ys

def generate_dense(pgn_file_path: str, engine_file_path: str):
    engine = chess.engine.SimpleEngine.popen_uci(engine_file_path)
    with open(pgn_file_path) as pgn:
        # from random position in the file
        pgn.seek(0, 2)
        pgn.seek(random.randint(0, pgn.tell()))
        
        game_counter = 1
        total = 1
        while True:
            game = chess.pgn.read_game(pgn)

            if not game:
                pgn.seek(0)
                continue
    
            board = game.board()
            move_counter = 1
            for node in game.mainline():
                board.push(node.move)
                cp = get_eval(engine, board)
                # log(f"{game.headers['Date']} {game.headers['White']} {game.headers['Black']} {game.headers['Result']}")
                log(f"Game: {game_counter} Move: {move_counter} Total: {total}")
                log(f"{board.fen()}\n{board}\n{cp}\n")
                # log(game.headers)
                if not cp:
                    break
                
                try:
                    X = get_dense_indeces(board)
                except:
                    print(f'Failrue to get halfkp: {board}')
                    continue
                move_counter += 1
                total += 1
                yield X, cp / 100
            game_counter += 1


#default
log_enabled = False

def log(s):
    global log_enabled
    if log_enabled:
        print(s)

#%% TEST
# Settings
if __name__ == "__main__":
    iterations = 0
    log_enabled = False
    pgn_path = "datasets/lichess_elite_2022-02.pgn"
    engine_path = "stockfish_15_x64_avx2.exe"
    
    generator = generate(pgn_path, engine_path)
    for i, r in enumerate(generator):
        print(i)
        if i >= (iterations - 1):
            generator.close()
