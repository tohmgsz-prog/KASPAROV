import chess
import chess.engine

STOCKFISH_PATH = r"C:\stockfish\stockfish\stockfish-windows-x86-64-avx2.exe"


def print_board_coords(board: chess.Board):
    files = "A B C D E F G H"
    print("\n    " + files)
    print("  +-----------------+")
    for rank in range(8, 0, -1):  # 8 -> 1
        row = []
        for file in range(8):     # A -> H
            sq = chess.square(file, rank - 1)
            piece = board.piece_at(sq)
            row.append(piece.symbol() if piece else ".")
        print(f"{rank} | " + " ".join(row) + f" | {rank}")
    print("  +-----------------+")
    print("    " + files + "\n")


def parse_move_input(s: str) -> str | None:
    """
    Aceita:
    - UCI: e2e4, g1f3, e7e8q
    - Humano: E2 E4, e2 e4, E7 E8 Q (promoção opcional)
    Retorna UCI em minúsculas (ex: e2e4, e7e8q) ou None se inválido.
    """
    s = s.strip()
    if not s:
        return None

    low = s.lower().replace("-", " ").replace("to", " ").replace("->", " ")
    parts = [p for p in low.split() if p]

    # Caso UCI direto: "e2e4" ou "e7e8q"
    if len(parts) == 1:
        u = parts[0]
        if len(u) in (4, 5):
            return u
        return None

    # Caso humano: "e2 e4" ou "e7 e8 q"
    if len(parts) in (2, 3):
        frm = parts[0]
        to = parts[1]
        promo = parts[2] if len(parts) == 3 else ""

        if len(frm) != 2 or len(to) != 2:
            return None
        if frm[0] not in "abcdefgh" or to[0] not in "abcdefgh":
            return None
        if frm[1] not in "12345678" or to[1] not in "12345678":
            return None

        uci = frm + to
        if promo:
            promo = promo[:1]
            if promo not in "qrbn":
                return None
            uci += promo
        return uci

    return None


def main():
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

    board = chess.Board()
    print("=== STOCKFISH BOT ===")
    print("Formatos aceites:")
    print("- UCI: e2e4, g1f3, e7e8q")
    print("- Humano: E2 E4   (promoção: E7 E8 Q)")
    print("Escreve 'quit' para sair.\n")

    print_board_coords(board)

    while not board.is_game_over():
        raw = input("O teu lance: ").strip()
        if raw.lower() == "quit":
            break

        move_uci = parse_move_input(raw)
        if move_uci is None:
            print("❌ Formato inválido. Ex: e2e4 ou E2 E4")
            continue

        try:
            move = chess.Move.from_uci(move_uci)
        except ValueError:
            print("❌ Lance inválido (UCI mal formado).")
            continue

        if move not in board.legal_moves:
            print("❌ Lance ilegal.")
            continue

        board.push(move)
        print(f"\n✅ Tu jogaste: {move.uci()}")
        print_board_coords(board)

        if board.is_game_over():
            break

        print("🤖 Stockfish pensa...")
        result = engine.play(board, chess.engine.Limit(time=1.0))
        board.push(result.move)
        print(f"Stockfish: {result.move.uci()}")
        print_board_coords(board)

    print("Fim:", board.result())
    engine.quit()


if __name__ == "__main__":
    main()
