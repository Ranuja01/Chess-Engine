// position.h — game-state plumbing for the native UCI binary (2026-10-19; dev_notes/UCI-BINARY-DESIGN-2026-10-19.md §2).
// FEN in/out, UCI move strings, applying the game's moves to the engine's BoardState history exactly as the search's
// make_move does, and perft over the SEARCH's own move generator (generateLegalMovesReordered). Not used by the Cython path.
#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include "search_engine.h"

namespace pos
{
    // The engine's root bookkeeping: the state history get_engine_move reads and the repetition counts it consults.
    struct Game
    {
        std::vector<BoardState> history;
        std::unordered_map<uint64_t, int> counts;
    };

    // FEN → BoardState. Castling becomes the engine's rook-square mask (python-chess `clean_castling_rights` semantics: a
    // right is kept only if its king and rook stand on their home squares). Standard chess only (KQkq / '-').
    bool parse_fen(const std::string &fen, BoardState &out, std::string &err);
    std::string to_fen(const BoardState &s);
    inline const char *START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    // Legal moves in the side to move's view, from the search's generator (promotion field: 1 = none, 2-5 = N/B/R/Q).
    void legal_moves(const BoardState &s, std::vector<Move> &out);
    std::string move_to_uci(const Move &m);
    // Matches the string against the LEGAL moves (so it can never feed an illegal move to update_state).
    bool parse_uci_move(const BoardState &s, const std::string &uci, Move &out);

    // The position after `m` — update_state plus the halfmove / fullmove / turn bookkeeping of search_engine.cpp make_move.
    BoardState apply(const BoardState &s, const Move &m);
    // The engine's repetition key (placement + side to move, as generateZobristHash).
    uint64_t rep_key(const BoardState &s);

    void reset(Game &g, const BoardState &root);   // history = {root}, counts = {root: 1}
    void push(Game &g, const Move &m);             // apply + record, as make_move does

    uint64_t perft(const BoardState &s, int depth);
}
