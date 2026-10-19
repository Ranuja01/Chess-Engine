// book.h — native polyglot opening-book reader for the UCI binary (2026-10-19; UCI design §2/§4). Same `.bin` files and the
// same choice rule as ChessAI.pyx opening_book (python-chess reader): the HIGHEST-weight entry for the position (the first on
// ties; entries of weight 0 skipped, python-chess's minimum_weight=1), castling converted from king-takes-rook to e1g1 form.
#pragma once
#include <cstdint>
#include <string>
#include "search_engine.h"

namespace book
{
    uint64_t key(const BoardState &s);   // standard polyglot hash (== chess.polyglot.zobrist_hash)
    // The book move for `s` as a LEGAL UCI string, or "" when the file has none / cannot be opened.
    std::string probe(const std::string &path, const BoardState &s);
}
