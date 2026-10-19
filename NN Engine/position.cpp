// position.cpp — see position.h. Mirrors search_engine.cpp make_move (2026-10-19 audit, search_engine.cpp:3237) so a game fed
// through UCI reaches the engine in exactly the state the Cython path would hand it.
#include "position.h"
#include <cctype>
#include <cstring>
#include <sstream>
#include "cpp_bitboard.h"
#include "cache_management.h"
#include "move_gen.h"

namespace pos
{
    namespace
    {
        const char *PIECES = "pnbrqk";   // index + 1 = python-chess piece type

        int sq_of(char f, char r)
        {
            if (f < 'a' || f > 'h' || r < '1' || r > '8') return -1;
            return (r - '1') * 8 + (f - 'a');
        }
    }

    bool parse_fen(const std::string &fen, BoardState &out, std::string &err)
    {
        std::istringstream in(fen);
        std::string place, side, castle = "-", ep = "-";
        int half = 0, full = 1;
        if (!(in >> place >> side)) { err = "FEN needs at least placement and side to move"; return false; }
        in >> castle >> ep;
        if (!(in >> half)) half = 0;
        if (!(in >> full)) full = 1;

        uint64_t bb[6] = {0, 0, 0, 0, 0, 0}, white = 0, black = 0;
        int rank = 7, file = 0;
        for (char c : place)
        {
            if (c == '/') { if (file != 8) { err = "FEN rank does not have 8 files"; return false; } --rank; file = 0; continue; }
            if (std::isdigit((unsigned char)c)) { file += c - '0'; if (file > 8) { err = "FEN rank overflows"; return false; } continue; }
            const char *p = std::strchr(PIECES, std::tolower((unsigned char)c));
            if (!p || file > 7 || rank < 0) { err = std::string("bad FEN piece '") + c + "'"; return false; }
            const uint64_t m = 1ULL << (rank * 8 + file);
            bb[p - PIECES] |= m;
            (std::isupper((unsigned char)c) ? white : black) |= m;
            ++file;
        }
        if (rank != 0 || file != 8) { err = "FEN placement must have 8 ranks"; return false; }
        if (__builtin_popcountll(bb[5] & white) != 1 || __builtin_popcountll(bb[5] & black) != 1) { err = "FEN needs one king per side"; return false; }
        if (side != "w" && side != "b") { err = "FEN side to move must be w or b"; return false; }
        const bool turn = side == "w";

        uint64_t rights = 0;
        if (castle != "-")
            for (char c : castle)
            {
                const uint64_t K = bb[5], R = bb[3];
                switch (c)
                {
                case 'K': if ((K & white & (1ULL << 4)) && (R & white & (1ULL << 7))) rights |= 1ULL << 7; break;
                case 'Q': if ((K & white & (1ULL << 4)) && (R & white & (1ULL << 0))) rights |= 1ULL << 0; break;
                case 'k': if ((K & black & (1ULL << 60)) && (R & black & (1ULL << 63))) rights |= 1ULL << 63; break;
                case 'q': if ((K & black & (1ULL << 60)) && (R & black & (1ULL << 56))) rights |= 1ULL << 56; break;
                default: err = std::string("unsupported castling field '") + castle + "' (standard chess only)"; return false;
                }
            }
        int ep_sq = -1;
        if (ep != "-")
        {
            if (ep.size() != 2 || (ep_sq = sq_of(ep[0], ep[1])) < 0) { err = "bad FEN en-passant square"; return false; }
        }
        out = BoardState(bb[0], bb[1], bb[2], bb[3], bb[4], bb[5], white, black, white | black, 0ULL, turn, rights, ep_sq, half, full);
        return true;
    }

    std::string to_fen(const BoardState &s)
    {
        std::string f;
        const uint64_t bb[6] = {s.pawns, s.knights, s.bishops, s.rooks, s.queens, s.kings};
        for (int r = 7; r >= 0; --r)
        {
            int empty = 0;
            for (int fl = 0; fl < 8; ++fl)
            {
                const uint64_t m = 1ULL << (r * 8 + fl);
                char c = 0;
                for (int t = 0; t < 6; ++t)
                    if (bb[t] & m) c = (s.occupied_colour[1] & m) ? (char)std::toupper(PIECES[t]) : PIECES[t];
                if (!c) { ++empty; continue; }
                if (empty) { f += char('0' + empty); empty = 0; }
                f += c;
            }
            if (empty) f += char('0' + empty);
            if (r) f += '/';
        }
        f += s.turn ? " w " : " b ";
        std::string c;
        if (s.castling_rights & (1ULL << 7)) c += 'K';
        if (s.castling_rights & (1ULL << 0)) c += 'Q';
        if (s.castling_rights & (1ULL << 63)) c += 'k';
        if (s.castling_rights & (1ULL << 56)) c += 'q';
        f += c.empty() ? "-" : c;
        f += ' ';
        if (s.ep_square >= 0) { f += char('a' + s.ep_square % 8); f += char('1' + s.ep_square / 8); }
        else f += '-';
        f += " " + std::to_string(s.halfmove_clock) + " " + std::to_string(s.fullmove_number);
        return f;
    }

    void legal_moves(const BoardState &s, std::vector<Move> &out)
    {
        out.clear();
        generateLegalMovesReordered(out, s.castling_rights, ~0ULL, ~0ULL, s.occupied, s.occupied_colour[1],
                                    s.occupied_colour[!s.turn], s.occupied_colour[s.turn], s.pawns, s.knights, s.bishops,
                                    s.rooks, s.queens, s.kings, s.ep_square, s.turn, 0, Move());
    }

    std::string move_to_uci(const Move &m)
    {
        std::string u;
        u += char('a' + m.from_square % 8); u += char('1' + m.from_square / 8);
        u += char('a' + m.to_square % 8);   u += char('1' + m.to_square / 8);
        if (m.promotion >= 2 && m.promotion <= 5) u += PIECES[m.promotion - 1];
        return u;
    }

    bool parse_uci_move(const BoardState &s, const std::string &uci, Move &out)
    {
        std::vector<Move> ms;
        legal_moves(s, ms);
        for (const Move &m : ms)
            if (move_to_uci(m) == uci) { out = m; return true; }
        return false;
    }

    BoardState apply(const BoardState &s, const Move &m)
    {
        uint64_t pawns = s.pawns, knights = s.knights, bishops = s.bishops, rooks = s.rooks, queens = s.queens, kings = s.kings;
        uint64_t white = s.occupied_colour[1], black = s.occupied_colour[0], occ = s.occupied, promoted = s.promoted;
        uint64_t rights = s.castling_rights;
        int ep = s.ep_square, half = s.halfmove_clock, full = s.fullmove_number;
        const uint64_t to_bb = BB_SQUARES[m.to_square], from_bb = BB_SQUARES[m.from_square];
        const bool pawn_move = (from_bb & s.pawns) != 0;
        const bool capture = (to_bb & s.occupied_colour[!s.turn]) || (pawn_move && (int)m.to_square == s.ep_square);
        update_state(m.to_square, m.from_square, pawns, knights, bishops, rooks, queens, kings, occ, white, black, promoted,
                     rights, ep, m.promotion, s.turn);
        // Halfmove clock: reset on a pawn move or a capture (the FIDE rule, = python-chess, which is what the Cython path hands
        // the engine at the root). ☠️ Quirk noted for the correctness step: search_engine.cpp make_move tests the pawn bit AFTER
        // update_state moved it, so inside search only captures reset the clock.
        half = (capture || pawn_move) ? 0 : half + 1;
        if (!s.turn) full += 1;
        return BoardState(pawns, knights, bishops, rooks, queens, kings, white, black, occ, promoted, !s.turn, rights, ep, half, full);
    }

    uint64_t rep_key(const BoardState &s)
    {
        return generateZobristHash(s.pawns, s.knights, s.bishops, s.rooks, s.queens, s.kings, s.occupied_colour[1],
                                   s.occupied_colour[0], s.turn);
    }

    void reset(Game &g, const BoardState &root)
    {
        g.history.clear();
        g.counts.clear();
        g.history.push_back(root);
        g.counts[rep_key(root)]++;
    }

    void push(Game &g, const Move &m)
    {
        const BoardState n = apply(g.history.back(), m);
        g.counts[rep_key(n)]++;
        g.history.push_back(n);
    }

    uint64_t perft(const BoardState &s, int depth)
    {
        std::vector<Move> ms;
        legal_moves(s, ms);
        if (depth <= 1) return depth <= 0 ? 1 : ms.size();
        uint64_t n = 0;
        for (const Move &m : ms) n += perft(apply(s, m), depth - 1);
        return n;
    }
}
