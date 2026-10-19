// book.cpp — see book.h. Polyglot format: 16-byte big-endian entries {key u64, move u16, weight u16, learn u32} sorted by key.
#include "book.h"
#include <fstream>
#include <vector>
#include "book_keys.h"
#include "position.h"

namespace book
{
    uint64_t key(const BoardState &s)
    {
        uint64_t h = 0;
        const uint64_t bb[6] = {s.pawns, s.knights, s.bishops, s.rooks, s.queens, s.kings};
        for (int t = 0; t < 6; ++t)
            for (uint64_t b = bb[t]; b; b &= b - 1)
            {
                const int sq = __builtin_ctzll(b);
                const bool white = (s.occupied_colour[1] >> sq) & 1;
                const int kind = 2 * t + (white ? 1 : 0);          // polyglot: black pawn 0, white pawn 1, black knight 2, ...
                h ^= RANDOM64[64 * kind + sq];                      // 8 * row + file == our square index
            }
        if (s.castling_rights & (1ULL << 7)) h ^= RANDOM64[768];
        if (s.castling_rights & (1ULL << 0)) h ^= RANDOM64[769];
        if (s.castling_rights & (1ULL << 63)) h ^= RANDOM64[770];
        if (s.castling_rights & (1ULL << 56)) h ^= RANDOM64[771];
        if (s.ep_square >= 0)
        {
            // only when a pawn of the side to move stands beside the double-pushed pawn (polyglot / python-chess rule)
            const int f = s.ep_square % 8;
            const int behind = s.turn ? s.ep_square - 8 : s.ep_square + 8;   // square of the pawn that just moved two
            const uint64_t ours = s.pawns & s.occupied_colour[s.turn];
            uint64_t adj = 0;
            if (f > 0) adj |= 1ULL << (behind - 1);
            if (f < 7) adj |= 1ULL << (behind + 1);
            if (adj & ours) h ^= RANDOM64[772 + f];
        }
        if (s.turn) h ^= RANDOM64[780];
        return h;
    }

    namespace
    {
        bool read_entry(std::ifstream &f, uint64_t idx, uint64_t &k, uint16_t &mv, uint16_t &w)
        {
            unsigned char b[16];
            f.seekg((std::streamoff)(idx * 16));
            if (!f.read(reinterpret_cast<char *>(b), 16)) return false;
            k = 0;
            for (int i = 0; i < 8; ++i) k = (k << 8) | b[i];
            mv = (uint16_t)((b[8] << 8) | b[9]);
            w = (uint16_t)((b[10] << 8) | b[11]);
            return true;
        }

        std::string decode(uint16_t mv, const BoardState &s)
        {
            const int to = (mv & 7) + 8 * ((mv >> 3) & 7), from = ((mv >> 6) & 7) + 8 * ((mv >> 9) & 7), pr = (mv >> 12) & 7;
            int t = to;
            if ((s.kings >> from) & 1)                                    // castling stored as king takes own rook
            {
                if (from == 4 && to == 7) t = 6;   if (from == 4 && to == 0) t = 2;
                if (from == 60 && to == 63) t = 62; if (from == 60 && to == 56) t = 58;
            }
            std::string u;
            u += char('a' + from % 8); u += char('1' + from / 8); u += char('a' + t % 8); u += char('1' + t / 8);
            if (pr >= 1 && pr <= 4) u += "nbrq"[pr - 1];
            return u;
        }
    }

    std::string probe(const std::string &path, const BoardState &s)
    {
        std::ifstream f(path, std::ios::binary);
        if (!f) return "";
        f.seekg(0, std::ios::end);
        const uint64_t n = (uint64_t)f.tellg() / 16;
        const uint64_t k = key(s);
        uint64_t lo = 0, hi = n;                                         // lower bound of k
        while (lo < hi)
        {
            const uint64_t mid = lo + (hi - lo) / 2;
            uint64_t mk; uint16_t mv, w;
            if (!read_entry(f, mid, mk, mv, w)) return "";
            if (mk < k) lo = mid + 1; else hi = mid;
        }
        std::string best;
        int best_w = 0;
        for (uint64_t i = lo; i < n; ++i)
        {
            uint64_t mk; uint16_t mv, w;
            if (!read_entry(f, i, mk, mv, w) || mk != k) break;
            if (w < 1 || w <= best_w) continue;                          // highest weight, first on ties
            const std::string u = decode(mv, s);
            Move m;
            if (!pos::parse_uci_move(s, u, m)) continue;                 // never trust a book move blindly
            best = u; best_w = w;
        }
        return best;
    }
}
