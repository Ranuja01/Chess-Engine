// uci_main.cpp — native UCI front end (2026-10-19; dev_notes/UCI-BINARY-DESIGN-2026-10-19.md). Built only by
// `overnight_runner.sh build_uci`; the Cython module does not include it.
// M1 (this version): the command loop with `position`, `d` (print FEN), `perft N`, `divide N`, `quit` — to verify FEN parsing,
// move application and the SEARCH's move generator against python-chess. UCI search commands arrive in M2.
#include <chrono>
#include <iostream>
#include <sstream>
#include <string>
#include "position.h"
#include "cpp_bitboard.h"
#include "cache_management.h"

namespace
{
    pos::Game g_game;
    bool g_ok = false;

    // `position startpos|fen <FEN> [moves m1 m2 ...]` — rebuilds the whole history every time (design §4).
    void cmd_position(std::istringstream &in, std::ostream &out)
    {
        std::string tok, fen;
        in >> tok;
        if (tok == "startpos") { fen = pos::START_FEN; in >> tok; }
        else if (tok == "fen")
        {
            while (in >> tok && tok != "moves") fen += (fen.empty() ? "" : " ") + tok;
        }
        else { out << "info string position: expected startpos or fen" << std::endl; return; }
        BoardState root(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, true, 0, -1, 0, 1);
        std::string err;
        if (!pos::parse_fen(fen, root, err)) { out << "info string bad FEN: " << err << std::endl; return; }
        pos::Game g;
        pos::reset(g, root);
        if (tok == "moves")
            while (in >> tok)
            {
                Move m;
                if (!pos::parse_uci_move(g.history.back(), tok, m))
                {
                    out << "info string illegal move " << tok << " in " << pos::to_fen(g.history.back()) << " (position unchanged)" << std::endl;
                    return;
                }
                pos::push(g, m);
            }
        g_game = std::move(g);
        g_ok = true;
    }

    void cmd_perft(std::istringstream &in, std::ostream &out, bool divide)
    {
        int depth = 0;
        in >> depth;
        if (!g_ok || depth < 1) { out << "info string perft: set a position and a depth >= 1" << std::endl; return; }
        const auto t0 = std::chrono::steady_clock::now();
        const BoardState &s = g_game.history.back();
        uint64_t total = 0;
        if (divide)
        {
            std::vector<Move> ms;
            pos::legal_moves(s, ms);
            for (const Move &m : ms)
            {
                const uint64_t n = pos::perft(pos::apply(s, m), depth - 1);
                out << pos::move_to_uci(m) << ": " << n << "\n";
                total += n;
            }
        }
        else total = pos::perft(s, depth);
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        out << "perft " << depth << " " << total << " time " << (int)(sec * 1000) << " ms" << std::endl;
    }
}

int main()
{
    initialize_attack_tables();
    initializeZobrist();
    std::string line;
    std::ostream &out = std::cout;
    while (std::getline(std::cin, line))
    {
        std::istringstream in(line);
        std::string cmd;
        in >> cmd;
        if (cmd == "quit") break;
        else if (cmd == "position") cmd_position(in, out);
        else if (cmd == "d") out << (g_ok ? pos::to_fen(g_game.history.back()) : std::string("(no position)")) << std::endl;
        else if (cmd == "perft") cmd_perft(in, out, false);
        else if (cmd == "divide") cmd_perft(in, out, true);
        else if (!cmd.empty()) out << "info string unknown command: " << cmd << std::endl;
    }
    return 0;
}
