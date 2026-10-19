// uci_main.cpp — native UCI front end (2026-10-19; dev_notes/UCI-BINARY-DESIGN-2026-10-19.md). Built only by
// selfplay/build_uci.sh; the Cython module does not include it.
// M1: position / d / perft / divide.  M2: uci / isready / ucinewgame / setoption / go / stop / quit, info + bestmove.
// Debug prints of the engine go to stdout (std::cout) — at start-up std::cout is re-pointed at stderr, and UCI is written
// through `g_out` on the real stdout, so the UCI stream stays clean without touching any engine print.
#include <atomic>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include "position.h"
#include "cpp_bitboard.h"
#include "cache_management.h"

extern std::atomic<bool> time_up;   // search_engine.cpp

namespace
{
    std::ostream *g_out = nullptr;
    std::mutex g_out_mx;
    void say(const std::string &s)
    {
        std::lock_guard<std::mutex> lk(g_out_mx);
        *g_out << s << std::endl;
    }

    pos::Game g_game;
    bool g_have_pos = false;

    // ── engine initialisation (lazy: env knobs latch at the first initialize_engine) ─────────────────────────────────────
    bool g_inited = false;
    bool g_resign = false;
    const ConfigData *g_base = nullptr;   // the preset `go` without limits plays exactly (parity with the UI / harness)
    int g_def_depth = 64, g_def_nodes = 0;
    int g_last_side = -1;

    void ensure_init()
    {
        if (g_inited) return;
        if (!std::getenv("V2_PRESET") && !std::getenv("EVAL_ARM"))
            setenv("V2_PRESET", "shipped", 0);          // default = the official ship (gen 4)
        BoardState s(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, true, 0, -1, 0, 1);
        std::string err;
        pos::parse_fen(pos::START_FEN, s, err);
        std::vector<BoardState> h;
        std::unordered_map<uint64_t, int> c;
        initialize_engine(h, c, s.pawns, s.knights, s.bishops, s.rooks, s.queens, s.kings, s.occupied, s.occupied_colour[1],
                          s.occupied_colour[0], s.promoted, s.castling_rights, s.ep_square, s.halfmove_clock, s.fullmove_number,
                          s.turn, true);
        if (!g_base) g_base = Config::ACTIVE;
        g_def_depth = Config::MAX_ITERATIVE_DEPTH;
        g_def_nodes = Config::NODE_LIMIT;
        g_inited = true;
    }

    // ── search thread ────────────────────────────────────────────────────────────────────────────────────────────────────
    std::thread g_worker;
    std::atomic<bool> g_searching{false};
    ConfigData g_cfg;
    std::chrono::steady_clock::time_point g_go_t0;

    std::string score_str(int score, int root_size)
    {
        const int a = score < 0 ? -score : score;
        if (a >= 9000000)
        {
            const int plies = 9999999 - a - root_size;
            if (plies >= 0) return "mate " + std::to_string(score > 0 ? (plies + 1) / 2 : -((plies + 1) / 2));
        }
        return "cp " + std::to_string(score / 10);
    }

    void on_iter(int depth, int score, long long nodes, const Move *pv, int pv_len, int root_size)
    {
        const long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - g_go_t0).count();
        std::string line = "info depth " + std::to_string(depth) + " score " + score_str(score, root_size) + " nodes " +
                           std::to_string(nodes) + " time " + std::to_string(ms) + " nps " +
                           std::to_string(ms > 0 ? nodes * 1000 / ms : 0);
        // The engine's PV table can hold stale entries past the real line ⇒ replay it from the root, stop at the first
        // move that is not legal there (a GUI rejects an illegal PV).
        if (pv_len > 0 && g_have_pos)
        {
            BoardState s = g_game.history.back();
            std::string pvs;
            for (int i = 0; i < pv_len; ++i)
            {
                Move m;
                if (!pos::parse_uci_move(s, pos::move_to_uci(pv[i]), m)) break;
                pvs += " " + pos::move_to_uci(m);
                s = pos::apply(s, m);
            }
            if (!pvs.empty()) line += " pv" + pvs;
        }
        say(line);
    }

    void wait_search()
    {
        if (g_worker.joinable()) g_worker.join();
    }

    void stop_search()
    {
        if (g_searching.load())
        {
            g_uci_stop.store(true);
            time_up.store(true);
        }
        wait_search();
    }

    // `go` parameters → the existing preset mechanism (design §4). Nothing given ⇒ the base preset unchanged.
    void cmd_go(std::istringstream &in)
    {
        stop_search();
        ensure_init();
        if (!g_have_pos) { std::string e; BoardState r(0,0,0,0,0,0,0,0,0,0,true,0,-1,0,1); pos::parse_fen(pos::START_FEN, r, e); pos::reset(g_game, r); g_have_pos = true; }
        long long wtime = -1, btime = -1, winc = 0, binc = 0, movetime = -1;
        int mtg = 0, depth = 0;
        long long nodes = 0;
        bool infinite = false;
        std::string t;
        while (in >> t)
        {
            if (t == "wtime") in >> wtime; else if (t == "btime") in >> btime;
            else if (t == "winc") in >> winc; else if (t == "binc") in >> binc;
            else if (t == "movestogo") in >> mtg; else if (t == "movetime") in >> movetime;
            else if (t == "depth") in >> depth; else if (t == "nodes") in >> nodes;
            else if (t == "infinite") infinite = true;
        }
        const BoardState &root = g_game.history.back();
        g_cfg = *g_base;
        Config::MAX_ITERATIVE_DEPTH = depth > 0 ? depth : g_def_depth;
        Config::NODE_LIMIT = nodes > 0 ? (int)std::min<long long>(nodes, INT_MAX) : g_def_nodes;
        const long long mine = root.turn ? wtime : btime, inc = root.turn ? winc : binc;
        const double overhead = 0.03;
        if (movetime >= 0)
        {
            const double s = std::max(0.01, movetime / 1000.0 - overhead);
            g_cfg.TIME_LIMIT = s;
            g_cfg.MOVE_TIMES.fill(s);
        }
        else if (mine >= 0)
        {
            // PLACEHOLDER clock rule (design §4; the adaptive time manager replaces it later)
            const double rem = mine / 1000.0, incs = inc / 1000.0;
            const double budget = rem / (mtg > 0 ? mtg : 30) + 0.8 * incs;
            const double hard = std::max(0.01, std::min(4.0 * budget, rem / 4.0) - overhead);
            g_cfg.TIME_LIMIT = hard;
            for (int d = 0; d < 64; ++d)
                g_cfg.MOVE_TIMES[d] = std::min(hard, budget * g_base->MOVE_TIMES[d] / std::max(1e-9, g_base->TIME_LIMIT));
        }
        else if (infinite || depth > 0 || nodes > 0)
        {
            g_cfg.TIME_LIMIT = 1e9;
            g_cfg.MOVE_TIMES.fill(1e9);
        }
        Config::ACTIVE = &g_cfg;
        Config::RESIGN_THRESHOLD = g_resign ? -15000 : INT_MIN / 2;
        Config::side_to_play = root.turn;
        if (g_last_side != (int)root.turn) { if (g_last_side != -1) clearAllCaches(); g_last_side = root.turn; }

        g_uci_stop.store(false);
        g_iter_callback = &on_iter;
        g_searching.store(true);
        g_go_t0 = std::chrono::steady_clock::now();
        std::vector<BoardState> hist = g_game.history;          // get_engine_move pushes its own move: search a copy
        std::unordered_map<uint64_t, int> counts = g_game.counts;
        g_worker = std::thread([hist = std::move(hist), counts = std::move(counts), root]() mutable {
            MoveData md = get_engine_move(hist, counts);
            std::vector<Move> legal;
            pos::legal_moves(root, legal);
            std::string best = "0000";
            if (md.a >= 1)
            {
                Move m((md.a - 1) + (md.b - 1) * 8, (md.c - 1) + (md.d - 1) * 8, 0);
                for (const Move &l : legal)
                    if (l.from_square == m.from_square && l.to_square == m.to_square &&
                        (md.promotion <= 1 ? l.promotion <= 1 : l.promotion == md.promotion)) { best = pos::move_to_uci(l); break; }
            }
            if (best == "0000" && !legal.empty()) best = pos::move_to_uci(legal[0]);   // never leave the GUI without a move
            g_iter_callback = nullptr;
            g_searching.store(false);
            say("bestmove " + best);
        });
    }

    void cmd_setoption(std::istringstream &in)
    {
        std::string t, name, value;
        in >> t;   // "name"
        while (in >> t && t != "value") name += (name.empty() ? "" : " ") + t;
        while (in >> t) value += (value.empty() ? "" : " ") + t;
        auto latch = [&](const char *env, const std::string &v) {
            if (g_inited) { say("info string " + name + " needs a restart (eval knobs latch at the first search)"); return; }
            setenv(env, v.c_str(), 1);
        };
        if (name == "EvalArm") { if (value == "0") { unsetenv("V2_PRESET"); latch("EVAL_ARM", "0"); } else latch("EVAL_ARM", value); }
        else if (name == "V2Preset") latch("V2_PRESET", value);
        else if (name == "Preset")
        {
            if (value == "LIGHTNING") g_base = &Configs::LIGHTNING;
            else if (value == "BLITZ") g_base = &Configs::BLITZ;
            else if (value == "STANDARD") g_base = &Configs::STANDARD;
            else if (value == "LONG_FORMAT") g_base = &Configs::LONG_FORMAT;
            else say("info string unknown Preset " + value);
        }
        else if (name == "Resign") g_resign = (value == "true");
        else say("info string unknown option " + name);
    }

    void cmd_position(std::istringstream &in)
    {
        std::string tok, fen;
        in >> tok;
        if (tok == "startpos") { fen = pos::START_FEN; in >> tok; }
        else if (tok == "fen") { while (in >> tok && tok != "moves") fen += (fen.empty() ? "" : " ") + tok; }
        else { say("info string position: expected startpos or fen"); return; }
        BoardState root(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, true, 0, -1, 0, 1);
        std::string err;
        if (!pos::parse_fen(fen, root, err)) { say("info string bad FEN: " + err); return; }
        pos::Game g;
        pos::reset(g, root);
        if (tok == "moves")
            while (in >> tok)
            {
                Move m;
                if (!pos::parse_uci_move(g.history.back(), tok, m))
                {
                    say("info string illegal move " + tok + " in " + pos::to_fen(g.history.back()) + " (position unchanged)");
                    return;
                }
                pos::push(g, m);
            }
        g_game = std::move(g);
        g_have_pos = true;
    }

    void cmd_perft(std::istringstream &in, bool divide)
    {
        int depth = 0;
        in >> depth;
        if (!g_have_pos || depth < 1) { say("info string perft: set a position and a depth >= 1"); return; }
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
                say(pos::move_to_uci(m) + ": " + std::to_string(n));
                total += n;
            }
        }
        else total = pos::perft(s, depth);
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        say("perft " + std::to_string(depth) + " " + std::to_string(total) + " time " + std::to_string((int)(sec * 1000)) + " ms");
    }
}

int main()
{
    static std::ostream real_out(std::cout.rdbuf());   // the real stdout, for UCI only
    g_out = &real_out;
    std::cout.rdbuf(std::cerr.rdbuf());                 // engine debug prints → stderr
    initialize_attack_tables();
    initializeZobrist();
    std::string line;
    while (std::getline(std::cin, line))
    {
        std::istringstream in(line);
        std::string cmd;
        in >> cmd;
        if (cmd == "uci")
        {
            say("id name NN Engine (v2 gen 4)");
            say("id author Ranuja");
            say("option name Preset type combo default STANDARD var LIGHTNING var BLITZ var STANDARD var LONG_FORMAT");
            say("option name EvalArm type spin default 1 min 0 max 1");
            say("option name V2Preset type string default shipped");
            say("option name Resign type check default false");
            say("uciok");
        }
        else if (cmd == "isready") { ensure_init(); say("readyok"); }
        else if (cmd == "ucinewgame") { stop_search(); ensure_init(); clearAllCaches(); g_last_side = -1; }
        else if (cmd == "setoption") cmd_setoption(in);
        else if (cmd == "position") { stop_search(); cmd_position(in); }
        else if (cmd == "go") cmd_go(in);
        else if (cmd == "stop") stop_search();
        else if (cmd == "quit") { stop_search(); break; }
        else if (cmd == "d") say(g_have_pos ? pos::to_fen(g_game.history.back()) : std::string("(no position)"));
        else if (cmd == "perft") cmd_perft(in, false);
        else if (cmd == "divide") cmd_perft(in, true);
        else if (!cmd.empty()) say("info string unknown command: " + cmd);
    }
    wait_search();
    return 0;
}
