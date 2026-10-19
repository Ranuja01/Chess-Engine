"""Build selfplay/GAMES-INDEX.csv: one row per match folder in selfplay/games/ (live or archived).

Reads only the small top-level files that stay in place after archiving (tournament.json / sprt.json / summary.csv)
plus E:\\chess_data\\archive\\manifest.csv for the archive path. Cheap: no per-game walk.
Run (Windows or WSL python):  python selfplay/build_games_index.py
"""
import os, json, csv, datetime

THIS = os.path.dirname(os.path.abspath(__file__))
GAMES = os.path.join(THIS, "games")
OUT = os.path.join(THIS, "GAMES-INDEX.csv")
MANIFESTS = [r"E:\chess_data\archive\manifest.csv", "/mnt/e/chess_data/archive/manifest.csv"]
ARCHIVE_DIR = r"E:\chess_data\archive\games"

COLS = ["tag", "last_write", "source", "p1", "p2", "preset", "games", "W", "L", "D", "score_pct", "elo", "elo_margin",
        "decision", "p1_config", "p2_config", "archived", "archive", "archive_files", "archive_bytes"]


def load_manifest():
    for p in MANIFESTS:
        if os.path.exists(p):
            with open(p, newline="") as f:
                return {r["tag"]: r for r in csv.DictReader(f) if r["status"] == "deleted"}
    return {}


def from_json(path, src):
    with open(path, encoding="utf-8") as f:
        j = json.load(f)
    W, L, D = j.get("p1_W", j.get("W")), j.get("p1_L", j.get("L")), j.get("p1_D", j.get("D"))
    return {"source": src, "p1": j.get("p1_label", ""), "p2": j.get("p2_label", ""), "preset": j.get("preset", ""),
            "games": j.get("decided", j.get("games", "")), "W": W, "L": L, "D": D, "score_pct": j.get("score_pct", ""),
            "elo": j.get("elo", ""), "elo_margin": j.get("elo_margin", ""), "decision": j.get("decision", ""),
            "p1_config": j.get("p1_config", ""), "p2_config": j.get("p2_config", "")}


def from_summary(path, score_col="p1_score", src="summary"):
    # Per-game rows with a score in {1, 0.5, 0}: summary.csv (p1_score; labels from white/black) or the older
    # selfplay.py results.csv (our_score, no labels => p1 "ours").
    W = L = D = 0; labels = set(); p1 = "" if score_col == "p1_score" else "ours"
    with open(path, newline="", encoding="utf-8", errors="ignore") as f:
        for r in csv.DictReader(f):
            try:
                s = float(r.get(score_col, ""))
            except ValueError:
                continue
            W += s == 1.0; L += s == 0.0; D += s == 0.5
            labels.update([r.get("white", ""), r.get("black", "")])
            if not p1 and r.get("p1_color") in ("white", "black"):
                p1 = r.get(r["p1_color"], "")
    n = W + L + D
    return {"source": src, "p1": p1, "p2": next((x for x in labels if x and x != p1), ""), "games": n,
            "W": W, "L": L, "D": D, "score_pct": round(100.0 * (W + 0.5 * D) / n, 1) if n else ""}


def main():
    man = load_manifest()
    rows = []
    for tag in sorted(os.listdir(GAMES)):
        d = os.path.join(GAMES, tag)
        if not os.path.isdir(d):
            continue
        row = dict.fromkeys(COLS, "")
        row["tag"] = tag
        tops = [os.path.join(d, x) for x in os.listdir(d) if os.path.isfile(os.path.join(d, x))]
        if tops:
            row["last_write"] = datetime.datetime.fromtimestamp(max(os.path.getmtime(p) for p in tops)).strftime("%Y-%m-%d")
        try:
            if os.path.exists(os.path.join(d, "tournament.json")):
                row.update(from_json(os.path.join(d, "tournament.json"), "tournament"))
            elif os.path.exists(os.path.join(d, "sprt.json")):
                row.update(from_json(os.path.join(d, "sprt.json"), "sprt"))
            elif os.path.exists(os.path.join(d, "summary.csv")):
                row.update(from_summary(os.path.join(d, "summary.csv")))
            elif os.path.exists(os.path.join(d, "results.csv")):
                row.update(from_summary(os.path.join(d, "results.csv"), "our_score", "results"))
            else:
                row["source"] = "none"
        except Exception as e:  # a malformed file must not drop the tag from the index
            row["source"] = "error: %s" % type(e).__name__
        m = man.get(tag)
        if m:
            row.update({"archived": m["when"][:10], "archive": os.path.join(ARCHIVE_DIR, tag + ".tar.zst"),
                        "archive_files": m["files"], "archive_bytes": m["archive_bytes"], "last_write": m["newest"]})
        rows.append(row)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    print("wrote %s: %d tags (%d archived)" % (OUT, len(rows), sum(1 for r in rows if r["archived"])))


if __name__ == "__main__":
    main()
