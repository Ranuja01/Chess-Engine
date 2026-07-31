"""Create a WSL-native executable launcher for Mediocre and confirm it drives via popen_uci."""
import os, chess, chess.engine

JAR = "/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Mediocre/mediocre_v0.5.jar"
WRAP = "/home/ranuja/mediocre_uci.sh"

with open(WRAP, "w") as f:
    f.write('#!/bin/bash\nexec java -Xmx1024M -jar "%s"\n' % JAR)
os.chmod(WRAP, 0o755)
print("wrapper:", WRAP, oct(os.stat(WRAP).st_mode & 0o777))

eng = chess.engine.SimpleEngine.popen_uci(WRAP)   # exec the wrapper as the UCI engine
try:
    print("ID:", eng.id)
    b = chess.Board()
    r = eng.play(b, chess.engine.Limit(depth=8), info=chess.engine.INFO_ALL)
    print("move:", r.move, "depth", r.info.get("depth"), "nodes", r.info.get("nodes"))
    print("WRAPPER_OK")
finally:
    eng.quit()
