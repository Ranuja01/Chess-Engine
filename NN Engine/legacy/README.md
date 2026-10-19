# legacy/ — inactive code kept for history (moved here 2026-10-19, history preserved via git mv)

The owner's earlier Python/NN engine (NNEngine.py, GUI.py, Rules.py, nnue*.py, create_weights*.py), its trained models
(*.onnx / *.keras, weights/, weights_quant/, NNTrainers/), the old eval_func Cython module, 2024 Windows builds, old tests and
notes, and old_CE/ (copies of an older engine version). Nothing in the live engine, diagnostics or self-play imports these.
Kept for the NN phase. Cython_Chess and main.py stay at the top level: ChessAI.pyx imports Cython_Chess, main.py is the
documented single-position test (NN Engine/CLAUDE.md).
