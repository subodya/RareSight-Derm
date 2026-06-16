"""Run the ACTUAL evaluate.py main() on a seeded, reduced episode set, then in the
SAME process recompute the forward() fusion accuracy on freshly seeded episodes.
If evaluate.py != ~54.66, eval_results.json's 63.56 reflects a live code-path
difference, not a stale file."""
import sys, os, numpy as np, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import importlib
ev = importlib.import_module("src.training.evaluate")

SEED = 42
ev.N_EPISODES = 100
ev.OUT_PATH = "checkpoints/_diag_eval_results_100.json"  # don't clobber real results
np.random.seed(SEED); torch.manual_seed(SEED)
print(">>> Running real evaluate.main() with N_EPISODES=100, seed=42\n")
ev.main()
