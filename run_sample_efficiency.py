"""
Standalone script to run the sample-efficiency experiment.

Stdout is tee'd to results/sample_efficiency/run.log so you can follow
progress with `tail -f results/sample_efficiency/run.log`.

The result dict (se_results) is saved to results/sample_efficiency/se_results.pt
and can be loaded back in the notebook with:

    import torch
    se_results = torch.load("results/sample_efficiency/se_results.pt", weights_only=False)
"""

import sys
import os
import torch

# ── redirect stdout to log file (and keep it in the console too) ──────────────
LOG_DIR = "results/sample_efficiency/test"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "run.log")

class _Tee:
    """Write to both a file and the original stdout."""
    def __init__(self, file_path):
        self._file = open(file_path, "w", buffering=1, encoding="utf-8")
        assert sys.__stdout__ is not None
        self._stdout = sys.__stdout__

    def write(self, data):
        self._file.write(data)
        self._stdout.write(data.encode(self._stdout.encoding, errors="replace").decode(self._stdout.encoding))

    def flush(self):
        self._file.flush()
        self._stdout.flush()

    def close(self):
        self._file.close()

tee = _Tee(LOG_PATH)
sys.stdout = tee

# ── imports (after redirect so import noise also goes to the log) ─────────────
import funcs_for_letters as ffl
import funcs
import setup

# ── load pruned model ──────────────────────────────────────────────────────────
device = setup.get_device()
print(f"Device: {device}")

pruned_model = torch.load("pruned_model.pth", weights_only=False, map_location=device)
pruned_model.eval()
print(f"Architecture: {ffl._arch_str(pruned_model)}")

DIGIT_FC_PATH = "fc_digit_pretrained.pth"
if not os.path.exists(DIGIT_FC_PATH):
    raise FileNotFoundError(
        f"{DIGIT_FC_PATH} not found — run the FC digit pre-training cell in training.ipynb first."
    )

# ── run experiment ─────────────────────────────────────────────────────────────
se_results = ffl.run_sample_efficiency_experiment(
    pruned_model,
    #fracs=[0.01, 0.05, 0.1, 0.175, 0.25, 0.5, 0.75, 1.0],
    fracs=[0.25],
    n_seeds=5,
    n_epochs_half=50,
    digit_fc_path=DIGIT_FC_PATH,
    output_dir=LOG_DIR,
)

# ── save result dict so the notebook can reload it ────────────────────────────
out_path = os.path.join(LOG_DIR, "se_results.pt")
torch.save(se_results, out_path)
print(f"\nse_results saved to {out_path}")

# ── restore stdout ─────────────────────────────────────────────────────────────
sys.stdout = sys.__stdout__
tee.close()
print(f"Log written to {LOG_PATH}")
