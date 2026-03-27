# Project Summary: Structured Neural Network Pruning & Transfer Learning

## Overview

This is a **structured neural network pruning research project** studying whether iterative pruning can create **interpretable, semantically specialized neuron clusters** in a neural network, and whether those clusters improve transfer learning efficiency.

**Core research questions:**
- Does iterative pruning on EMNIST digit classification produce neuron clusters that specialize for individual digit classes?
- Does that sparse, specialized structure confer advantages when transferred to letter recognition (especially with limited data)?

---

## Pipeline

### Phase 1 — Blind Pruning (`funcs.py`)
Iteratively removes neurons (2.5%/round) and connections (35%/round) using a "downstream blend" importance score (activation variance + outgoing weight magnitude). Dead/unreachable neurons are cleaned up each round. Stops when <100 neurons or <2000 connections remain.

### Phase 2 — Cluster-Guided Pruning
Discovers clusters via structural adjacency + NMF on activations. Gradually cuts cross-cluster connections (70%/round). Error-driven regrowth spawns new neurons for underperforming clusters.

### Finalization
Cuts ALL remaining cross-cluster connections, retrains 50 epochs to develop functional digit specialization. Gated by digit alignment score (>0.15 required to proceed).

### Transfer Experiment (`funcs_for_letters.py`)
6 variants trained on EMNIST lowercase letters (26 classes) across 5 seeds:

| Variant | Description |
|---------|-------------|
| `frozen_transfer` | Pruned model, digit layers frozen |
| `unfrozen_transfer` | Pruned model, all layers trainable |
| `fc_baseline` | Standard fully-connected network |
| `fc_digit_transfer` | FC pretrained on digits, then transferred |
| `random_frozen` | Randomly pruned frozen model |
| `random_frozen_regrowth` | Random frozen + regrowth |

---

## Metrics

| Metric | Description |
|--------|-------------|
| **Learning curves** | Accuracy vs. epoch for all 6 variants |
| **AUC** | Normalized area under the learning curve |
| **Final accuracy** | Last-epoch mean ± std across 5 seeds |
| **Milestone epochs** | First epoch reaching 70% / 80% accuracy |
| **Per-run curves** | Raw accuracy curves per seed (JSON) |
| **Digit alignment** | Fraction of activation attributable to dominant digit class per cluster (0–1) |
| **Cluster selectivity** | Normalized entropy of class-conditional activations |
| **Cluster criticality** | Per-class accuracy drop when ablating each cluster |

All metrics are computed at 8 data fractions for the sample efficiency experiment: `[1%, 5%, 10%, 17.5%, 25%, 50%, 75%, 100%]`.

---

## Graphs & Visualizations

| File | Type | Shows |
|------|------|-------|
| `learning_curves.html` | Plotly interactive | Mean ± std accuracy curves for all 6 variants over 100 epochs |
| `final_comparison.png` | Static PNG | Bar chart of final accuracy + AUC for all variants with error bars |
| `epoch_significance_fc_baseline.png` | Static PNG | t-test p-values comparing each variant vs. `fc_baseline` per epoch |
| `epoch_significance_fc_digit_transfer.png` | Static PNG | Same, vs. `fc_digit_transfer` baseline |
| `crossover_vs_fraction.png` | Static PNG | Accuracy curves across data fractions — shows where pruned model overtakes FC baseline |
| `cluster_ablation.html` | Plotly heatmap | Per-letter accuracy drop when each cluster is ablated |
| `layer0_all_neurons.html` | Plotly heatmap | 28×28 weight maps for all layer-0 neurons |
| `layer0_cluster_X.html` | Plotly heatmap | Layer-0 weight maps for neurons feeding into cluster X only |
| Network topology (via `network_map.py`) | Plotly node-link | Interactive graph: neurons as nodes, colored by cluster/component |
| Activation maximization (via `activation_maximization.py`) | PNG | Synthetic images that maximally activate each cluster |

---

## Results Directory Structure

```
results/
├── auc.txt/json
├── final_accuracy.txt/json
├── milestone_epochs.txt/json
├── learning_curves.html
├── final_comparison.png
├── epoch_significance_fc_baseline.png
├── epoch_significance_fc_digit_transfer.png
├── cluster_ablation.html
├── layer0_all_neurons.html
├── layer0_cluster_{2,4,5,6,7,8,9,10}.html
└── sample_efficiency/
    ├── crossover_vs_fraction.png
    ├── se_results.pt
    ├── run.log
    └── frac_X.XX/  (one per fraction: 0.01, 0.05, 0.1, 0.17, 0.25, 0.5, 0.75, 1.0)
        ├── summary.txt
        ├── auc.txt/json
        ├── final_accuracy.txt/json
        ├── test_accuracy.txt/json
        ├── milestone_epochs.txt/json
        ├── per_run_curves.json
        ├── learning_curves.html
        ├── cluster_ablation.html
        └── layer0_*.html
```

---

## Key Findings

- Pruned clusters achieve digit alignment ~0.51 (vs. 0.1 chance level) — structural specialization is real
- Frozen pruned models match FC baseline at full data (~87% vs. ~88%) but **outperform at low data fractions** (e.g., 74% vs. 71% at 10% data)
- Both pruning phases are necessary: topology compression creates structure, retraining creates functional specialization

---

## Main Scripts

| Script | Purpose |
|--------|---------|
| `NeuralNetwork.py` | Core MLP model class with training, importance scoring, and gradient masking |
| `setup.py` | Configuration parameters and EMNIST data loading |
| `funcs.py` | Full pruning pipeline and clustering logic (~1300 lines) |
| `analysis.py` | Post-hoc interpretability analysis (cluster criticality, selectivity, prototypes) |
| `plots.py` | Matplotlib and Plotly visualization utilities |
| `network_map.py` | Interactive network topology visualization |
| `activation_maximization.py` | Gradient-ascent visualization of cluster selectivity |
| `funcs_for_letters.py` | Transfer learning experiment framework (~1200 lines) |
| `run_sample_efficiency.py` | Runs sample efficiency experiments across 8 data fractions |
| `letters.ipynb` | Jupyter notebook for running and exploring experiments interactively |
