# Report Structure

## Suggested Title
*"Structured Pruning for Interpretable Digit Recognition: Emergent Class Specialisation and Transfer to Letter Classification"*

---

## Abstract (~150 words)
- RQ1 in one sentence
- RQ2 in one sentence
- Approach: structured iterative pruning + NMF clustering
- Key quantitative results (cluster alignment score, transfer AUC vs. FC baseline)
- One-sentence conclusion

---

## 1. Introduction (~0.5 page)

**Throughline:** broad (opacity + compression tension) → gap (pruning literature ignores semantic structure) → two RQs → preview of findings.

- **Hook:** Neural networks are opaque; compression research trades accuracy for efficiency; interpretability research trades performance for explainability — can structured pruning achieve both?
- **Gap:** Prior work (magnitude pruning, lottery ticket) treats sparsity as a means to efficiency. Whether pruned subnetworks develop semantic structure is underexplored.
- **RQ1:** Does iterative structured pruning on a digit classifier produce neuron clusters that specialise for individual digit classes?
- **RQ2:** Does such a specialised sparse structure confer advantages in sample efficiency or learning speed when transferred to letter recognition?
- **Preview:** State yes/partial/no for both RQs with one supporting number each.

---

## 2. Related Work (~0.5–0.75 page)

Three paragraphs, each closing with how it connects to this work:

1. **Neural network pruning** — magnitude-based pruning (Han et al., 2015), Lottery Ticket Hypothesis (Frankle & Carlin, 2019), structured vs. unstructured pruning, iterative pruning-retraining cycles.
2. **Interpretability and modularity** — NMF for part-based representations (Lee & Seung, 1999), modular networks, neuron specialisation in vision models, superposition hypothesis (Elhage et al., 2022).
3. **Transfer learning** — fine-tuning and frozen features (Yosinski et al., 2014), transfer from compressed models, sample efficiency as a proxy for transfer quality.

---

## 3. Methods (~1.5–2 pages)

All described mathematically — no library names, no Python.

### 3.1 Dataset
- EMNIST Digits: 240k train / 40k test, 10 classes
- EMNIST Letters: 26 classes
- Preprocessing: orientation correction, normalisation to [0, 1]
- Train / validation / fresh-validation / test split sizes

### 3.2 Model Architecture
- MLP: 784 → [128, 128, 128, 64] → 10
- ELU activation
- Sparse connection masks: forward pass as element-wise product of weight matrix and binary mask (write the formula)
- Loss function and optimiser (described abstractly)

### 3.3 Phase 1 — Blind Pruning
- Neuron importance score ("downstream blend"): weighted combination of activation variance and outgoing weight magnitude — **write the formula**
- Global-budget neuron pruning: percentage per round, minimum layer width constraint
- Connection pruning: weight-magnitude threshold, percentage per round
- Retraining schedule: epochs per round

### 3.4 Phase 2 — Cluster-Guided Pruning
- Trigger: network falls below neuron / connection count thresholds
- Structural adjacency matrix construction
- NMF-based soft cluster discovery; acceptance criterion: within-cluster correlation > cross-cluster correlation
- Gradual cross-cluster connection cutting: percentage per round
- Error-driven regrowth: identify underperforming clusters, spawn neurons with constrained incoming/outgoing connections — **write the constraint**

### 3.5 Finalisation
- All cross-cluster connections severed simultaneously
- 50-epoch retraining to develop functional specialisation
- Acceptance gate: digit-alignment > 0.15

### 3.6 Transfer Experiment Variants

| Variant | Structure | Initialisation | Hidden weights |
|---|---|---|---|
| Pruned transfer | pruned | pretrained | frozen |
| Pruned unfrozen | pruned | pretrained | trainable |
| FC baseline | full | fresh | trainable |
| Random frozen | pruned topology | random | frozen |
| … | … | … | … |

### 3.7 Sample Efficiency Experiment
- Letter training fractions: [1%, 5%, 10%, 17.5%, 25%, 50%, 75%, 100%]
- Each fraction: train, evaluate final test accuracy

### 3.8 Evaluation Metrics
Defined mathematically:
- **Digit alignment**: for each cluster, fraction of total activation attributable to its dominant class
- **Selectivity score**: normalised entropy over class-conditional mean activations
- **AUC**: area under accuracy-vs-epoch learning curve
- **Milestone epoch**: first epoch reaching 70% / 80% accuracy

---

## 4. Results (~2–2.5 pages including figures)

### 4.1 Compression and Accuracy
- Table: original vs. final neuron count per layer, parameter count, test accuracy
- Demonstrates dramatic compression with accuracy retention

### 4.2 Cluster Interpretability — RQ1
- **Figure 1:** Bar chart of digit-alignment per cluster, with chance-level baseline
- **Figure 2:** Prototype images (mean of top-10% activating images) — one row per cluster
- Selectivity scores: mean ± std over clusters
- Ablation: per-class accuracy drop when each cluster is zeroed (heatmap or table)

### 4.3 Transfer Learning — RQ2
- **Figure 3:** Learning curves (accuracy vs. epoch) for all variants, mean ± std over 5 seeds
- Table: AUC and milestone epoch (mean ± std) per variant
- Note whether the gap between pruned-transfer and FC-baseline is statistically meaningful

### 4.4 Sample Efficiency
- **Figure 4:** Accuracy vs. training fraction for pruned-frozen, FC-baseline, random-frozen
- Highlight any regime where the pruned model has an advantage

---

## 5. Discussion (~0.5–0.75 page)

- **Why clusters emerge:** pruning eliminates redundant cross-class pathways; retraining reinforces surviving class-specific ones — connect to Lottery Ticket intuition
- **Interpreting transfer results:** if pruned structure helps → sparse structure encodes reusable digit features; if not → digit features are too specific, or letter classes too distant
- **Limitations:** single architecture, narrow transfer domain (EMNIST digits → letters), cluster acceptance threshold is heuristic

---

## 6. Conclusion (~0.25 page)
- Restate RQ1 and RQ2 with one-sentence answers
- Broader implication: structured pruning as a vehicle for interpretability, not just compression
- One sentence on future directions

---

## References (~1 page)

Suggested targets (~10–15 papers):

| Paper | Relevance |
|---|---|
| Han et al. (2015) — "Learning both weights and connections" | Magnitude-based pruning |
| Frankle & Carlin (2019) — Lottery Ticket Hypothesis | Iterative pruning + sparse subnetworks |
| LeCun et al. (1990) — Optimal Brain Damage | Early pruning by curvature |
| Lee & Seung (1999) — NMF | Part-based representations |
| Hoefler et al. (2021) — Sparsity in DNNs survey | Structured pruning overview |
| Elhage et al. (2022) — Superposition hypothesis | Neuron polysemanticity / specialisation |
| Olah et al. (2020) — Zoom In: circuits | Neural interpretability |
| Yosinski et al. (2014) — How transferable are features? | Transfer learning |
| Pan & Yang (2010) — Transfer learning survey | Transfer learning background |
| Cohen et al. (2017) — EMNIST dataset | Dataset |
| Clevert et al. (2015) — ELU activation | Activation function |

---

## Appendix (not required reading for TA)
- Full hyperparameter table: all pruning percentages, learning rates, thresholds
- Additional activation-maximization visualizations
- Per-run (all 5 seeds) learning curves

---

## Page Budget

| Section | Est. pages |
|---|---|
| Abstract | 0.15 |
| Introduction | 0.5 |
| Related Work | 0.75 |
| Methods | 1.75 |
| Results (4 figures) | 2.5 |
| Discussion | 0.75 |
| Conclusion | 0.25 |
| References | 1.0 |
| **Total** | **~7.65** |
