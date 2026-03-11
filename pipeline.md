# Project-ML Pipeline — Complete Reference

> Full codebase analysis: all math, all formulas (LaTeX), all bugs, all obsolete code.
> LaTeX: `$$...$$` = display block, `$...$` = inline.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Data Loading](#2-data-loading)
3. [Model Architecture](#3-model-architecture)
4. [Training](#4-training)
5. [Neuron Importance Scoring](#5-neuron-importance-scoring)
6. [Phase 1 — Blind Pruning Loop](#6-phase-1--blind-pruning-loop)
7. [Phase 2 — Cluster-Guided Pruning](#7-phase-2--cluster-guided-pruning)
8. [Finalise](#8-finalise)
9. [Analysis Tools](#9-analysis-tools)
10. [Transfer to Letters — Experiment](#10-transfer-to-letters--experiment)
11. [Visualisation](#11-visualisation)
12. [Bug Report](#12-bug-report)
13. [Dead / Obsolete Code](#13-dead--obsolete-code)

---

## 1. System Overview

The pipeline trains an MLP on EMNIST digits, then iteratively compresses it via structured
pruning while discovering interpretable neuron clusters that specialise for individual digit
classes. The compressed model is then used as a frozen feature extractor for transfer
learning to EMNIST letters.

```
EMNIST Digits
     |
     v
 Initial MLP Training  [128 -> 128 -> 128 -> 64 -> 10]
     |
     v
 Phase 1 Pruning Loop  (blind: prune neurons + connections + regrow)
     |   triggers when total_neurons < 100 OR total_connections < 2000
     v
 Phase 2 Pruning Loop  (cluster-guided: cut cross-cluster connections + error-driven regrowth)
     |
     v
 _finalise             (re-cluster, alignment gate, isolation cut, final retrain 50 epochs)
     |
     v
 pruned_model.pth      (~30-40 neurons, functional cluster specialisation)
     |
     +---> analysis.ipynb   (ablation, selectivity, prototypes, network graphs)
     |
     +---> letters.ipynb    (6-variant transfer-learning experiment on EMNIST letters)
```

**Key files:**

| File | Role |
|------|------|
| `NeuralNetwork.py` | Model class, training, importance scoring |
| `funcs.py` | All pruning, clustering, regrowth logic |
| `setup.py` | Hyperparameters and digit data loading |
| `analysis.py` | Post-hoc analysis (ablation, prototypes, selectivity) |
| `plots.py` | All matplotlib/Plotly visualisations |
| `network_map.py` | Interactive network topology graphs |
| `activation_maximization.py` | Gradient-ascent cluster visualisation |
| `funcs_for_letters.py` | Transfer-learning experiment helpers |
| `training.ipynb` | Main training notebook |
| `analysis.ipynb` | Analysis notebook |
| `letters.ipynb` | Letters transfer-learning notebook |

---

## 2. Data Loading

### 2.1 EMNIST Digits (`setup.get_dataloaders`)

- Dataset: EMNIST `digits` split — 240 000 training samples, 40 000 test, 10 classes (0–9)
- Transforms applied to every image:

```
rotate 90 degrees clockwise  ->  flip horizontally
```

These two operations correct the EMNIST orientation so digit images appear right-side up,
matching the MNIST convention.

- Pixel values: $[0, 1]$ after `ToTensor` (background $\approx 0$, stroke $\approx 1$)
- Splits (from `TRAIN_VAL_SPLIT=0.8`, `TRAIN_FRESH_SPLIT=0.1`):
  - **train**: 80% of training set
  - **val**: 20% of training set (used every round for pruning decisions)
  - **fresh**: 10% carved from train (used for final evaluation)
  - **test**: original test set

### 2.2 EMNIST Letters (`funcs_for_letters.get_letters_dataloaders`)

- Dataset: EMNIST `byclass` split, filtered to **lowercase** a–z only
- EMNIST `byclass` class encoding:
  - 0–9: digits
  - 10–35: uppercase A–Z
  - 36–61: lowercase a–z
- Filter at DataLoader level via `Subset` (raw labels before transform):

```python
mask = (dataset.targets >= 36) & (dataset.targets <= 61)
Subset(dataset, mask.nonzero(as_tuple=True)[0])
```

- Label remapping: $y \leftarrow y - 36$, so labels become $0$–$25$
- Same rotate + flip transforms as digit dataset
- Splits: `val_frac=0.1` of filtered train held out as validation

---

## 3. Model Architecture

### 3.1 MLP Definition (`NeuralNetwork.__init__`)

```
Input: [B, 1, 28, 28]
  |
  Flatten -> [B, 784]
  |
  Linear(784, h_0) -> ELU
  Linear(h_0, h_1) -> ELU
  ...
  Linear(h_{L-2}, h_{L-1}) -> ELU
  |
  Linear(h_{L-1}, C)        <- output layer, no activation
  |
  logits: [B, C]
```

Default hidden sizes: `[128, 128, 128, 64]`, output $C = 10$ (digits) or $26$ (letters).

Activation function: **ELU** (Exponential Linear Unit):

$$\text{ELU}(x) = \begin{cases} x & x > 0 \\ \alpha(e^x - 1) & x \leq 0 \end{cases}$$

with $\alpha = 1$.

### 3.2 Sparse Connection Masks

Each hidden `Linear` layer $\ell$ has an associated binary mask
$M^{(\ell)} \in \{0,1\}^{h_\ell \times h_{\ell-1}}$
stored in `model.connection_masks[layer_stack_idx]`.

During training, after each backward pass:

$$\nabla W^{(\ell)} \leftarrow \nabla W^{(\ell)} \odot M^{(\ell)}$$

This ensures pruned connections (mask = 0) accumulate no gradient and remain zero.

### 3.3 Layer Naming Convention

Logical layer names (`layer_0`, `layer_1`, …) refer to **hidden** layers in forward order.
`layer_0` = the first linear layer (784 -> h_0), which is the **input projection** and is
**excluded from clustering** because its weights encode pixel patterns, not neuron-to-neuron
functional specialisation.

---

## 4. Training

### 4.1 Loss Function

Cross-entropy with L1 regularisation:

$$\mathcal{L} = -\frac{1}{B}\sum_{i=1}^{B} \log p_{y_i} + \lambda_1 \sum_{\theta \in \Theta} |\theta|$$

where $p_{y_i}$ is the softmax probability of the true class, $\lambda_1 = 10^{-5}$ by
default, and $\Theta$ includes all parameters (weights + biases of all layers).

### 4.2 Optimiser

Adam with default $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$.
Learning rate passed per call; default $\eta = 10^{-3}$.

### 4.3 Early Stopping

Validation accuracy is checked every `val_interval` epochs (default 5).
If improvement is less than `early_stop_delta = 0.001` for `patience = 3` consecutive
checks, training stops.

> **BUG (NeuralNetwork.py:106):** `epochs_no_improve += val_interval` should be `+= 1`.
> The counter advances by `val_interval` per check, making the patience counter
> `val_interval`-times too large. Functionally the stopping fires at the right check-count
> but the internal counter is inflated.

### 4.4 Gradient Masking (Connection Mask Enforcement)

After `loss.backward()` and before `optimizer.step()`:

```python
for layer_idx, mask in model.connection_masks.items():
    layer = model.layer_stack[layer_idx]
    if layer.weight.grad is not None:
        layer.weight.grad.data *= mask.to(layer.weight.grad.device)
```

Ensures pruned connections stay at zero throughout training.

---

## 5. Neuron Importance Scoring

### 5.1 Modes (`compute_neuron_importance`)

**`'var'` — Activation Variance**

$$I_i^{\text{var}} = \text{Var}_{x \sim \mathcal{D}}\bigl[\text{pre\_act}_i(x)\bigr]$$

Low variance = neuron fires the same for everything = safe to prune.

**`'weight'` — Weight Magnitude**

$$I_i^{\text{weight}} = \sum_j |W_{ij}| + |b_i|$$

Total L1 norm of incoming weights plus bias. Data-free.

**`'downstream_blend'` — Downstream Influence (used by pruning loop)**

$$I_i^{\text{ds}} = \beta \cdot \overline{|a_i|} \cdot \sum_j |W^{(\ell+1)}_{ji}| + (1 - \beta) \cdot \text{Var}[a_i]$$

where $\overline{|a_i|}$ is mean absolute post-activation, the second term sums all
outgoing weights from neuron $i$ into the next layer (downstream attention), and
$\beta = 0.7$ (hardcoded).

> **BUG (NeuralNetwork.py:287):** No guard for the last hidden layer — `linear_indices[k+1]`
> will `IndexError` if called on the output layer.
> **BUG (NeuralNetwork.py:285):** $\beta = 0.7$ is hardcoded; cannot be changed without
> modifying source.

**`'combined'` (default) — Blended Variance + Weight**

$$I_i^{\text{comb}} = \alpha \cdot I_i^{\text{var}} + (1 - \alpha) \cdot I_i^{\text{weight}}$$

Default $\alpha = 0.7$. Note: the pruning loop calls `downstream_blend`, not `combined`.

---

## 6. Phase 1 — Blind Pruning Loop

Runs until `total_neurons < PHASE2_MIN_NEURONS` (100) **or**
`total_connections < PHASE2_MIN_CONNECTIONS` (2000).

Each round:

```
1. Compute neuron importance  (type='downstream_blend')
2. Prune neurons              (global budget)
3. Remove dead neurons
4. Remove unreachable neurons
5. Prune connections
6. Retrain N_RETRAIN_EPOCHS=8 epochs
7. Check convergence / accuracy drop
```

### 6.1 Global Budget Neuron Pruning (`prune_hidden_neurons`)

All hidden layers compete in a single budget. Let $N$ = total hidden neurons, $r$ =
`prune_frac` = 0.025.

$$n_{\text{prune}} = \lfloor r \cdot N \rfloor$$

Find threshold $\tau$ such that exactly $n_{\text{prune}}$ neurons have $I_i < \tau$.
Remove those neurons: delete their rows from layer $\ell$ and corresponding columns from
layer $\ell+1$. Update connection masks accordingly.

Minimum-width guard: no hidden layer drops below `MIN_WIDTH = 25` neurons.

### 6.2 Dead Neuron Removal (`remove_dead_neurons`)

A neuron is **dead** if it has no incoming AND no outgoing active connections:

$$\text{dead}_i = \left(\sum_j |W^{(\ell)}_{ij}| = 0\right) \wedge \left(\sum_j |W^{(\ell+1)}_{ji}| = 0\right)$$

### 6.3 Unreachable Neuron Removal (`remove_unreachable_neurons`)

A neuron is **unreachable** if not on any complete path from input to output.

Forward reachability:

$$\text{fwd}_i^{(1)} = \left(\sum_j |W^{(0)}_{ij}| > 0\right)$$

$$\text{fwd}_i^{(\ell)} = \exists\, j : \text{fwd}_j^{(\ell-1)} \wedge |W^{(\ell-1)}_{ij}| > 0 \quad (\ell > 1)$$

Backward reachability (analogous, propagating from output layer back).

Keep only neurons with $\text{fwd}_i \wedge \text{bwd}_i$.

### 6.4 Connection Pruning (`_prune_connections_native`)

For each primitive layer pair, connection importance:

$$\text{imp}_{ji} = |W_{ji}| \cdot \overline{|a_i|}$$

where $\overline{|a_i|}$ is mean absolute post-activation of source neuron $i$.
Falls back to $|W_{ji}|$ if activations unavailable.

Among non-zero connections, prune the weakest fraction:

$$n_{\text{prune}}^{(\ell)} = \max\!\left(1,\; \lfloor r_c \cdot n_{\text{nonzero}}^{(\ell)} \rfloor\right)$$

with $r_c =$ `PRUNE_CON_FRAC` = 0.35.

> **BUG (funcs.py:401):** `torch.kthvalue(scores, n_prune)` throws `IndexError` if
> `n_prune > scores.numel()`. Fix: `n_prune = min(n_prune, scores.numel())`.

---

## 7. Phase 2 — Cluster-Guided Pruning

Triggers when `total_neurons < 100` OR `total_connections < 2000`.

Each round:

```
1. Cluster neurons          (structural graph + NMF)
2. Accept clustering if quality Q is sufficient
3. Cut cross-cluster connections  (frac=0.70 per round, gradually)
4. Retrain
5. Error-driven regrowth    (if regrow_frac > 0 and clusters found)
```

### 7.1 Clustering (`cluster_neurons_fabio`)

**Step 1 — Structural adjacency:**

Build undirected edge between neurons $i$ (layer $\ell$) and $j$ (layer $\ell+1$) if:

$$|W^{(\ell+1)}_{ji}| > \epsilon_w \quad (\epsilon_w = 10^{-10})$$

**Step 2 — Connected components:**

Union-find on the adjacency graph gives **structural clusters**: groups of neurons that
are topologically connected by any nonzero weight path.

**Step 3 — NMF on component-level activations:**

Let $A \in \mathbb{R}^{N \times C}_{\geq 0}$ be the component mean activation matrix,
shifted non-negative:

$$A \leftarrow A - \min_{\text{col}}(A)$$

Non-negative Matrix Factorisation:

$$A \approx W H, \quad W \in \mathbb{R}^{N \times k},\; H \in \mathbb{R}^{k \times C}$$

Assign component $c$ to NMF factor: $\hat{k}(c) = \arg\max_k H_{kc}$

**Step 4 — Component strength filtering:**

$$s_c = \|W_{:,k}\|_F \cdot \|H_{k,:}\|_F$$

Components with $s_c < \frac{\bar{s}}{2k}$ are merged with the dominant cluster.

**Step 5 — Validate:** $k_{\min} \leq k_{\text{found}} \leq k_{\max}$; raise `ValueError`
otherwise.

### 7.2 Cluster Quality Gate (`_cluster_quality`)

$$Q = \frac{\overline{|\rho_{\text{within}}|}}{\overline{|\rho_{\text{cross}}|}}$$

where $\rho_{ij}$ is the Pearson correlation between activation series of neurons $i$ and $j$.

$Q > 1$ means within-cluster neurons are more correlated than cross-cluster neurons.
New clustering accepted only if $Q$ exceeds threshold.

### 7.3 Cross-Cluster Connection Pruning (`prune_cross_cluster_connections`)

For each hidden layer pair, identify cross-cluster connections:

$$\mathcal{C}_{\times} = \{(j, i) : W_{ji} \neq 0 \;\wedge\; \text{cluster}(j) \neq \text{cluster}(i)\}$$

Sort by ascending $|W_{ji}|$. Zero the weakest fraction $r_{\times} = 0.70$ per round.

### 7.4 Cluster–Digit Alignment Score (`cluster_digit_alignment`)

Assign each sample to its dominant cluster:

$$\hat{c}(x) = \arg\max_c \; \frac{1}{|\mathcal{N}_c|} \sum_{i \in \mathcal{N}_c} a_i(x)$$

Count digit occurrences per cluster: $n_{c,d}$ = samples of digit $d$ assigned to cluster $c$.

Per-cluster alignment (1 minus normalised Shannon entropy):

$$\text{align}_c = 1 - \frac{H(p_c)}{\log D}, \quad p_{c,d} = \frac{n_{c,d}}{\sum_d n_{c,d}}, \quad D = 10$$

$$H(p) = -\sum_d p_d \log p_d$$

$\text{align}_c = 1$: cluster sees only one digit. $\text{align}_c = 0$: uniform over all digits.

Mean alignment: $\bar{\text{align}} = \frac{1}{|\mathcal{C}|} \sum_c \text{align}_c$

### 7.5 Error-Driven Regrowth (`error_driven_regrowth`)

**Ablation-based underperformance detection:**

Zero all neurons in cluster $c$ temporarily, measure accuracy drop:

$$\delta_c = \text{acc}_{\text{baseline}} - \text{acc}_{\text{zeroed }c}$$

Mean drop and growth threshold:

$$\bar{\delta} = \frac{1}{|\mathcal{C}|}\sum_c \delta_c, \qquad \tau_{\text{grow}} = \frac{\bar{\delta}}{\phi}$$

with $\phi =$ `threshold_frac` = 1.5. If $\delta_c < \tau_{\text{grow}}$, cluster $c$
is underperforming and receives $n_{\text{spawn}}$ new neurons.

**Weight initialisation:**

New neuron's weight for allowed connection $j$:

$$w_j \sim \text{Exp}(\bar{w}) \cdot \text{sign}(u), \quad u \sim \mathcal{U}(-1, 1)$$

where $\bar{w}$ = mean absolute value of existing non-zero weights. Apply sparsity cutoff:
$w_j \leftarrow 0$ if $|w_j| < 0.1 \cdot \max_j |w_j|$.

Incoming connections restricted to same-cluster neurons in the previous layer.
Outgoing connections restricted to same-cluster neurons in the next layer.

The new neuron's global index is appended to `cluster_map[c]` and its weight row is
concatenated to the end of the layer's weight matrix.

---

## 8. Finalise

`_finalise` runs once after the pruning loop terminates.

```
1. Re-cluster the compressed network (cluster_neurons_fabio)
2. Measure digit alignment
3. Alignment gate: if align < 0.15, SKIP isolation cut
4. If gate passes: cut ALL cross-cluster connections (all-at-once, not gradual)
5. Remove unreachable neurons
6. Final retrain: N_FINAL_RETRAIN_EPOCHS = 50 epochs
7. Save final_cluster_map, final_layer_mapping, final_alignment_score on model object
```

**Why the alignment gate matters:**

Before the final retrain, clusters exist only **structurally** (separate connected
components) but have $\bar{\text{align}} \approx 0.000$ — each cluster fires uniformly
for all digits. Cutting at this point destroys the network.

After the 50-epoch final retrain, $\bar{\text{align}} \approx 0.511$ — functional
specialisation has emerged within the fixed topology.

The gate `MIN_ALIGNMENT_FOR_CUT = 0.15` (hardcoded in `_finalise`) ensures the
isolation cut only happens when clusters already have some functional identity.

> **Key finding:** Pruning creates **structure** (topology). Retraining creates **function**
> (digit specialisation). The two phases are necessary and sequential.

---

## 9. Analysis Tools

### 9.1 Cluster Ablation (`cluster_criticality_per_class`)

For each cluster $c$ with neurons $\mathcal{N}_c$:

1. Record per-class accuracy: $\text{acc}^{\text{pre}}_d$ for all digit/letter classes $d$.
2. Zero all neuron weights in $\mathcal{N}_c$ (incoming row weights, outgoing column weights,
   and biases).
3. Record per-class accuracy: $\text{acc}^{\text{post}}_d$.
4. Accuracy drop: $\Delta_d^c = \text{acc}^{\text{pre}}_d - \text{acc}^{\text{post}}_d$
5. Restore weights.

The function is fully general: works for any number of classes (10 digits or 26 letters)
via `defaultdict`.

> **BUG (analysis.py:51-52):** Layer index computed as
> `linear_indices[int(layer_name.split('_')[1])]`. This works for the current architecture
> but only because logical layer index = linear layer position in `layer_stack`. Will break
> if any non-linear layers are inserted before the target layer.

### 9.2 Cluster Selectivity (`compute_cluster_selectivity`)

For each cluster $c$, total activation contributed by class $d$:

$$s_{c,d} = \sum_{x: y(x)=d} \frac{1}{|\mathcal{N}_c|}\sum_{i \in \mathcal{N}_c} a_i(x)$$

Normalised to probability: $p_{c,d} = s_{c,d} / \sum_d s_{c,d}$

Normalised entropy (0 = maximally selective, 1 = uniform):

$$e_c = \frac{H(p_c)}{\log D}$$

> **Dead code:** `class_means` dict is filled inside this function but never used in the
> entropy computation (only `class_totals` matters).

### 9.3 Prototype Images (`compute_prototypes_all_clusters`)

Cluster strength per sample:

$$s_c(x) = \frac{1}{|\mathcal{N}_c|}\sum_{i \in \mathcal{N}_c} a_i(x)$$

Top-$k$ selection: $k = \max(1, \lfloor f \cdot N \rfloor)$, $f = 0.1$

**Prototype** (mean of top-$k$ images, normalised to $[0,1]$):

$$\mathbf{P}_c = \frac{1}{k} \sum_{x \in \text{top-}k} \mathbf{x}$$

**Difference map** (prototype minus global mean, normalised):

$$\mathbf{D}_c = \mathbf{P}_c - \overline{\mathbf{X}}$$

> **BUG (analysis.py:170):** When `use_global_mean=False`, the code computes
> `cluster_mean = images.mean(dim=0)` which is the **global** mean of all images, not the
> mean of cluster-assigned images. The `use_global_mean=False` path is broken.

### 9.4 Class-Conditioned Prototype (`compute_cluster_class_prototypes`)

Extension conditioned on the cluster's most critical class from ablation.

For cluster $c$:

1. Top class: $d^* = \arg\max_d \Delta_d^c$
2. Filter to class-$d^*$ images: $\mathcal{X}_{d^*}$
3. Cluster strength on filtered set
4. Top-$k$ prototype: $\mathbf{P}_c^{d^*}$ (same formula as §9.3 but filtered)
5. Class diff (normalised): $\mathbf{D}_c^{d^*} = \mathbf{P}_c^{d^*} - \overline{\mathbf{X}_{d^*}}$

> **BUG (analysis.py:368):** `$\Delta_d^c$` can be negative (ablation improved accuracy).
> `max(drops)` then picks the least-negative entry as "most critical", which is wrong.
> Should filter to positive drops first.

### 9.5 Consensus Pixel Map (`compute_cluster_consensus_pixels`)

Hard-assign each sample to its dominant cluster:

$$\hat{c}(x) = \arg\max_c s_c(x)$$

Fraction of cluster-dominant images where pixel $(p,q)$ exceeds binarisation threshold
$\theta_b = 0.15$:

$$\text{cons}_c^{(p,q)} = \frac{1}{|\hat{c}^{-1}(c)|}\sum_{x: \hat{c}(x)=c} \mathbf{1}\!\left[x^{(p,q)} > \theta_b\right]$$

> **Issue:** Parameter `consensus_threshold` is declared but never applied to the output.

### 9.6 Digit Consensus (`compute_cluster_digit_consensus`)

Active digits for cluster $c$: $\mathcal{D}_c = \{d : w_{c,d} > 0.05\}$

Per-digit mean images: $\overline{\mathbf{X}}_d = \frac{1}{N_d}\sum_{x: y(x)=d} \mathbf{x}$

Consensus (pixels bright in ALL active digit classes):

$$\mathbf{\text{cons}}_c = \min_{d \in \mathcal{D}_c} \overline{\mathbf{X}}_d$$

---

## 10. Transfer to Letters — Experiment

### 10.1 Model Variants (`funcs_for_letters.py`)

| Variant | Hidden weights | Frozen? | Regrowth in half 2 |
|---------|---------------|---------|-------------------|
| `frozen_transfer` | Copied from pruned model | Yes | No |
| `frozen_regrowth` | Copied from pruned model | Yes | Yes |
| `unfrozen_transfer` | Copied from pruned model | No | No |
| `fc_baseline` | Random, fully connected (no masks) | No | No |
| `random_frozen` | $\mathcal{N}(0, \hat{\sigma})$, same sparsity | Yes | No |
| `random_frozen_regrowth` | $\mathcal{N}(0, \hat{\sigma})$, same sparsity | Yes | Yes |

For `random_frozen*`, empirical std from pruned model non-zero hidden weights:

$$\hat{\sigma} = \text{std}\!\left(\{w : w \neq 0,\; w \in W^{(\ell)}_{\text{hidden}}\}\right)$$

Layer-0 (input projection) is **always** Xavier-initialised fresh. Output head is **always**
random, never copied.

### 10.2 Training Protocol (`run_experiment`)

For each seed $s \in \{0, \ldots, n_{\text{seeds}}-1\}$:

1. Set `torch.manual_seed(s)`, `np.random.seed(s)`
2. Split letter training set in half (seed-derived random split)
3. **Half 1** ($n_{\text{epochs\_half}}$ epochs): all variants train on half-1, no regrowth
4. **Half 2** ($n_{\text{epochs\_half}}$ epochs):
   - Non-regrowth variants: continue on half-2
   - Regrowth variants: each epoch, train then call `error_driven_regrowth`; if neurons added,
     re-register partial-freeze hooks and recreate Adam
5. **Ablation** (regrowth variants only): per-letter accuracy drop per cluster

Validation accuracy recorded after every epoch → learning curves of length $2 \cdot n_{\text{epochs\_half}}$.

### 10.3 Partial Freeze After Regrowth (`_refreeze_after_regrowth`)

When regrowth extends hidden layer $\ell$ from $n_{\text{orig}}$ to $n_{\text{new}}$ rows:

1. Remove old freeze hooks
2. Enable `requires_grad_(True)` on full weight/bias
3. Register gradient hook that zeroes gradients of original rows:

$$\nabla W^{(\ell)}_{i,:} \leftarrow \begin{cases} 0 & i < n_{\text{orig}} \\ \nabla W^{(\ell)}_{i,:} & i \geq n_{\text{orig}} \end{cases}$$

### 10.4 Saved Outputs (`save_results`)

| File | Content | Formula |
|------|---------|---------|
| `learning_curves.html` | Mean ± std all 6 variants | $\bar{a}_e = \frac{1}{S}\sum_s a_{s,e}$, band = $\bar{a}_e \pm \sigma_e$ |
| `auc.json/txt` | Area under learning curve | $\text{AUC}_v \approx \frac{\text{trapz}(\bar{a}_v)}{T}$ |
| `milestone_epochs.json/txt` | First epoch $\geq 0.70$ and $\geq 0.80$ | $e^* = \min\{e : \bar{a}_e \geq \theta\}$ |
| `final_accuracy.json/txt` | Last-epoch mean ± std | $\bar{a}_T \pm \sigma_T$ |
| `per_run_curves.json` | All raw seed curves | — |
| `cluster_ablation.html` | Mean letter-accuracy drop per cluster | $\bar{\Delta}_c = \frac{1}{S}\sum_s \Delta_{c,s}$ |
| `summary.txt` | Human-readable block | — |
| `layer0_all_neurons.html` | All layer-0 weight heatmaps (28x28, RdBu) | — |
| `layer0_cluster_{id}.html` | Layer-0 neurons feeding cluster $c$ | $\{i : \sum_{j \in \mathcal{N}_c^{(1)}} |W^{(1)}_{ji}| > 0\}$ |

### 10.5 Intersection Image (`_intersection_image`)

For images from significant ablation classes, binarise then intersect:

1. Binarise: $b_x^{(p,q)} = \mathbf{1}\!\left[x^{(p,q)} > \theta_p\right]$, $\theta_p = 0.3$
2. Intersection: $\mathbf{I}^{(p,q)} = \min_{x \in \mathcal{S}} b_x^{(p,q)} \in \{0, 1\}$

Gives a binary image showing pixels that are consistently bright across ALL sampled images
of the significant classes — i.e., the shared stroke features that the cluster responds to.

---

## 11. Visualisation

### 11.1 Layer-0 Receptive Fields (`plot_layer0_receptive_fields`)

Each neuron $i$ in layer-0 has weight vector $\mathbf{w}_i \in \mathbb{R}^{784}$.
Reshaped to $28 \times 28$ with custom colormap:
- **Black** = negative weight (pixel suppresses neuron)
- **White** = zero (pixel has no effect)
- **Blue** = positive (pixel excites neuron)

Symmetric colour range: $[\!-v_{\max}, +v_{\max}]$, $v_{\max} = \max_{i,p} |w_{i,p}|$.

### 11.2 First-Layer Weight Maps (`plot_first_layer_weights`)

Same data as §11.1 but neurons sorted by:
- L2 norm: $\|\mathbf{w}_i\|_2$
- Activation variance: $\text{Var}_x[a_i(x)]$
- Coefficient of variation: $\text{CV}_i = \sigma_i / (\bar{a}_i + \epsilon)$

Uses `RdBu_r` colormap.

> **Inconsistency:** `plot_first_layer_weights` uses `RdBu_r` (red/blue);
> `plot_layer0_receptive_fields` uses custom black/white/blue. Same data, different conventions.

### 11.3 Network Topology (`network_map.py`)

Node positions: neuron $i$ in layer $\ell$ (containing $n_\ell$ neurons) placed at:

$$(x, y) = \left(\ell,\; \frac{i}{n_\ell - 1}\right)$$

Edge transparency: $\alpha_{ji} = 0.6 \cdot \frac{|W_{ji}|}{\max |W^{(\ell)}|}$

Two colouring modes:
- **Components**: union-find on nonzero weights, random colour per connected component
- **Clusters**: colour by `cluster_map` assignment; unassigned neurons in grey

### 11.4 Activation Maximisation (`activation_maximization.visualize_cluster`)

Optimise image $\mathbf{x}$ to maximise cluster $c$ activation:

$$\mathbf{x}^* = \arg\max_{\mathbf{x}} \left[\frac{1}{|\mathcal{N}_c|}\sum_{i \in \mathcal{N}_c} a_i(\mathbf{x}) - \lambda_{\text{TV}} \cdot \text{TV}(\mathbf{x})\right]$$

Total variation regulariser:

$$\text{TV}(\mathbf{x}) = \frac{1}{HW}\sum_{p,q}\bigl(|x_{p,q+1} - x_{p,q}| + |x_{p+1,q} - x_{p,q}|\bigr)$$

with $\lambda_{\text{TV}} = 0.05$, 500 gradient-ascent steps, Gaussian smoothing every 50 steps.

> **Note:** `layer_mapping` parameter declared but never used inside this function.

---

## 12. Bug Report

### Critical — Crashes or Silently Wrong Results

| # | File | Line | Description | Fix |
|---|------|------|-------------|-----|
| C1 | `NeuralNetwork.py` | 106 | `epochs_no_improve += val_interval` should be `+= 1`. Counter is `val_interval`-times larger than intended. | Change to `+= 1` |
| C2 | `NeuralNetwork.py` | 287 | `linear_indices[k+1]` in `downstream_blend` has no guard for the last hidden layer. Will `IndexError` when scoring the final hidden layer. | Add `if k + 1 < len(linear_indices):` guard |
| C3 | `funcs.py` | 401 | `torch.kthvalue(scores, n_prune)` throws `IndexError` if `n_prune > scores.numel()`. Can happen on very small layers. | `n_prune = min(n_prune, scores.numel())` |
| C4 | `analysis.py` | 170 | `use_global_mean=False` branch computes global mean of ALL images instead of mean of cluster-assigned images. Diff map is wrong for this mode. | Compute mean over `images[top_idx]` |
| C5 | `plots.py` | 154 | Variable `i` used after `for i, ...` loop as `range(i+1, ...)`. If `cluster_results` is empty, `i` is undefined → `NameError`. | Initialise `i = -1` before loop |

### Medium — Incorrect Behaviour or Latent Risk

| # | File | Line | Description | Fix |
|---|------|------|-------------|-----|
| M1 | `setup.py` | ~50 | `HP_SEARCH_GRID_STAGE2` contains `phase2_min_neurons=250`. Known to cause cascade collapse (see MEMORY.md). | Remove 250 from grid |
| M2 | `setup.py` | — | `TOPOLOGY_THRESHOLD = 0.15` defined but never referenced anywhere in codebase. | Remove or wire up |
| M3 | `funcs.py` | 1020 | `prev_cluster_map` only updated inside `except` block. If clustering succeeds normally, it may remain stale from a previous round. | Move update outside `except` |
| M4 | `funcs_for_letters.py` | 496 | `lr=1e-3` hardcoded in Adam creation inside `_train_one_epoch`, ignoring the function's `lr` parameter. | Pass `lr` argument through |
| M5 | `analysis.py` | 203 | `consensus_threshold` declared in `compute_cluster_consensus_pixels` signature but never applied. | Apply threshold to output |
| M6 | `network_map.py` | 317-318 | `legend_labels` dict built from `unique_clusters` only but `UNASSIGNED` is added to `sorted_cids`. Unassigned entry missing from legend. | Add `UNASSIGNED` to `legend_labels` |
| M7 | `analysis.py` | 368 | In `compute_cluster_class_prototypes`, `drops[d]` can be negative (ablation improved accuracy). `max(drops)` then picks least-negative, not most-critical. | Filter to `drops[d] > 0` first |
| M8 | `funcs_for_letters.py` | 64-65 | `.logical_and()` chained on tensors. Prefer `&` operator for clarity and safety. | Replace with `& ` |
| M9 | `funcs.py` | 1249 | `weight_threshold=1e-10` in adjacency construction is near-zero, over-connecting components. | Raise to at least `1e-4` and document |

### Low — Code Quality, Hardcoded Values

| # | File | Line | Description |
|---|------|------|-------------|
| L1 | `NeuralNetwork.py` | 285 | `beta=0.7` hardcoded in `downstream_blend`. Should be a parameter. |
| L2 | `funcs.py` | 724 | `epsilon=0.1` hardcoded in regrowth weight init. Should be a parameter. |
| L3 | `funcs.py` | 885 | `MIN_ALIGNMENT_FOR_CUT=0.15` hardcoded in `_finalise`. Not in `setup.py`. |
| L4 | `setup.py` | 13 | `N_TRAIN_EPOCHS=10` but comment says "overridden to 8 on CPU". Misleading. |
| L5 | `setup.py` | — | `BATCH_SIZE=8000` is 8% of training set per step. May cause unstable gradients. Consider 2048-4096. |
| L6 | `plots.py` | 59/95 | Two functions use different colormaps for identical data (layer-0 weights). |
| L7 | `network_map.py` | 76/268 | Component mode uses RNG seed 42; cluster mode uses seed 7. Inconsistent. |
| L8 | `network_map.py` | 413 | y-axis range `[-0.18, 1.08]` hardcoded in interactive graph. Breaks with many layers. |
| L9 | `activation_maximization.py` | 49 | Smoothing frequency (every 50 steps) hardcoded. Should be a parameter. |
| L10 | `funcs_for_letters.py` | 778-779 | Milestone thresholds 0.70 and 0.80 hardcoded in `save_results`. |
| L11 | `plots.py` | 180 | `plot_loss` docstring says "accuracy". Copy-paste error. |

---

## 13. Dead / Obsolete Code

| Function / Symbol | File | Lines | Why Obsolete |
|-------------------|------|-------|--------------|
| `compute_regrow_from_pruned` | `funcs.py` | 1080–1095 | Computes regrow counts per layer, but `regrow_hidden_neurons` is never called in the pruning loop. Error-driven regrowth replaced this mechanism entirely. |
| `compute_regrow_from_width` | `funcs.py` | 1097–1109 | Same reason. Never called. |
| `cluster_neurons` (NMF-only) | `funcs.py` | 1160–1227 | Superseded by `cluster_neurons_fabio` (structural + NMF hybrid). Never called in any notebook or function. |
| `split_clusters_by_layer` | `funcs.py` | 1308–1330 | Only useful for per-layer clustering mode, which is disabled. Never called. |
| `layer_mapping` parameter | `activation_maximization.visualize_cluster` | 6 | Declared in signature but never accessed inside the function body. |
| `class_means` dict | `analysis.compute_cluster_selectivity` | 94–104 | Accumulated in a loop but never read afterward. Only `class_totals` feeds the entropy computation. |
| `TOPOLOGY_THRESHOLD` | `setup.py` | — | Defined but not referenced in any `.py` or notebook file. |
| `REGROW_FRAC` | `setup.py` | — | Passed to `compute_regrow_from_pruned` / `compute_regrow_from_width` which are both dead. `error_driven_regrowth` ignores this value; it uses `threshold_frac` instead. |

---

*End of pipeline.md*
