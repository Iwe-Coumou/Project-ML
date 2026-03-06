# Pruning Pipeline Flowchart

```
┌─────────────────────────────────────────────────────────────┐
│                         SETUP                               │
│  EMNIST dataset → train/val/test/fresh splits               │
│  Network: 784 → [64→64→64→32] → 47                         │
│  Baseline training → baseline_acc                           │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              pruning() INITIALIZATION                       │
│  diag_loader = 2000 random samples (fast activation proxy)  │
│  loader_to_use = train_loader (batch_size from setup.py)    │
│  in_phase2 = False                                          │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
         ┌────────────────────────┐
    ┌───►│    PRUNING ROUND N     │
    │    └────────────┬───────────┘
    │                 │
    │    ┌────────────▼───────────────────────────────────┐
    │    │ STEP 1 — Neuron Importance                     │
    │    │  get_layer_data(diag_loader)                   │
    │    │  compute_neuron_importance(downstream_blend):  │
    │    │    I = 0.7×(mean_act × W_next_col_L1)          │
    │    │      + 0.3×variance                            │
    │    └────────────┬───────────────────────────────────┘
    │                 │
    │    ┌────────────▼───────────────────────────────────┐
    │    │ STEP 2 — Prune Neurons (global budget)         │
    │    │  Rank ALL neurons across ALL layers together   │
    │    │  Remove bottom 10% globally                    │
    │    │  Update weights + bias + connection_masks      │
    │    │  (weak layers lose more, strong layers less)   │
    │    └────────────┬───────────────────────────────────┘
    │                 │
    │    ┌────────────▼───────────────────────────────────┐
    │    │ STEP 3 — Prune Connections                     │
    │    │  Skip pixel→layer_0 and output layer           │
    │    │  For each primitive layer:                     │
    │    │    importance = |W[j,i]| × mean_act[i]         │
    │    │    Remove bottom 25% of NON-ZERO connections   │
    │    │    (zeros never reselected — bug fix)          │
    │    │  Update connection_masks                       │
    │    └────────────┬───────────────────────────────────┘
    │                 │
    │    ┌────────────▼───────────────────────────────────┐
    │    │ STEP 4 — Remove Unreachable Neurons            │
    │    │  Forward pass: which neurons receive signal    │
    │    │               from layer_0?                    │
    │    │  Backward pass: which neurons connect          │
    │    │               to output?                       │
    │    │  Keep only neurons in BOTH sets                │
    │    │  Update weights + connection_masks             │
    │    └────────────┬───────────────────────────────────┘
    │                 │
    │    ┌────────────▼───────────────────────────────────┐
    │    │ STEP 5 — Phase 2 Check                         │
    │    │  neurons < 50 OR connections < 200?            │
    │    │       NO → skip clustering                     │
    │    │       YES → in_phase2 = True                   │
    │    │             get_layer_data(diag_loader)         │
    │    │             cluster_neurons_fabio() [NMF]       │
    │    │             if >1 cluster found:               │
    │    │               cut cross-cluster connections    │
    │    └────────────┬───────────────────────────────────┘
    │                 │
    │    ┌────────────▼───────────────────────────────────┐
    │    │ STEP 6 — Stopping Checks                       │
    │    │  min_width reached? → EXIT LOOP                │
    │    │  max_rounds reached? → EXIT LOOP               │
    │    │  accuracy drop > 20%? → EXIT LOOP              │
    │    └────────────┬───────────────────────────────────┘
    │                 │
    │    ┌────────────▼───────────────────────────────────┐
    │    │ STEP 7 — Retrain                               │
    │    │  For each batch:                               │
    │    │    zero_grad()                                 │
    │    │    loss.backward()                             │
    │    │    zero gradients of pruned connections        │
    │    │      (gradient masking via connection_masks)   │
    │    │    optimizer.step()                            │
    │    │  Early stopping: patience=3                    │
    │    └────────────┬───────────────────────────────────┘
    │                 │
    └─────────────────┘  (next round)

                      │ EXIT LOOP
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                     _finalise()                             │
│                                                             │
│  1. get_layer_data → cluster_neurons_fabio() [NMF]         │
│     if >1 cluster: cut ALL cross-cluster connections        │
│                                                             │
│  2. remove_unreachable_neurons()                            │
│     (ghost neurons left after cluster isolation cut)        │
│                                                             │
│  3. Final retrain                                           │
│     patience=10, epochs=N_FINAL_RETRAIN_EPOCHS             │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
              final_model returned
```

## Key Design Decisions

| Component | Choice | Reason |
|-----------|--------|--------|
| Neuron importance | `downstream_blend` | Keeps neurons the next layer actually listens to, not just loud ones |
| Neuron pruning budget | Global across all layers | Lets weak layers shrink faster, creating asymmetric structure |
| Connection importance | `\|W\| × mean_act` | Prunes connections where the source neuron barely fires |
| Connection pruning pool | Non-zero only | Prevents budget being wasted on already-zeroed connections |
| Gradient masking | Before `optimizer.step()` | Prevents Adam accumulating momentum for dead connections |
| Activation collection | 2000-sample `diag_loader` | Avoids full dataset pass every round |
| Phase 2 trigger | neurons < 50 OR connections < 200 | Switch to structure-discovery mode once network is small enough |
