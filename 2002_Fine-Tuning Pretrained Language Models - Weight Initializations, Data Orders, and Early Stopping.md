# Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping
> [2002.06305](https://arxiv.org/abs/2002.06305)<br>

## Summary 
1. Confirm fine-tuning instability in BERT across different `weight initialization (WI) in the classification layer` and `training data ordering (DO), determined by shuffling`, affects validation performance.
2. Propose a solution (Early Stopping Strategy) to overcome the instability as in [No.1 in Tech Insights](#tech-insights).

## Tech Insights 
1. When facing unstable fine-tuning, `TRY MORE, STOP EARLY, CONTINUE SOME`
    - **Try more**: Start multiple training trails (with same hyperparameters and different WI and DO)
    - **Stop early**: Stop the trial if validation loss does not improve for t steps
    - **Continue some**: Continue training with top-k performance trials
2. `More granular validation yields a better chance to spot optimal checkpoints and improve selection.`
3. `Validation performance early in training is highly correlated with performance late in training.` (as shown in the [figure](#3-performance-correlation--early--late-val-accuracy-) at top)

---

## Motivation 
At the moment, `fine‑tuning BERT is inconsistent`: large performance variance arises solely from random seeds, even under the same settings (same hyperparameters)

## Experiment
### 1. Expected Validation Performance 
- `More granular validation (e.g., 10×/epoch) yields a better chance to spot optimal checkpoints and improve selection.`
<div align=center><img src="/figures/2002.06305.1.png" style="height: 300px; width: auto;"/></div>

### 2. WI & DO
- Both WI and DO equally contribute to the instability.
- Statistically significant differences between best vs worst random seeds are confirmed via ANOVA tests (p < 0.05).<br>
  &rarr; `Fine-tuning instability isn’t random “noise”.`<br>
  &rarr; `It systematically arises from both WI and DO.`
<div align=center><img src="/figures/2002.06305.2.png" style="height: 200px; width: auto;"/></div>
<div align=center><img src="/figures/2002.06305.3.png" style="height: 200px; width: auto;"/></div>

### 3. Performance Correlation [ Early & Late val-accuracy ]
- Best WI seeds perform well **across tasks**.<br>
  Best DO seeds perform well **on specific tasks**.
<div align=center><img src="/figures/2002.06305.4.png" style="height: 200px; width: auto;"/></div>
