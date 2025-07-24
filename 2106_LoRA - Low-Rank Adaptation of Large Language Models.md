# LoRA: Low-Rank Adaptation of Large Language Models
> [2106.09685](https://arxiv.org/abs/2106.09685)<br>
> LoRA
<div align=center><img src="/figures/2106.09685.1.png" style="height: 150px; width: auto;"/></div>

## Summary 
1. Where to attach LoRA modules?
    - `Best: ∆Wq + ∆Wk + ∆Wv + ∆Wo`
    - `Efficiency: ∆Wq + ∆Wv`
2. How many ranks perform better?
    - `Ranks 4~16 have peak performance in GPT2-medium for their task.`
3. How to set Amplification Facotr?
    - From the `equation in H.4`, for their experiments
        - `r = 4, Amplification Facotr ~ 20`
        - `r = 64, Amplification Facotr ~ 2`

## Tech Insights 
1. `∆W could have a very small “intrinsic rank”.`
2. `LoRA learns stable signals`
3. `∆W contains those “task-specific” directions that are otherwise not emphasized in W.`

---

## Motivation 
1. `Fine-tuning pretrained backbone is necessary to boost performance.` ([1810 / BERT](https://github.com/YCChu1995/Paper-Summary/blob/main/1810_BERT%20-%20Pre-training%20of%20Deep%20Bidirectional%20Transformers%20for%20Language%20Understanding.md))
2. `Most models have low intrinsic dimension.` ([1804 / Measuring the Intrinsic Dimension of Objective Landscapes](https://github.com/YCChu1995/Paper-Summary/blob/main/1804_Measuring%20the%20Intrinsic%20Dimension%20of%20Objective%20Landscapes.md))
3. Existing parameter-efficient methods (e.g. adapters, prefix-tuning) either decreased performance, limited sequence length, or introduced inference latency.

## Experiment
### 1. Normalized subspace similarity between **_ranks_**
- Top singular-vector directions of Ar=8 and Ar=64 are the most useful, while other directions potentially contain mostly random noises accumulated during training.<br>
  &rarr; `∆W could have a very small “intrinsic rank”.`
<div align=center><img src="/figures/2106.09685.2.png" style="height: 200px; width: auto;"/></div>

### 2. Normalized subspace similarity between **_runs_** ( with different random seeds )
- `LoRA learns stable signals`, not just stochastic noise.
<div align=center><img src="/figures/2106.09685.3.png" style="height: 200px; width: auto;"/></div>

### 3. Normalized subspace similarity between **pretrained W** & **fine-tuned ΔW**
- ∆W does not contain the top singular directions of W<br>
  &rarr; `∆W contains those “task-specific” directions that are otherwise not emphasized in W.`
<div align=center><img src="/figures/2106.09685.4.png" style="height: 200px; width: auto;"/></div>

### 4. Where to attach LoRA modules? & How many ranks perform better?
- Table 6 shows that LoRA already performs competitively with a very small r (more so for {Wq, Wv} than just Wq).<br>
  &rarr; `∆W could have a very small “intrinsic rank”.`
- From Table 5 & Table 6, where to attach?
    - `Best: ∆Wq + ∆Wk + ∆Wv + ∆Wo`
    - `Efficiency: ∆Wq + ∆Wv`
    - `Worst: Only ∆Wq or ∆Wk`
- From Table 6 & Table 18, how many ranks?
    - `Empirical evidence shows that ranks 4~16 have peak performance in GPT2-medium.`
    - Quote from H.2, `The relationship between model size and the optimal rank for adaptation is still an open question.`
  
<div align=center><img src="/figures/2106.09685.5.png" style="height: 200px; width: auto;"/></div>
<div align=center><img src="/figures/2106.09685.6.png" style="height: 150px; width: auto;"/></div>
<div align=center><img src="/figures/2106.09685.7.png" style="height: 150px; width: auto;"/></div>

### 5. How to set Amplification Facotr?
- From the `equation in H.4`, for their experiments
    - `r = 4, Amplification Facotr ~ 20`
    - `r = 64, Amplification Facotr ~ 2`
<div align=center><img src="/figures/2106.09685.8.png" style="height: 150px; width: auto;"/></div>
