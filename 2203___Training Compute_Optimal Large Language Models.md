# Training Compute-Optimal Large Language Models
> [2203.15556](https://arxiv.org/abs/2203.15556)<br>

<div align=center><img src="/figures/2203.15556.01.png" style="height: 150px; width: auto;"/></div>

## Summary 
1. 

## Tech Insights 
1. Overestimating the number of training step with setting cosine cycle length leads to performance drops.
<div align=center><img src="/figures/2203.15556.A1.png" style="height: 150px; width: auto;"/></div>

3. 

---

## Motivation 
The authors disagree with the larger model size trend; they believe that the model size and the number of training tokens should be scaled equally.
1. Prior work [(Kaplan et al., 2001.08361)](https://github.com/YCChu1995/Paper-Summary/blob/main/2001_Scaling%20Laws%20for%20Neural%20Language%20Models.md) suggests that one should increase 5.4x on N and 1.8x on D while C is increased on 10x, which leads to the larger model size trend.
2. Gathering extremely large high‐quality datasets (≥ 1 trillion tokens) seemed plausible but expensive, which also leads to the larger model size trend.
3. Many recent large LMs have grown primarily by increasing model size (e.g., GPT‑3 175B, Gopher 280B, Megatron‑Turing NLG 530B) while holding the number of training tokens roughly constant (~300 B tokens) despite increasing compute budgets..
4. Such undertrained over-parameterized models cost a lot during fine-tuning and inference.

## Chain of Thoughts

## Experiment
### 1.
### 2. 
- x
- y<br>
&rarr; y1 + y2 = y3
- z
