# QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models
> [2309.14717](https://arxiv.org/abs/2309.14717)<br>
> QA-LoRA
<div align=center><img src="/figures/2309.14717.1.png" style="height: 200px; width: auto;"/></div>

## Summary 
Merge learned LoRA weights into quantized pretrained weights `without any quantization error` via the following two constraints.
1. For pretrained weights, `quantize L groups in each column`.
   > Elements in the same group share a pair of scaling and zero factors.
2. For LoRA weights, `separate rows of A into L groups as well`.
   > Making elements of AB in the grouped location sharing the same value, satisfies the quantization property as below.<br>
   > All possible values of elements of the LoRA matrix in the grouped location should form an arithmetic set with the common difference being the scaling factor from pretrained weights quantization.

## Tech Insights 
1. To `avoid quantization error` during merging learned LoRA weights into quantized pretrained weights, we need `All possible values of elements of the LoRA matrix in the grouped location should form an arithmetic set` with the common difference being the scaling factor from pretrained weights quantization".
2. Making elements of AB form an arithmetic set with a common difference is intractable in continuous and gradient-based optimization.<br>
   &rarr; Making elements of the LoRA matrix `share the same value`.
---

## Motivation 
Existing quantization-aware fine‑tuning (QAT) methods either fine-tune all parameters (resource-heavy) or retain adapter precision, failing to deliver both accuracy and inference/numeric efficiency simultaneously.

## Chain Of Thought
<div align=center><img src="/figures/2309.14717.2.png" style="height: 1400px; width: auto;"/></div>
