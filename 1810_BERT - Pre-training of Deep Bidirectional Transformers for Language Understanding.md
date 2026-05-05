# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
> [1810.04805](https://arxiv.org/abs/1810.04805)<br>
> BERT
<div align=center><img src="/figures/1810.04805.01.png" style="height: 250px; width: auto;"/> <img src="/figures/1810.04805.02.png" style="height: 250px; width: auto;"/></div>

## Summary 
1. **Architecture**: `Bidirectional Encoder` (only the encoder from Transformer)
    - [**Encoder Backbone**](#1-encoder-backbone) : `Bidirectional Encoder` (only the encoder from Transformer)
    - [**Input Embeddings**](#2-input-embeddings) : `Token Embeddings` + `Segment Embeddings` + `Absolute Trainable Position Embeddings`

2. [**Pretraining Tasks**](#3-pretraining-tasks--objectives): Both MLM and NSP losses are optimized simultaneously.
    - `MLM` to capturing `token-level` understanding 
    - `NSP` to capturing `sentence-level` understanding 

## Tech Insights 
1. `MLM converges more slowly but surpasses LTR almost immediately.`
2. Follow-up researches show that `NSP is not efficient for Pretraining Tasks`.
3. Quote from paper in section 4.1, "Additionally, for BERTLARGE we found that `finetuning was sometimes unstable on small datasets`,so we ran several random restarts and selected the  best model on the Dev set."
   
---

## Motivation 
1. No good bidirectional LM at the moment.<br>
   > Uni-directional LM (e.g. GPT) only consider past context; ELMo's shallow concatenation doesn't deeply integrate bidirectional information.
2. Prior architectures required task-specific modules, complicating fine-tuning and deployment.
   > Existing encodings failed to generalize across diverse downstream tasks with single fine-tuning pipelines.

## Architecture Details
### 1. Encoder Backbone
- `Bidirectional Encoder` (only the encoder from the Transformer)
- Each encoder layer contains
    - A multi-head self-attention sublayer (heads = hidden_size / 64).
    - A position-wise feed-forward network of dimension 4 × hidden_size.
    - Residual connections and LayerNorm, following either post‑LN or pre‑LN conventions (original BERT uses post‑LN)
 
### 2. Input embeddings
- `Token embeddings` (from vocabulary Lookup)
- `Segment embeddings` (indicating sentence tag, A or B)
- `Position embeddings` (`absolute positional embeddings, a trainable embedding matrix` (e.g. 512 positions × 768 dims for BERT‑Base), learned during pre-training)
<div align=center><img src="/figures/1810.04805.03.png" style="height: 150px; width: auto;"/></div>
<div align=center><img src="/figures/1810.04805.04.png" style="height: 300px; width: auto;"/></div>

### 3. Pretraining Tasks & Objectives
- Joint Training<br>
  `Both MLM and NSP losses are optimized simultaneously`, summing their cross-entropy objectives for combined backprop.<br>
  This dual objective is critical for capturing both `token-level and sentence-level understanding`.
- `MLM` (Masked Language Modeling)<br>
    - Purpose
      Capturing `token-level` understanding.
    - Task<br>
      15% of tokens are randomly selected; among those:
        - 80% replaced with [MASK]
        - 10% with a random token
        - 10% left unchanged
    - Objective<br>
      Use `cross‑entropy loss to predict original tokens at masked positions`.
- `NSP` (Next Sentence Prediction)
    - Purpose
      Capturing `sentence-level` understanding.
    - Task<br>
      For each sentence pair (A, B), 50% of the time B is the actual next sentence (“IsNext”), and 50% is a random, unrelated sentence (“NotNext”).
    - Objective<br>
      Uses the [CLS] token embedding fed into a `binary classification` head (softmax(W * h_CLS)) to predict next‑sentence status, “IsNext” or “NotNext”.
      
### 4. Ablation for Different Masking Procedures
- `MLM converges more slowly but surpasses LTR almost immediately.`
<div align=center><img src="/figures/1810.04805.05.png" style="height: 300px; width: auto;"/></div>
