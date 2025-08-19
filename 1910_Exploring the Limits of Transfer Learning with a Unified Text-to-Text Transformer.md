# Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer
> [1910.10683](https://arxiv.org/abs/1910.10683)<br>
> T5
<div align=center><img src="/figures/1910.10683.01.png" style="height: 150px; width: auto;"/></div>

## Summary 
1. This paper proposes to solve all different tasks as a single seq-2-seq task.
   > Input: "summarize: \<article>" → Output: "\<summary>"
2. Architecture: `Transformer Variant` (encoder–decoder)<br>
   Improvement from Transformer :
   - Layernorm placed before residual
     1. `Simplified` (no bias)
     2. `Pre-LayerNorm` instead of Post-LayerNorm ([2002.04745](https://arxiv.org/abs/2002.04745))
        > `x + Sublayer(LayerNorm(x))` &larr; `LayerNorm(x + Sublayer(x))`<br>
   - Dropout on
     1. Feed-forward network
     2. Skip connection
     3. Attention weights
     4. The input and output of the entire attention stack.
   - `Relative position encoding` (Different among attention heads / Shared across all layers)
     > Implemented as learned scalar logits shared across layers (32 relative position embeddings covering exponentially increasing offset ranges up to 128).
   - Attention uses multi-head (12 heads in base)
   - Standard feed-forward (d_ff = 3072 in base).
3. Vocabulary size = 32k wordpieces
   > English + small amounts of German/French/Romanian
4. Dataset : Colossal Clean Crawled Corpus (C4)
   > ~750GB filtered, hundreds of GBs of English

## Tech Insights 
1. `MTL works, but how data are mixed should be tuned as a hyperparameter.`
2. Scaling `Model Size` and `Pretraining Steps`, not `Ensembling`.
3. `Architectural` and `methodological` refinements are as important as `scaling`. 

---

## Motivation 
There are different structures for different tasks.
> encoder-only BERT-style masked-language pre-training; decoder-only autoregressive GPT-style; encoder-decoder seq2seq

The field lacked a standard experimental bed to compare choices.

## Experiment
### 1. Model Architectures
- `Parameter sharing between encoder and decoder` is a powerful compression technique
  > Enabling smaller models with near-top performance.
- `Depth matters`
  > Shrinking layer depth drops downstream effectiveness.
- `Encoder–decoder` beats `decoder-only` architectures. (even with modifications like prefix LM)
- Pre-training Objectives: `Denoising (Span-Corruption) Over Language Modeling`
  >  - Original Text: "Thank you for inviting me to your party last week."<br>
  >  - Denoising:<br>
  >    - Input: "Thank you <extra_id_0> me to your party <extra_id_1> week."<br>
  >    - Output: "<extra_id_0> for inviting <extra_id_1> last <extra_id_2>"<br>
  >  - LM:<br>
  >    - Input: "Thank you for inviting me to"<br>
  >    - Output: " your party last week."
<div align=center><img src="/figures/1910.10683.02.png" style="height: 250px; width: auto;"/></div>

### 2. Training Strategy
#### 2-1. Pre-training Unsupervised Objectives
<div align=center><img src="/figures/1910.10683.03.png" style="height: 115px; width: auto;"/></div>

- Combination Picked: `BERT-style` + `Replace Spans` + `Corruption Rate: 15%` + `Avg Corrupted Span Length: 3`
<div align=center><img src="/figures/1910.10683.04.png" style="height: 115px; width: auto;"/></div>

<table>
  <tr>
    <td width="50%">
      <h4> 2-1-1. High-level Objective </h4>
      <p><b>BERT-style</b> beats <b>Prefix LM</b> beats <strong>Deshuffling</strong>.</p>
      <div align=center><img src="/figures/1910.10683.05.png" style="height: 100px; width: auto;"/></div>
    </td>
    <td width="50%">
      <h4> 2-1-2. BERT-style Varients </h4>
      <p><b>Replace Spans</b> slightly improves general performance.</p>
      <div align=center><img src="/figures/1910.10683.06.png" style="height: 100px; width: auto;"/></div> 
    </td>
  </tr>
  <tr>
    <td width="50%">
      <h4> 2-1-3. Corruption Rates </h4>
      <p><b>15%</b> performs the best.</p>
      <div align=center><img src="/figures/1910.10683.07.png" style="height: 100px; width: auto;"/></div>
    </td>
    <td width="50%">
      <h4> 2-1-4. Spans Lengths </h4>
      <p><b>3</b> performs the best.</p>
      <div align=center><img src="/figures/1910.10683.08.png" style="height: 100px; width: auto;"/></div> 
    </td>
  </tr>
</table>

#### 2-2. Pre-training Datasets
##### 2-2-1. Different Corpora
- Cleaning matters, `heuristic filtering improves downstream transfer performance`.
- Domain-specific data can outperform broader data, but with trade-off between domain alignment and data scale/diversity.
<div align=center><img src="/figures/1910.10683.09.png" style="height: 110px; width: auto;"/></div> 

##### 2-2-2. Dataset Size & Repetition
- `Repetition isn’t enough`, because of overfitting.
  > As the dataset size shrinks (requiring more repetition), downstream performance drops across all benchmarks.
<div align=center><img src="/figures/1910.10683.10.png" style="height: 150px; width: auto;"/> <img src="/figures/1910.10683.11.png" style="height: 150px; width: auto;"/></div> 

#### 2-3. Training Strategy
> `Multi-Task Pretraining` + `Fine-Tuning` achieves comparable performance to the standard baseline (`Pretraining` + `Fine-Tuning`).

##### 2-3-1. Fine-Tuning Methods
- `Full fine-tuning remains the most robust.`
- `Adapter Layers` should be in `proper dimension`, but still come with a `performance gap` compared to full fine-tuning.
- `Gradual unfreezing is almost as effective as full fine-tuning.`
<div align=center><img src="/figures/1910.10683.12.png" style="height: 120px; width: auto;"/></div>

##### 2-3-2. Multi-Task Learning (MTL)
- `MTL works, but only if you carefully balance` how examples from different tasks are sampled during training.
- `Temperature-scaled mixing (T=2) outperforms` naive proportional mixing.
<div align=center><img src="/figures/1910.10683.13.png" style="height: 200px; width: auto;"/></div>

##### 2-3-3. Combining MTL with Fine-Tuning
- `Multi-task pretraining` + `fine-tuning` achieves `comparable performance` to the standard baseline (pretraining + fine-tuning per downstream task)
- Leave-one-out multi-task doesn't significantly degrade performance, indicating `positive transfer among tasks`.
- `Pretraining is crucial for downstream performance.` (Because relying solely on supervised multi-task training degrades performance).
- Although certain well-populated tasks like `translation remain relatively robust`, suggesting less dependency on massive unlabeled pretraining for those tasks.
<div align=center><img src="/figures/1910.10683.14.png" style="height: 100px; width: auto;"/></div>

### 3. Scaling
- Where to put more budget? `Model Size`, `Training Duration`, or `Ensembling`.
    1. `Model Size` and  `Pretraining Steps`, not `Ensembling`
    2. `Architectural` and `methodological` refinements are as important as `scaling`. 
<div align=center><img src="/figures/1910.10683.15.png" style="height: 150px; width: auto;"/></div>
