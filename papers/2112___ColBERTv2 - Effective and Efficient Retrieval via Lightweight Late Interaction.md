# ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction
> [2112.01488](https://arxiv.org/abs/2112.01488)<br>
> ColBERTv2
<div align=center><img src="/figures/2112.01488.01.png" style="height: 150px; width: auto;"/></div>

### 2. Embedding Representation
#### 2-1. ColBERT Vectors cluster into semantically pure regions
- Hypothesis<br>
  ColBERT vectors `cluster into regions` that capture highly-specific token semantics.<br>
  > Vectors corresponding to each sense of a word `cluster closely`, with only `minor variation` due to context.
- Supporting evidence
  - Figure 2 (a)<br>
    Most clusters contain very `few distinct tokens` (~90% of clusters have ≤ 16 tokens)<br>
    &rarr; Each cluster represents a specific semantic region.
  - Figure 2 (b)<br>
    Most tokens `appear in very few clusters` (~50% of tokens appear in ≤ 16 clusters)<br>
    &rarr; A word does NOT spread everywhere in embedding space
  <div align=center><img src="/figures/2112.01488.02.png" style="height: 250px; width: auto;"/></div>
#### 2-2. Residual Representation for Embeddings
- Residual Representation<br>
  Given a set of embedding centroids $C$, ColBERTv2 encodes each vector $v$ as the `index` $t$ of its `closest centroid` $C_t$ and a `quantized vector` $\tilde{r}$.
  <div align=center>$$\tilde{v} = C_t + \tilde{r}$$</div>
  
  > $C = \\{ C_1, C_2, ..., C_{\left| C \right|} \\}$<br>
  > $t \in  \\{ 1, 2, ...,  \left| C \right| \\}$
- Memory footprint per embedding vector
  <div align=center>$$\left \lceil log |C| \right \rceil + bn$$</div>

  > centroid index: $\left \lceil log |C| \right \rceil$<br>
  > residual: $bn$ ($b$-bit encoding to a $n$-dimensional embedding vecotr)
  - ColBERT v1: 128 dims x 16 bits = 2048 bits = `256 bytes`
  - ColBERT v2: centroid ~32 bits
    1. b=1 &rarr; 160 bits = `20 bytes`
    2. b=2 &rarr; 288 bits = `36 bytes`
- Residual quantization<br>
  To encode $\tilde{r}$, we quantize every dimension of $r$ into one or two bits. 

## Summary 
1.  ColBERTv2 improves the `retrieval quality` of multi-vector models while reducing their `space footprint`.

## Tech Insights 
1. Denoising (hard negatives) training boosts performance.
2. ColBERT Vectors cluster into `semantically pure regions`. ([evidence](#2-1-colbert-vectors-cluster-into-semantically-pure-regions))
   > Vectors corresponding to each sense of a word `cluster closely`, with only `minor variation` due to context.<br>
   
   &rarr; `Residual Embedding Representation` is a memory efficiency method for this case.

---

## Motivation 
- Previous work, ColBERT, has `higher accuracy` than single-vector DPR does, but comes with `more memory cost`.<br>
  &rarr; ColBERT tries to `reduce the memory footprint` while keeping `token-level interaction` and `retrieval quality`.

## Improvements
### 1. Training
#### 1-1. Training Schema
- Methods
  1. ColBERT:<br>
     - MS MARCO official `triples` $<q, d^+, d^−>$.
       > For each query $q$, a positive $d^+$ is human-annotated, and each negative $d^−$ is sampled from unannotated BM25-retrieved passages.
     - Weakness of official `triples`: 
       1. `False Negatives`
          > BM25 retrieves top-k passages &rarr; Treated as negatives &rarr; But many of them are actually relevant, just not labeled.
       2. Weak / `Easy Negatives` (low training signal)
          > BM25-based supervision produces `unchallenging negatives`.
       3. Bias Toward BM25 (teacher bias problem)
  2. ColBERT v2:<br>
     - `w-ways tuples`: $<q, d^+_1, d^−_2, ..., d^−_w>$. + `in-batch negatives` per GPU
       > $d^+_1$: `ground-truth positive` (human labeled) or `top-ranked` (by cross-encoder)<br>
       > $d^−_i$: `lower-ranked` (by cross-encoder)
     - Generation steps:
       1. Retrieve top-k passages via ColBERT.
       2. Re-ranking top-k passages via MiniLM.
       3. Build the w-way tuple from the re-ranking result.
- Bonus trick
  - `In-batch negatives` per GPU<br>
    Treat positives of other in-batch samples as negatives.
#### 1-2. Training Objective
- Objective function<br>
  `KL-Divergence loss` with a `restricted scale` (to distill the cross-encoder’s scores).
- Why `restricted scale`? Because scales between T(MiniLM) and S(ColBERT) didn't align!<br>
  - ColBERT scores (`bounded`)
    - Cosine similarity per token $\in [-1,1]$
    - Sum over tokens → still bounded
  - MiniLM scores (`unbounded`)
    - Arbitrary unbounded scale
    - Could be logits, relevance scores, ...
