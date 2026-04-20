# ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction
> [2112.01488](https://arxiv.org/abs/2112.01488)<br>
> ColBERTv2
<div align=center><img src="/figures/2112.01488.01.png" style="height: 150px; width: auto;"/></div>

## Summary 
1. ColBERTv2 improves `performance` by training with `hard negatives`.
2. ColBERTv2 improves the `retrieval quality` and `space footprint` of multi-vector models via `residual representation` and `quantization`.
3. Due to the embedding vector storing architecture, a `two stage retrieval system` is need.
4. A new evaluation benchmark, LoTTE

## Tech Insights 
1. Denoising (hard negatives) training boosts performance.
2. ColBERT Vectors cluster into `semantically pure regions`. ([evidence](#2-1-colbert-vectors-cluster-into-semantically-pure-regions))
   > Vectors corresponding to each sense of a word `cluster closely`, with only `minor variation` due to context.<br>
   
   &rarr; `Residual Embedding Representation` is a memory efficiency method for these semantically-pure-clustered embedding vectors.
3. ColBERTv2 quantizes the residual in only 1 or 2 bits.<br>
   &rarr; Regarding cosine similarity, directions matter more than values.
4. Centroids Building
   - How many centroids do we need?<br>
     <div align=center>$$\left| C \right| \propto \sqrt{n_{embeddings}}$$</div>
     
     > $\left| C \right|$ is the number of clusters.<br>
     > $n_{embeddings} \approx N \times T$ where the corpus has $N$ passages and each passage has ~ $T$ tokens.
   - How large is the corpus subset to estimate centroids?<br>
     <div align=center>$$size_{subset} \approx \sqrt{size_{corpus}}$$</div>
     
     > Since `the embedding space has high redundancy`, estimating clusters does NOT require full dataset.<br>
     > &rarr; We can `learn the global codebook` by `estimating centroids` with only `a subset of the corpus` ($size_{subset} \approx \sqrt{size_{corpus}}$).
5. Two stage retrieval system
   - Since ColBERTv2 embedding vectors are stored in cluster `without grouping by document ID`, `two stage retrieval system` is required.
     - Stage 1: `Retrieve` top- $n_{candidate}$ `candidate documents` by `approximate MaxSim score` among top- $n_{probe}$ `candidate centroids`.
     - Stage 2: `Re-rank` top- $n_{candidate}$ `candidate documents` by `approximate MaxSim score` among `all passage embedding vectors` corresponding to top- $n_{candidate}$ candidate documents.
   - $$MaxSim_{approx,stage_1} \leq MaxSim_{approx,stage_2} \leq MaxSim_{true}$$
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
  2. ColBERTv2:<br>
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
#### 2-2. Residual Representation for Embedding Vectors
- Residual Representation<br>
  Given a set of embedding centroids $C$, ColBERTv2 encodes each vector $v$ as the `index` $t$ of its `closest centroid` $C_t$ and a `quantized vector` $\tilde{r}$.
  <div align=center>$$\tilde{v} = C_t + \tilde{r}$$</div>
  
  > $C = \\{ C_1, C_2, ..., C_{\left| C \right|} \\}$<br>
  > $t \in  \\{ 1, 2, ...,  \left| C \right| \\}$
- Memory footprint per embedding vector
  <div align=center>$$\left \lceil log |C| \right \rceil + bn$$</div>

  > centroid index: $\left \lceil log |C| \right \rceil$<br>
  > residual: $bn$ ($b$-bit encoding to a $n$-dimensional embedding vecotr)
  - ColBERTv1: 128 dims x 16 bits = 2048 bits = `256 bytes`
  - ColBERTv2: centroid ~32 bits
    1. b=1 &rarr; 160 bits = `20 bytes`
    2. b=2 &rarr; 288 bits = `36 bytes`
- Residual quantization<br>
  To encode $\tilde{r}$, we quantize every dimension of $r$ into `1 or 2 bits`.

### 3. Inference
#### 3.1 Indexing
- **Stage 1: Centroid Selection & Estimation**
  - Centroid Selection<br>
    The number of centroids: $\left| C \right| \propto \sqrt{n_{embeddings}}$
    > Empirical results suggest setting: $\left| C \right| \propto \sqrt{n_{embeddings}}$<br>
    > $\left| C \right|$ is the number of clusters.<br>
    > $n_{embeddings} \approx N \times T$ where the corpus has $N$ passages and each passage has ~ $T$ tokens.
  - Centroid Estimation<br>
    To reduce memory comsumption, one could `learn a global codebook (centroids) first, then encode everything with it`.<br>
    Since `the embedding space has high redundancy`, estimating clusters do NOT require full dataset.<br>
    &rarr; We can `learn the global codebook` by `estimating centroids` with only `a subset of the corpus` ($size_{subset} \approx \sqrt{size_{corpus}}$).
    - Step 1: Computing and storing embeddings from `a subset of the corpus`.
    - Step 2: Estimate centroids via `k-means clustering on subset embeddings`.
- **Stage 2: Passage Encoding** (w/ estimated centroids)
  - Computing embeddings of the entire corpus, but only storing `the index` to the nearest centroid and `the quantized residual` for each passage.
  - Steps:
    1. Compute embedding vectors for every tokens.
    2. Assign each token embedding to its nearest centroid.
    3. Compute residual.
    4. Quantize + store compressed version.
    5. Discard raw embeddings immediately.
- **Stage 3: Index Inversion**<br>
  To support fast nearest-neighbor search, we group the embedding IDs that correspond to each centroid together, and save this inverted list to disk.<br>
  At search time, this allows us to quickly find token-level embeddings similar to those in a query.
#### 3.2 Retrieval
- Idea
  - Since ColBERTv2 embedding vectors are stored in cluster `without grouping by document ID`, `two stage retrieval system` is required.
    - Stage 1: `Retrieve` top- $n_{candidate}$ `candidate documents` among top- $n_{probe}$ `candidate centroids` by approximate MaxSim score.
    - Stage 2: `Re-rank` top- $n_{candidate}$ `candidate documents` by approximate MaxSim score with all passage embedding vectors corresponding to top- $n_{candidate}$ candidate documents.
  
- Steps:
  - Stage 0: Setup
    Given a `query representation` $Q = \\{Q_1, Q_2, ..., Q_m \\}$ and `searching hyperparameters` $n_{probe}, n_{candidate} \geq 1$. 
  - Stage 1: `Candidate Document Retrieval` (among tokens in `candidate centroids`)
    1. Found top-$n_{probe}$ nearest centroids as `candidate centroids` for every query vector $Q_i$.
    2. Identify `document embeddings` that belong to the candidate centroids (via the inverted list from **Stage 3: Index Inversion** during indexing).
    3. `Decompress` candidate document embeddings.
    4. Compute the `cosine similarity` $sim(d_j, Q_k)$ between `decompressed document embeddings` $d_j$ and `every query vector` $Q_k, \forall k$.
    5. The cosine similarities are then `grouped by document ID for each query vector`, and scores `corresponding to the same document` are `max-reduced`.
    6. Calculate `approximate MaxSim score` by summing across the query tokens.
       > The `Approximate MaxSim score` is a `lower-bound` on the `true MaxSim`.<br>
       > Because the **approximate MaxSim score** is calculated only on a `subset` (candidate document embeddings) of the corpus.
    7. Get `retrieved candidate documents`, top-$n_{candidate}$ documents with the highest approximate MaxSim score.
  - Stage 2: `Document Re-ranking` (among tokens in `retrieved candidate documents`)
    > Since the `Approximate MaxSim score` in Stage 1 is a `lower-bound` on the `true MaxSim`, further re-ranking can approach to the `true MaxSim`.<br>
    > $$MaxSim_{approx,stage_1} \leq MaxSim_{approx,stage_2} \leq MaxSim_{true}$$
    1. Fetch `candidate passages` for re-ranking, which are all embeddings that belong to the `retrieved candidate documents`.
    2. `Re-ranking` candidate documents by calculating the MaxSim score.
       > The MaxSim score here is still an approximation, since it is still on a subset of the corpus.

### 4. LoTTE: Long-Tail, Cross-Domain Retrieval Evaluation
- A cross-domain retrieval benchmark built from `StackExchange` that evaluates models on `long-tail`, `real-world queries`, less knowledge-density under domain shift.
