# ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT
> [2004.12832](https://arxiv.org/abs/2004.12832)<br>
> ColBERT
<div align=center><img src="/figures/2004.12832.01.png" style="height: 200px; width: auto;"/></div>

### 5. Top-k Re-ranking w/ ColBERT
### 6. End-to-end Top-k Retrieval w/ ColBERT

### Section 4.4 whyquery augmentation is important to ColBERT
### Section 4.5 How to futher boost the indexing computation
## Summary 
1. 

## Tech Insights 
1. `Query augmentation` is intended to serve as a `soft query expansion` with `differentiable mechanism` to amplify important query signals.
2. Data (documents) in a `batch sharing same shape` (document length) yields more `uniform` and `compact` tensor shapes for the GPU, `improving GPU throughput`.
   > This is why in [4. Offine Indexing: Computing & Storing Document Embeddings](#4-offine-indexing-computing--storing-document-embeddings) they padded documents to the same length (max length).

---

## Motivation 
- There is a performance-cost gap, and this work filles it (as precise as BERT, but much cheaper than BERT).
  1. Traditional approaches, BM25, are NOT precise but cheap.
  2. Current methods, Duet, are a little more precise but a lot expensive.
<div align=center><img src="/figures/2004.12832.01.png" style="height: 200px; width: auto;"/></div>

- Current methods
  1. **Representation-focused models** (Figure 2-a) & **interaction-based models** (Figure 2-b, 2-c)
     - **Representation-focused models** are possible to precompute document representations offine. (Faster)
     - **Interaction-based models** tend to be superior for IR tasks. (Better)
  2. This paper observe that `the fine-grained matching` of **interaction-based models** and `the precomputation of document representations` of **representation-based models** can be `combined` by retaining yet judiciously delaying the query–document interaction.
<div align=center><img src="/figures/2004.12832.02.png" style="height: 200px; width: auto;"/></div>

## Model
### 1. Architecture
<div align=center><img src="/figures/2004.12832.03.png" style="height: 200px; width: auto;"/></div>

### 2. Query & Document Encoders
#### 2-1. Query Encoder
- $E_q \equiv Normalize( CNN( BERT( "[cls][Q]q_0q_1...q_l[mask][mask]...[mask]" ) ) )$
- Pre-tokenizer<br>
  Tokenization &rarr; Special token prepending &rarr; Sequence length matching
  1. Tokenization ($t_0t_1...t_l$ &rarr; $q_0q_1...q_l$)
     > BERT-based WordPiece `Tokenization`.
  2. Special token prepending ($q_0q_1...q_l$ &rarr; $[cls][Q]q_0q_1...q_l$)
     > Prepend the token `[Q]` to the query, placing this token right after BERT’s sequence start token `[CLS]`.
  3. Sequence length matching - `Query augmentation` (soft query expansion) ($[cls][Q]q_0q_1...q_l$ &rarr; $[cls][Q]q_0q_1...q_l[mask][mask]...[mask]$)
     > If the query has `fewer` than the pre-defined number $N_q$, `pad` it with BERT’s special `[mask]` tokens up to the length $N_q$.<br>
     > Otherwise, `truncate` it to the first $N_q$ tokens.
- BERT (B, T) &rarr; (B, T, C)<br>
  Passes the contextualized output representations through a linear layer with no activations.
- Linear layer w/o activation (B, T, C) &rarr; (B, T, m)<br>
  This layer serves to control the dimension of ColBERT’s embeddings, producing $m$-dimensional embeddings for the layer’s output size $m$.<br>
  As we discuss later in more detail, we typically fix $m$ to be `much smaller` than BERT’s output embedding dimension.
  > Projecting to lower dimension helps query efficiency.
- Normalization (B, T, m) &rarr; (B, T, m)
  Each output embeddings are normalized to L2-norm equal to one
#### 2-2. Document Encoder
- $E_d \equiv Filter( Normalize( CNN( BERT( "[cls][D]d_0d_1...d_l" ) ) ) )$
- Pre-tokenizer<br>
  Same as the one in query encoder, but without `Query augmentation`.
- Filter<br>
  Removing the embeddings corresponding to `punctuation symbols`, determined via a pre-defined list.<br>
  This filtering is meant to `reduce the number of embeddings` per document, as we hypothesize that (even contextualized) embeddings of punctuation are unnecessary for effectiveness.
### 3. Late Interaction
$$S_{q,d} \equiv \underset{i \in \left[ \left| E_q \right| \right]}{\sum} \underset{j \in \left[ \left| E_d \right| \right]}{max} E_{q_i} \cdot E_{d_j}^T$$
- For each query token:
  1. Find most similar document token
  2. Sum these max similarities
- $S_{q,d}$ is the relevance score of document $d$ to query $q$
- $E_q$, $E_d$ are the embeddings corresponds to $q$, $d$
### 4. Offine Indexing: Computing & Storing Document Embeddings
- Core concept
  1. `Cap` the sequence length on a per-batch basis to `improve GPU throughput`. (Step.4)
     > Data (documents) in a `batch sharing same shape` (document length) yields more `uniform` and `compact` tensor shapes for the GPU, `improving GPU throughput`.
  2. To save computational waste from `over padding` when capping the sequence length, `batch` documents with comparable length. (Step.3)
  3. To help batching documents with comparable length, `sort` documents by length. (Step.2)
  4. To save sorting cost, `divide` the gigantic corpus into smaller big document chunk. (Step.1)
- Steps
  1. `Divide` the gigantic corpus into smaller big document chunk of $B$ documents (e.g., $B$ = 100,000) to save sorting cost in the following step.
  2. `Sort` documents in each document chunk by length.
  3. `Batch` $b$ documents (e.g., $b$ = 128) with comparable length.
  4. `Cap` the sequence length on a per-batch basis.
  5. `Storing` document embeddings after ColBERT.
