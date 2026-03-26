# ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT
> [2004.12832](https://arxiv.org/abs/2004.12832)<br>
> ColBERT
<div align=center><img src="/figures/2004.12832.01.png" style="height: 200px; width: auto;"/></div>

## Summary 
- Combining perks from both `representation-focused models` and `Interaction-based models` works.
  > `Contextualized Embeddings` + `Late Interaction` &rarr; `ColBERT`
- `ColBERT` works in both `retrieval` and `re-ranking`.
- There are many [tricks](#4-offline-indexing-computing--storing-document-embeddings) to improve indexing throughput.

## Tech Insights 
1. `Query augmentation` is intended to serve as a `soft query expansion` with a `differentiable mechanism` to amplify important query signals.
2. Data (documents) in a `batch sharing the same shape` (document length) yields more `uniform` and `compact` tensor shapes for the GPU, `improving GPU throughput`.
   > This is why in [4. Offine Indexing: Computing & Storing Document Embeddings](#4-offline-indexing-computing--storing-document-embeddings) they padded documents to the same length (max length).
3. Ablation studies
   - For single-vector re-ranking, `dot product` outperforms `cosine similarity`.
   - For similarity calculation over contextualized document embeddings, `Maximum-Similarity` outperforms `Average-Similarity`.
   - `Query Augmentation` boosts performance.
   - For document retrieving, retrieving with `ColBERT` outperforms `BM25`
4. Tricks for indexing optimization
   - Multi-GPU document processing
   - Per-batch maximum sequence length
   - Length-based bucketing
   - Multi-core pre-processing
5. The `most space-efficient setting` in Table 4 is `only slightly worse` in MRR@10 than the `most space-consuming one`.
   <div align=center><img src="/figures/2004.12832.T4.png" style="height: 150px; width: auto;"/></div>

---

## Motivation 
- There is a performance-cost gap, and this work filles it (as precise as BERT, but much cheaper than BERT).
  1. Traditional approaches, BM25, are NOT precise but cheap.
  2. Current methods, Duet, are a little more precise but a lot more expensive.
<div align=center><img src="/figures/2004.12832.01.png" style="height: 200px; width: auto;"/></div>

- Current methods
  1. **Representation-focused models** (Figure 2-a) & **interaction-based models** (Figure 2-b, 2-c)
     - **Representation-focused models** are possible to precompute document representations offline. (Faster)
     - **Interaction-based models** tend to be superior for IR tasks. (Better)
  2. This paper observes that `the fine-grained matching` of **interaction-based models** and `the precomputation of document representations` of **representation-based models** can be `combined` by retaining yet judiciously delaying the query–document interaction.
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
  > Projecting to a lower dimension helps query efficiency.
- Normalization (B, T, m) &rarr; (B, T, m)
  Each output embeddings are normalized to L2-norm equal to one
#### 2-2. Document Encoder
- $E_d \equiv Filter( Normalize( CNN( BERT( "[cls][D]d_0d_1...d_l" ) ) ) )$
- Pre-tokenizer<br>
  Same as the one in the query encoder, but without `Query augmentation`.
- Filter<br>
  Removing the embeddings corresponding to `punctuation symbols`, determined via a pre-defined list.<br>
  This filtering is meant to `reduce the number of embeddings` per document, as we hypothesize that (even contextualized) embeddings of punctuation are unnecessary for effectiveness.
### 3. Late Interaction
$$S_{q,d} \equiv \underset{i \in \left[ \left| E_q \right| \right]}{\sum} \underset{j \in \left[ \left| E_d \right| \right]}{max} E_{q_i} \cdot E_{d_j}^T$$
- For each query token:
  1. Find the most similar document token
  2. Sum these max similarities
- $S_{q,d}$ is the relevance score of document $d$ to query $q$
- $E_q$, $E_d$ are `the bags of the contextualized embeddings` corresponding to $q$, $d$
### 4. Offline Indexing: Computing & Storing Document Embeddings
- Core concept
  1. `Cap` the sequence length on a per-batch basis to `improve GPU throughput`. (Step.4)
     > Data (documents) in a `batch sharing the same shape` (document length) yields more `uniform` and `compact` tensor shapes for the GPU, `improving GPU throughput`.
  2. To save computational waste from `over padding` when capping the sequence length, `batch` documents with comparable length. (Step.3)
  3. To help batch documents with comparable length, `sort` documents by length. (Step.2)
  4. To save sorting cost, `divide` the gigantic corpus into smaller chunks. (Step.1)
- Steps:
  1. `Divide` the gigantic corpus into smaller big document chunks of $B$ documents (e.g., $B$ = 100,000) to save sorting cost in the following step.
  2. `Sort` documents in each document chunk by length.
  3. `Batch` $b$ documents (e.g., $b$ = 128) with comparable length.
  4. `Cap` the sequence length on a per-batch basis.
  5. `Storing` document embeddings after ColBERT.
<div align=center><img src="/figures/2004.12832.06.png" style="height: 150px; width: auto;"/></div>

## Inference
### 1. Top-k Re-ranking w/ ColBERT
- Steps:
  1. `Load` the indexed documents' representations into memory, representing each document as a matrix of embeddings.
  2. `Gather` the document representations into a `3-dimensional tensor` $D$ ($k$, $T_d^{max}$, $m$) consisting of k document matrices.<br>
     `Pad` the k documents to their `maximum length` to facilitate batched operations.<br>
     `Move` the tensor D to the GPU’s memory.
  3. `Compute` the bag of contextualized embeddings to the `query`, $E_q$ ($T_q$, $m$). (Concurrently with step.2)
  4. `Compute` a `batch dot-product` of $E_q$ and $D$ on the GPU, possibly over multiple mini-batches.<br>
      The output 3-dimensional tensor ($k$, $T_q$, $T_d^{max}$) is a collection of `cross-match matrices` between `query` and `each document`.
  5. `Max` over document tokens. ($k$, $T_q$, $T_d^{max}$) &rarr; ($k$, $T_q$)
  6. `Sum` over query tokens. ($k$, $T_q$) &rarr; ($k$,)
  7. `Sort` the k documents by their total scores.
- Pseudo code:
  ```
  # Query & Document embeddings
  E_q = torch.randn(T_q, C) # (T_q, C)
  D = torch.randn(k, T_d_max, C) # (k, T_d_max, C)
  
  # Similarity (Step.4)
  sim_q_d = (D @ E_q.transpose(0,1)).transpose(1,2) # (k, T_q, T_d_max)
  
  # Max over document tokens (Step.5)  
  sim_q = sim_q_d.max(dim=2).values # (k, T_q)
  
  # Sum over query tokens (Step.6)
  scores = sim_q.sum(dim=1) # (k,)
  ```
- ColBERT is not only cheaper, but it also scales much better with $k$
  > ColBERT decouples encoding from interaction, so increasing $k$ only increases cheap similarity computations, not expensive Transformer passes.
  <div align=center><img src="/figures/2004.12832.04.png" style="height: 200px; width: auto;"/></div>
### 2. End-to-end Top-k Retrieval w/ ColBERT
- Focus on retrieving the top-k results directly from a `large document collection` with $N$ documents, where $k$ << $N$.
- Performance boost by utilizing `fast vector-similarity data structures` instead of applying MaxSin between one query embedding and all documents' embeddings.
- Inference (2-stage procedure):
  1. Approximate stage<br>
     - Issue $N_q$ contextualized query embeddings ($E_q$) onto FAISS index to `retrieve` the top - $k'$ (e.g., $k'=\frac{k}{2}$) matches over all contextualized document embeddings ($E_d$) each.<br>
       This produces $N_{ret} = N_q \times k'$ document IDs in total.
       > ANN search: $E_{q,i}$ &rarr; top - $k'$ $E_d,j$ in FAISS
     - `Filter` repeated document IDs. 
     - Keep raising $k'$ until $N_{ret, uniq} \geq k$.
  2. Refinement stage<br>
     `Re-ranking` over all $N_{ret, uniq}$ documents.
  3. FAISS Inference details
     - `IVFPQ` (inverted file with product quantization) index
     - Partitions the embedding space into `P cells` (e.g., $P$ = 1000) based on `k-means clustering` and then assigns each document embedding to its nearest cell based on the selected vector-similarity metric.
     - For serving queries, when searching for the top - $k'$ matches for a single query embedding, only the `nearest p partitions` (e.g., $p$ = 10) are searched.
     - `Product quantization` to improve memory efficiency, every embedding is divided into s (e.g., s = 16) sub-vectors, each represented using one byte.
## Ablation Studies
- For single-vector re-ranking, `dot product` ( Model [A] ) outperforms `cosine similarity`.
- For similarity calculation over contextualized document embeddings, `Maximum-Similarity` ( Model [D] ) outperforms `Average-Similarity`( Model [B] ).<br>
  These results suggest the importance of individual terms in the query, paying special attention to particular terms in the document.
- `Query Augmentation` ( Model [C] & [D] ) boosts performance.
- For document retrieving in the 2-stage (retrieval + re-ranking) pipeline, retrieving with `ColBERT` ( Model [F] ) outperforms `BM25`.
<div align=center><img src="/figures/2004.12832.05.png" style="height: 150px; width: auto;"/></div>
