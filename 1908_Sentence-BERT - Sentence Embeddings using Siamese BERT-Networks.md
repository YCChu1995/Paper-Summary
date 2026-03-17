# Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
> [1908.10084](https://arxiv.org/abs/1908.10084)<br>
> Sentence-BERT
<div align=center><img src="/figures/1908.10084.01.png" style="height: 200px; width: auto;"/> <img src="/figures/1908.10084.02.png" style="height: 200px; width: auto;"/></div>

## Summary 
1. Sentence-BERT is an `SOTA` (BERT-quality) `embedding` model w/o cross-encoder architecture.
2. Training on `NLI` produces universal embeddings.
3. Both `Pooling strategy` and `Concatenation model` are important to the performance.

## Tech Insights 
1. Get a useful sentence embedding model from fine-tuning a pre-trained BERT, which is fast (< 20 min).
   > Previous methods started the training from the beginning, a random initialization ...
2. They found that the distance metric among `Euclidean `, `negative Manhatten`, and `negative Euclidean` has roughly the same evaluation performance. [mentioned in section 4.0 in the paper]
3. `Pooling strategy`: MEAN > CLS > MAX
4. `Concatenation model`: $(u, v, \left| u-v \right|)$ ~ $(u, v, \left| u-v \right|, u \cdot v)$<br>
   But I think they both have good performance `w/o significant difference`, therefore, testing is required for implementation.

---

## Motivation 
1. BERT is too slow for large-scale sentence-pair regression tasks, including semantic similarity search.
   - The architecture of BERT, cross-encoder, is in $O(N^2)$.
     > Run BERT(query, doc) for all docs
   - Required architecture is in $O(N)$
     > encode(query) + encode(document) &rarr; vector search
2. Historical embedding era
   - Word Embeddings Era (2013-2017)
     > Sentence embeddings = mean(word vectors)
   - Sentence Encoders (2017-2018)
     > These improved performance but were limited.
   - BERT Revolution (2018)
     > BERT improves many areas, NLI, STS (Semantic Textual Similarity), and QA, but not yet retrieval.

## Model
### 1. Architecture
- BERT / RoBERTa + pooling
- Pooling strategies
  > BERT outputs token embeddings: [h1, h2, ..., hn] 
  1. CLS-token
     > embedding = h_cls
  2. MEAN-strategy (the mean of output vectors, BEST-stratege from [Ablation Study](#4-ablation-study)) 
     > embedding = mean(h1 ... hn)
  3. MAX-strategy (a max-over-time of output vectors)
     > embedding = max(h1 ... hn)
- Fine-tune method
  1. Siamese networks
  2. Triplet networks
### 2. Objective Functions
- Classification Objective Function<br>
  A `softmax classifier` on sentence embeddings ($V_{concat}$)
  > $n$ is the dimension of the sentence embeddings and $k$ is the number of labels.<br>
  > $u$ and $v$ are sentence embeddings.
  <div align=center>$$output = softmax(W_t \cdot V_{concat})$$</div>
  <div align=center>$$W_t \in \mathbb{R}^{3n \times k}, V_{concat} = (u, v, \left| u-v \right|)$$</div>
  <div align=center><img src="/figures/1908.10084.01.png" style="height: 200px; width: auto;"/></div>
- Regression Objective Function<br>
  The `cosine similarity` between the two sentence embeddings $u$ and $v$ w/ `mean-squared-error` loss.
  <div align=center><img src="/figures/1908.10084.02.png" style="height: 200px; width: auto;"/></div>
- Triplet Objective Function<br>
  `Triplet loss` tunes the network such that the distance between $a$ and $p$ is smaller than the distance between $a$ and $n$.
  > Given an `anchor sentence` $a$, a `positive sentence` $p$, and a `negative sentence` $n$.<br>
  > $s_a$, $s_n$, $s_p$ are sentence embeddings for $a$, $n$, $p$.<br>
  > Margin $\epsilon$ ensures that $s_p$ is at least $\epsilon$ closer to $s_a$ than $s_n$.<br>
  > Author run experiment w/ $\left|\left| \cdot \right|\right|_{2}$ and $\epsilon=1$
  <div align=center>$$loss = max(\left|\left| s_a - s_p \right|\right| - \left|\left| s_a - s_n \right|\right| + \epsilon, 0)$$</div>
### 3. Training Details
- Training data
  1. SNLI
  2. Multi-Genre NLI
- Hyperparameters
  1. 3-way softmax classifier objective function for one epoch
  2. Batch-size = 16
  3. Adam optimizer (lr = 2e-5)
  4. A linear learning rate warm-up over 10% of the training data
  5. The default pooling strategy is MEAN.
### 4. Ablation Study
- `Concatenation mode` matters much than `pooling strategy` (especially on NLI data).
- Pooling strategy: MEAN > CLS > MAX
  > Previous work, [InferSent](https://arxiv.org/abs/1705.02364), found that MAX overperforms MEAN pooling for the BiLSTM-layer of InferSent.
- Concatenation model: $(u, v, \left| u-v \right|)$ > $(u, v, \left| u-v \right|, u \cdot v)$
  >  Previous works, [InferSent](https://arxiv.org/abs/1705.02364) and [Universal Sentence Encoder](https://arxiv.org/abs/1803.11175), use $(u, v, \left| u-v \right|, u \cdot v)$ instead.
  <div align=center><img src="/figures/1908.10084.T6.png" style="height: 200px; width: auto;"/></div>
