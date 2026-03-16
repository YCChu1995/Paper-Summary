# Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
> [1908.10084](https://arxiv.org/abs/1908.10084)<br>
> Sentence-BERT
<div align=center><img src="/figures/1908.10084.01.png" style="height: 150px; width: auto;"/></div>

## Model
### 1. Architecture
- BERT / RoBERTa + pooling
- Pooling strategies
  1. CLS-token
  2. MEAN-strategy (the mean of output vectors)
  3. MAX-strategy (a max-over-time of output vectors)
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
  <div align=center><img src="/figures/1908.10084.01.png" style="height: 150px; width: auto;"/></div>
- Regression Objective Function<br>
  The `cosine similarity` between the two sentence embeddings $u$ and $v$ w/ `mean-squared-error` loss.
  <div align=center><img src="/figures/1908.10084.02.png" style="height: 150px; width: auto;"/></div>
- Triplet Objective Function<br>
  `Triplet loss` tunes the network such that the distance between $a$ and $p$ is smaller than the distance between $a$ and $n$.
  > Given an `anchor sentence` $a$, a `positive sentence` $p$, and a `negative sentence` $n$.<br>
  > $s_a$, $s_n$, $s_p$ are sentence embeddings for $a$, $n$, $p$.<br>
  > Margin $\epsilon$ ensures that $s_p$ is at least $\epsilon$ closer to $s_a$ than $s_n$.<br>
  > Author run experiment w/ $\left|\left| \cdot \right|\right|_{2}$ and $\epsilon=1$
  <div align=center>$$loss = max(\left|\left| s_a - s_p \right|\right| - \left|\left| s_a - s_n \right|\right| + \epsilon, 0)$$</div>
  
### 3. Training Details

## Summary 
1. Sentence-BERT is an `SOTA` (BERT-quality) `embedding` model w/o cross-encoder architecture.

## Tech Insights 
1. Get a useful sentence embedding model from fine-tuning a pre-trained BERT, which is fast (< 20 min).
   > Previous methods started the training from the beginning, a random initialization ...
2. 

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

## Chain of Thoughts

## Experiment
### 1.
### 2. 
- x
- y<br>
&rarr; y1 + y2 = y3
- z
