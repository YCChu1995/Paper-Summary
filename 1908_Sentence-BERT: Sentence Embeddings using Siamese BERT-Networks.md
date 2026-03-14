# Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
> [1908.10084](https://arxiv.org/abs/1908.10084)<br>
> Sentence-BERT
<div align=center><img src="/figures/1908.10084.01.png" style="height: 150px; width: auto;"/></div>

## Model

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
