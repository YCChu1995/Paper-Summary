# ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction
> [2112.01488](https://arxiv.org/abs/2112.01488)<br>
> ColBERTv2
<div align=center><img src="/figures/2112.01488.01.png" style="height: 150px; width: auto;"/></div>

## Improvements
### 1. Training Schema
- Methods
  1. ColBERT:<br>
     MS MARCO official triples $<q, d^+, d^−>$.
     > For each query $q$, a positive $d^+$ is human-annotated, and each negative $d^−$ is sampled from unannotated BM25-retrieved passages.
  2. ColBERT v2:<br>
     
- Weakness in the ColBERT method 
  1. False Negatives
     > BM25 retrieves top-k passages &rarr; Treated as negatives &rarr; But many of them are actually relevant, just not labeled.
  2. Weak / Easy Negatives (low training signal)
     > BM25-based supervision produces `unchallenging negatives`.
  3. Bias Toward BM25 (teacher bias problem)

## Summary 
1.  ColBERTv2 improves the `retrieval quality` of multi-vector models while reducing their `space footprint`.

## Tech Insights 
1. 

---

## Motivation 
- Previous work, ColBERT, has `higher accuracy` than single-vector DPR does, but comes with `more memory cost`.<br>
  &rarr; ColBERT tries to `reduce the memory footprint` while keeping `token-level interaction` and `retrieval quality`.

## Chain of Thoughts

## Experiment
### 1.
### 2. 
- x
- y<br>
&rarr; y1 + y2 = y3
- z
