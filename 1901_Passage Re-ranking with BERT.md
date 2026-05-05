# Passage Re-ranking with BERT
> [1901.04085](https://arxiv.org/abs/1901.04085)<br>
<div align=center><img src="/figures/1901.04085.01.png" style="height: 150px; width: auto;"/></div>

## Summary 
1. Passage Re-ranking with BERT `outperforms` previous methods. ([source](#experimental-result))
2. Passage Re-ranking with BERT is `easy to fine-tune`. ([source](#experimental-result))

## Tech Insights 
1. To prevent testing data leakage, we should do pre-training without them.
   > The official pre-trained BERT models were `pre-trained` on the full Wikipedia.<br>
   > Wikipedia documents that are also used in the `test set` of TREC-CAR.<br>
   > Thus, to avoid this leak of test data into training, we pre-trained the BERT re-ranker only on half of Wikipedia used by TREC-CAR’s training set.

---

## Motivation 
1. Traditional search ranking relies heavily on `lexical matching`.
   > Fail example (lexical mismatching)<br>
   > Query: car engine repair<br>
   > Document: how to fix automobile motor
2. Previous re-ranking models require `handcrafted features`, and are `shallow`, `weak contextual understanding`, and `poor cross-token interaction`.
3. This research fine-tunes a pretrained LM (BERT) for passage re-ranking.

## Experimental Result
- Passage Re-ranking with BERT `outperforms` previous methods.
  <div align=center><img src="/figures/1901.04085.T1.png" style="height: 150px; width: auto;"/></div>

- Passage Re-ranking with BERT is `easy to fine-tune`.
  > 100k iterations * batch size of 128 query-passage pairs ~ `2%` of the full training set
  <div align=center><img src="/figures/1901.04085.01.png" style="height: 150px; width: auto;"/></div>
