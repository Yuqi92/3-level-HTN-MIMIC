# 3-level-HTN-MIMIC
Three-level Hierarchical Transformer Networks for Long-sequence and Multiple Clinical Documents Classification

We present a Three-level Hierarchical Transformer Network (3-level-HTN) for modeling long-term dependencies across clinical notes for the purpose of patient-level prediction. 

The network is equipped with three levels of Transformer-based encoders to learn progressively from words to sentences, sentences to notes, and finally notes to patients. 

The first level from word to sentence directly applies a pre-trained BERT model as a fully trainable component. 

While the second and third levels both implement a stack of transformer-based encoders, before the final patient representation is fed into a classification layer for clinical predictions. 

Compared to conventional BERT models, our model increases the maximum input length from 512 tokens to much longer sequences that are appropriate for modeling large numbers of clinical notes. 

We empirically examine different hyper-parameters to identify an optimal trade-off given computational resource limits. 

Our experiment results on the MIMIC-III dataset for different prediction tasks demonstrate that the proposed Hierarchical Transformer Network outperforms previous state-of-the-art models, including but not limited to BigBird.
## Citing

```
@article{si2021hierarchical,
  title={Hierarchical Transformer Networks for Longitudinal Clinical Document Classification},
  author={Si, Yuqi and Roberts, Kirk},
  journal={arXiv preprint arXiv:2104.08444},
  year={2021}
}
```

