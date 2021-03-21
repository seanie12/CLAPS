# Contrastive Learning with Adversarial Perturbations for Conditional Text Generation
This is the Pytorch implementation for the paper **Contrastive Learning with Adversarial Perturbations for Conditional Text Generation** (**ICLR 2021**): [[Paper]](https://openreview.net/forum?id=Wga_hrCa3P3)

## Abstract
<img align="middle" width="800" src="https://github.com/seanie12/CLAPS/blob/main/images/method_fig.png">

Recently, sequence-to-sequence (seq2seq) models with the Transformer architecture have achieved remarkable performance on various conditional text generation tasks, such as machine translation. However, most of them are trained with teacher forcing with the ground truth label given at each time step, without being exposed to incorrectly generated tokens during training, which hurts its generalization to unseen inputs, that is known as the "exposure bias" problem. In this work, we propose to solve the conditional text generation problem by contrasting positive pairs with negative pairs, such that the model is exposed to various valid or incorrect perturbations of the inputs, for improved generalization. However, training the model with naïve contrastive learning framework using random non-target sequences as negative examples is suboptimal, since they are easily distinguishable from the correct output, especially so with models pretrained with large text corpora. Also, generating positive examples requires domain-specific augmentation heuristics which may not generalize over diverse domains. To tackle this problem, we propose a principled method to generate positive and negative samples for contrastive learning of seq2seq models. Specifically, we generate negative examples by adding small perturbations to the input sequence to minimize its conditional likelihood, and positive examples by adding  large perturbations while enforcing it to have a high conditional likelihood. Such "hard'' positive and negative pairs generated using our method guides the model to better distinguish correct outputs from incorrect ones. We empirically show that our proposed method significantly improves the generalization of the seq2seq on three text generation tasks --- machine translation, text summarization, and question generation.

__Contribution of this work__
- To mitigate the exposure bias problem, we propose a contrastive learning framework for conditional sequence generation, which contrasts a positive pair of source and target sentence to
negative pairs in the latent embedding space, to expose the model to various valid or incorrect
outputs.
- To tackle the ineffectiveness of conventional approach for constructing negative and positive examples for contrastive learning, we propose a principled method to automatically generate negative and positive pairs, that are more difficult and allows to learn more meaningful representations.
- We show that our proposed method, CLAPS, significantly improves the performance of seq2seq
model on three different tasks: machine translation, text summarization, and question generation.




# Reference
To cite the code/data/paper, please use this BibTex
```bibtex
@inproceedings{
lee2021contrastive,
title={Contrastive  Learning  with Adversarial Perturbations for Conditional Text Generation},
author={Seanie Lee and Dong Bok Lee and Sung Ju Hwang},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=Wga_hrCa3P3}
}
```
