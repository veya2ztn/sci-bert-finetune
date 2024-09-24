This is the embedding training code for the scientific embedding project: [[2405.11461] DocReLM: Mastering Document Retrieval with Language Model (arxiv.org)](https://arxiv.org/abs/2405.11461).

The dataset is from the Synthetic data build via [veya2ztn/Synthetic-Science: Those script try to create Synthetic Science QA answer-question pair efficiently and reasoning (github.com)](https://github.com/veya2ztn/Synthetic-Science) based on [veya2ztn/uparxive: llm-friendly dataest for the whole arxiv .tex source. (github.com)](https://github.com/veya2ztn/uparxive)

# llm_train

This repo integrate embedder training method

- ART
- SGPT
- Finetune
  - Pipline training
  - Tensor Parallel: 1D, 2D and so on
- Gradient Cache
- Qlora
- Uniem
