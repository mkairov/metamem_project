# In-Context Learning in LLMs via Test-Time Training
## AIRI Summer School 2025

[project keynote](meta_memory_keynote.pdf)

[project report](meta_memory_report.pdf)

We propose a hybrid framework that integrates gradient-based meta-learning and test-time adaptation to enable efficient in-context memorization and question answering. A compact memory module is meta-trained to acquire efficient initializations and hyperparameters for the inner loop and then updated at inference via a small number of gradient steps on a self-supervised next-token objective. At query time, the adapted memory serves as a compressed representation of contextual information, enabling fact retrieval through an attention mechanism. Empirical evaluation on a synthetic associative retrieval benchmark demonstrates our model's ability to retrieve information acquired through meta-learning to answer questions based on the provided context while reducing computational overhead.
