# ModernBERT

ModernBERT was released at the end of 2024 and it was a major upgrade to BERT (2018). Here, I just save some of my notes derived from a presentation I did on ModernBERT in January 2025.

### BERT recap

BERT (Bidirectional Encoder Representations from Transformers) is an encoder-only model. Such models are among the most widely used in Natural Language Processing (NLP), as they are well suited for a variety of classification and other language udnerstanding tasks.

BERT was a major leap when released and was SOTA for 11+ NLP benchmarks. However it's now outdated: hardware has gotten better, datasets are larger, and new techniques are available.

For instance, BERT was trained on 3.3B words: 2.5B words from Wikipedia and 800M from Google's BooksCorpus. 

The original BERT model had to variants. BERT-base which had 110M parameters and was trained on 4 TPUs for 4 days, and BERT-large which had 340M parameters and was trained on 16 TPUs for 4 days.

BERT was pre-trained on two tasks: Masked Language Modeling (MLM) and Next Sentence Prediction (NSP). MLM required the model to fill-in the blank while in NSP BERT had to decide if a given sentence is the next one or not.
90% of examples where 128 tokens long, and 10% filled the entire context window of 512 tokens.

 ### Difference between ModernBERT and BERT

The team that developed ModernBERT used advancements on a number of areas to improve performance compared to BERT.

First, the architecture. They removed bias terms to decrease complexity, used RoPE instead of absloute PE, applied pre-Norm instead of post-Norm, replaced GeLU with GeGLU, alternated global attention with local attention every third layer, removed padding, used BPE as in OLMO and used flash attention.

Second, they design experiments with a model-aware approach, minding tensor core requirements, tile quantization, and wave quantization.

Third, the only trained on MLM, no NSP. Upped mased tokens from 15% to 30%, and trained on 1.7T tokens at 1,024 token length, 250B at max context (8,192), and annealing on 50B tokens from higher quality sources.

### Results



### References
- https://arxiv.org/abs/2412.13663
- https://huggingface.co/blog/modernbert
- https://huggingface.co/blog/bert-101
- https://arxiv.org/abs/1810.04805
