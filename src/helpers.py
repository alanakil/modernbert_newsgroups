# %%
import torch


# %%
def identify_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    return device


# %%
def tokenize_newsgroups(example, tokenizer):
    # Tokenize the 'text' field and apply truncation.
    return tokenizer(example['text'], truncation=True)


# %%
# Define a function to compute the token length for each example
def compute_length(example, tokenizer):
    # Tokenize the 'text' field without truncation
    tokens = tokenizer.tokenize(example['text'])
    # Store the token count in a new field 'length'
    example['length'] = len(tokens)
    return example


# %%
def tokenize_and_trim(example, tokenizer, max_length=200):
    # Tokenize the 'text' field with truncation enabled and a specified max_length.
    # The returned dictionary will include fields like 'input_ids' and 'attention_mask'.
    return tokenizer(example['text'], truncation=True, max_length=max_length)


# %%