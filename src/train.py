# %%
# Write about this protject

# %%
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,  # Using AutoModelForSequenceClassification as our model class.
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed,
)
import matplotlib.pyplot as plt

# %%
# 1. Device Setup: Use MPS if available (Apple Silicon), otherwise CUDA or CPU.
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device")
else:
    device = torch.device("cpu")
    print("Using CPU device")

# Set seed for reproducibility
set_seed(422)

# %%
# 2. Model and Tokenizer Initialization
model_name = "answerdotai/ModernBERT-base"  # Or "answerdotai/ModernBERT-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# %%
# 3. Load Datasets
# 20 Newsgroups dataset (assumes the "SetFit/20_newsgroups" dataset is available)
newsgroups = load_dataset('SetFit/20_newsgroups')


# %%
# 4. Tokenization with Dynamic Padding (no fixed padding here)
def tokenize_newsgroups(example):
    # Tokenize the 'text' field and apply truncation.
    return tokenizer(example['text'], truncation=True)


# Define a function to compute the token length for each example
def compute_length(example):
    # Tokenize the 'text' field without truncation
    tokens = tokenizer.tokenize(example['text'])
    # Store the token count in a new field 'length'
    example['length'] = len(tokens)
    return example


# Use parallel processing (num_proc=4) to speed up tokenization if you have multiple cores.
newsgroups_encoded = newsgroups.map(tokenize_newsgroups, batched=True)
# multinli_encoded = multinli.map(tokenize_multinli, batched=True, num_proc=8)


newsgroups = newsgroups.map(compute_length)

# Extract the 'length' field from the dataset
lengths = newsgroups["train"]['length']


# Plot the distribution of token lengths
plt.figure(figsize=(10, 6))
plt.hist(lengths, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel("Token Count")
plt.ylabel("Frequency")
plt.title("Distribution of Training Example Lengths (Token Count) in 20 Newsgroups")
plt.show()


# %%
# Define a function to tokenize and trim examples to a maximum length (e.g., 256 tokens)
def tokenize_and_trim(example):
    # Tokenize the 'text' field with truncation enabled and a specified max_length.
    # The returned dictionary will include fields like 'input_ids' and 'attention_mask'.
    return tokenizer(example['text'], truncation=True, max_length=200)


# Apply the tokenize_and_trim function to both splits (train and test)
newsgroups_trimmed = newsgroups.map(tokenize_and_trim, batched=True)

# %%
# 5. Label Encoding and Dataset Formatting
# For the 20 Newsgroups dataset, ensure the target column is correctly encoded.
newsgroups_encoded = newsgroups_encoded.class_encode_column('label')
newsgroups_encoded.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# %%
# 6. Data Collator for Dynamic Padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# %%
# 7. Training Arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=1e-5,
    per_device_train_batch_size=256,
    per_device_eval_batch_size=256,
    num_train_epochs=3,
    weight_decay=0.01,
)

# %%
# 8. Fine-tuning on the 20 Newsgroups Dataset
# Create a model with 20 output labels.
model_newsgroups = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=20
)
# Optionally, move the model to the selected device.
model_newsgroups.gradient_checkpointing_enable()
model_newsgroups.to(device)

trainer_newsgroups = Trainer(
    model=model_newsgroups,
    args=training_args,
    train_dataset=newsgroups_trimmed['train'],
    eval_dataset=newsgroups_trimmed['test'],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

print("Starting training on the 20 Newsgroups dataset...")
trainer_newsgroups.train()
print("Finished training on the 20 Newsgroups dataset.\n")

# %%
# Save the Trained Model Locally
save_directory = "../artifacts/trained_model"
trainer_newsgroups.save_model(save_directory)
# Save the tokenizer as well so that it can be easily reloaded later.
tokenizer.save_pretrained(save_directory)
print(f"Trained model and tokenizer saved to '{save_directory}'.")

# %%
