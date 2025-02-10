# %%
# Write about this protject

# %%
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed,
)
import matplotlib.pyplot as plt

import helpers

# %%
# Set seed for reproducibility
set_seed(422)

device = helpers.identify_device()

# %%
# Model and Tokenizer Initialization
model_name = "answerdotai/ModernBERT-base"  # Or "answerdotai/ModernBERT-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# %%
# Load Datasets
newsgroups = load_dataset('SetFit/20_newsgroups')

# %%
num_labels = len(set(newsgroups["train"]["label"]))
print(f"There are {num_labels} classes of documents")

# %%
# Print out the labels
print(f"Labels: {set(newsgroups["train"]["label_text"])}")

# %%
# For each label, print one example document


# %%
# Tokenization with Dynamic Padding (no fixed padding here)
newsgroups_encoded = newsgroups.map(
    lambda x: helpers.tokenize_newsgroups(x, tokenizer),
    batched=True
)
newsgroups = newsgroups.map(
    lambda x: helpers.compute_length(x, tokenizer)
)

# %%
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
# Apply the tokenize_and_trim function to both splits (train and test)
max_length = 50
newsgroups_trimmed = newsgroups.map(
    lambda x: helpers.tokenize_and_trim(x, tokenizer, max_length=max_length),
    batched=True
)

train_dataset = newsgroups_trimmed['train']
eval_dataset = newsgroups_trimmed['test']

# %%
# For the 20 Newsgroups dataset, ensure the target column is correctly encoded.
newsgroups_encoded = newsgroups_encoded.class_encode_column('label')
newsgroups_encoded.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# %%
# 6. Data Collator for Dynamic Padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# %%
# Training Arguments
num_epochs = 10
batch_size = 256
lr = 1e-5
weight_decay = 0.01
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    logging_strategy='epoch', 
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=weight_decay,
)

# %%
# Fine-tuning on the 20 Newsgroups Dataset
# Create a model with 20 output labels.
model_newsgroups = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels
)
model_newsgroups.gradient_checkpointing_enable()
model_newsgroups.to(device)

trainer_newsgroups = Trainer(
    model=model_newsgroups,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# %%
print("Starting training on the 20 Newsgroups dataset...")
trainer_newsgroups.train()
print("Finished training on the 20 Newsgroups dataset.\n")

# %%
# Access the log history (a list of dictionaries)
log_history = trainer_newsgroups.state.log_history

# Initialize lists to store metrics per epoch
epochs = []
train_losses = []
eval_losses = []

# Iterate over the log history and extract entries that have an "epoch" key.
for log in log_history:
    if "epoch" in log:
        epochs.append(log["epoch"])
        # Some log entries may have only one type of loss.
        train_losses.append(log.get("loss", None))
        eval_losses.append(log.get("eval_loss", None))

# Remove any None values (in case some epochs didn't log one of the losses)
# Here, we assume that each epoch should have both values.
epochs = [e for e, t, v in zip(epochs, train_losses, eval_losses) if t is not None and v is not None]
train_losses = [t for t in train_losses if t is not None]
eval_losses = [v for v in eval_losses if v is not None]

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, label="Train Loss", marker="o")
plt.plot(epochs, eval_losses, label="Eval Loss", marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Evaluation Loss per Epoch")
plt.legend()
plt.grid(True)
plt.show()

# %%
# Save the model and tokenizer
save_directory = "../artifacts/trained_model"
trainer_newsgroups.save_model(save_directory)
tokenizer.save_pretrained(save_directory)
print(f"Trained model and tokenizer saved to '{save_directory}'.")

# %%
