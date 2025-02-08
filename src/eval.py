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
from sklearn.metrics import roc_curve, auc
import seaborn as sns
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)
import numpy as np
from sklearn.preprocessing import label_binarize
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
set_seed(42)

# %%
# 2. Model and Tokenizer Initialization
model_name = "answerdotai/ModernBERT-base"  # Or "answerdotai/ModernBERT-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# %%
# 3. Load Datasets
# 20 Newsgroups dataset (assumes the "SetFit/20_newsgroups" dataset is available)
newsgroups = load_dataset('SetFit/20_newsgroups')
# MultiNLI dataset (this downloads the multi_nli dataset from Hugging Face)
# multinli = load_dataset('multi_nli')


# %%
# 4. Tokenization with Dynamic Padding (no fixed padding here)
def tokenize_newsgroups(example):
    # Tokenize the 'text' field and apply truncation.
    return tokenizer(example['text'], truncation=True)


def tokenize_multinli(example):
    # Tokenize both 'premise' and 'hypothesis' fields.
    return tokenizer(example['premise'], example['hypothesis'], truncation=True)


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

# For MultiNLI, encode the 'label' column (assuming it is named "label").
# multinli_encoded = multinli_encoded.class_encode_column('label')
# multinli_encoded.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

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
# 8a. Fine-tuning on the 20 Newsgroups Dataset
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
# Load eval
newsgroups = load_dataset('SetFit/20_newsgroups')
eval_dataset = newsgroups["test"]

# Tokenize the evaluation dataset
eval_dataset = eval_dataset.map(tokenize_and_trim, batched=True)
# If needed, encode the label column (if it wasn't already done during training)
eval_dataset = eval_dataset.class_encode_column("label")
# Format the dataset to return PyTorch tensors
eval_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])


# %%
# Load the Trained Model Locally
# Specify the directory where the model was saved
save_directory = "../artifacts/trained_model"

# Load the model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(save_directory)
tokenizer = AutoTokenizer.from_pretrained(save_directory)
# Create dummy TrainingArguments (only the evaluation batch size is used here)
training_args = TrainingArguments(
    output_dir="./results",
    per_device_eval_batch_size=8,  # adjust batch size as needed
)

# Create the Trainer using the loaded model
trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)


# %%
# -------------------------------
# Run Predictions on the Entire Validation (Test) Set
# -------------------------------
print("\nRunning predictions on the validation set...")
predictions_output = trainer.predict(eval_dataset)
logits = predictions_output.predictions
true_labels = predictions_output.label_ids  # True labels
predicted_labels = np.argmax(logits, axis=1)

# %%
# -------------------------------
# Compute Evaluation Metrics
# -------------------------------
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='macro')
recall = recall_score(true_labels, predicted_labels, average='macro')
f1 = f1_score(true_labels, predicted_labels, average='macro')
conf_mat = confusion_matrix(true_labels, predicted_labels)

# For multiclass AUC-ROC, we need predicted probabilities.
probs = softmax(logits, axis=1)
try:
    auc_roc = roc_auc_score(true_labels, probs, multi_class='ovr', average='macro')
except Exception as e:
    auc_roc = None
    print("Could not compute AUC-ROC:", e)

# -------------------------------
# Print Evaluation Metrics
# -------------------------------
print("\nEvaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (macro): {precision:.4f}")
print(f"Recall (macro): {recall:.4f}")
print(f"F1 Score (macro): {f1:.4f}")
print("Confusion Matrix:")
print(conf_mat)
if auc_roc is not None:
    print(f"AUC-ROC (macro, OVR): {auc_roc:.4f}")
else:
    print("AUC-ROC could not be computed.")

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", cbar=True)

plt.xlabel("Predicted Labels", fontsize=12)
plt.ylabel("True Labels", fontsize=12)
plt.title("Confusion Matrix", fontsize=14)
plt.show()

# %%
# Number of classes (assumes classes are 0-indexed)
n_classes = probs.shape[1]

# Convert true labels to a one-hot encoded format.
true_labels_binarized = label_binarize(true_labels, classes=list(range(n_classes)))

plt.figure(figsize=(12, 10))

# Plot ROC curve for each class.
for i in range(n_classes):
    # Compute fpr, tpr for the i-th class.
    fpr, tpr, _ = roc_curve(true_labels_binarized[:, i], probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')

# Plot the chance line.
plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Chance')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate", fontsize=14)
plt.ylabel("True Positive Rate", fontsize=14)
plt.title("ROC Curves for Each Class (One-vs-Rest)", fontsize=16)
plt.legend(loc="lower right", fontsize=10)
plt.grid(True)
plt.show()


# %%
