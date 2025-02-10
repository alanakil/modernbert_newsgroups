# %%
# Write about this protject

# %%
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

import helpers

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# %%
# Set seed for reproducibility
set_seed(422)

device = helpers.identify_device()

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
# Load the model and tokenizer
save_directory = "../artifacts/trained_model"
model = AutoModelForSequenceClassification.from_pretrained(save_directory)
tokenizer = AutoTokenizer.from_pretrained(save_directory)
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
# Apply the tokenize_and_trim function to both splits (train and test)
max_length = 100
newsgroups_trimmed = newsgroups.map(
    lambda x: helpers.tokenize_and_trim(x, tokenizer, max_length=max_length),
    batched=True
)

train_dataset = newsgroups_trimmed['train']
eval_dataset = newsgroups_trimmed['test']

# %%
# Training Arguments
num_epochs = 5
batch_size = 128
lr = 1e-5
weight_decay = 0.01
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=weight_decay,
)

# %%
# 6. Data Collator for Dynamic Padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# %%
# Fine-tuning on the 20 Newsgroups Dataset
# Create a model with 20 output labels.
model.gradient_checkpointing_enable()
model.to(device)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
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
conf_mat = confusion_matrix(true_labels, predicted_labels).squeeze()
conf_mat = np.round(100*conf_mat / np.sum(conf_mat, axis=1).reshape(20, 1), 0)

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
# sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", cbar=True)
sns.heatmap(conf_mat, annot=True, cmap="Blues", cbar=True)

plt.xlabel("Predicted Labels", fontsize=12)
plt.ylabel("True Labels", fontsize=12)
plt.title("Confusion Matrix", fontsize=14)
plt.show()

# %%
# Convert logits to probabilities using softmax
probs = softmax(logits, axis=1)
n_classes = probs.shape[1]

# Binarize the true labels for a one-vs-rest approach.
# This creates a binary matrix of shape (n_samples, n_classes)
true_labels_bin = label_binarize(true_labels, classes=np.arange(n_classes))

# Compute micro-average ROC curve and AUC.
# Flatten the arrays to treat the problem as one binary classification.
fpr, tpr, _ = roc_curve(true_labels_bin.ravel(), probs.ravel())
roc_auc = auc(fpr, tpr)

# Plot the overall ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'Overall ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Overall ROC Curve (Micro-Average)')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

print(f"Overall AUC-ROC: {roc_auc:.4f}")


# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_curve, auc
from sklearn.calibration import calibration_curve

# --- 1. Calibration ---
# Compute calibration curve: this will give you the fraction of positives
# for each bin of predicted probability.
plt.figure(figsize=(12, 10))

# Loop through each class and plot its calibration curve.
for i in range(n_classes):
    # Binarize the true labels for class i:
    true_binary = (true_labels == i).astype(int)
    # Get predicted probabilities for class i:
    prob_pred = probs[:, i]
    
    # Compute the calibration curve.
    fraction_of_positives, mean_predicted_value = calibration_curve(true_binary, prob_pred, n_bins=10)
    
    plt.plot(mean_predicted_value, fraction_of_positives, marker='o', linewidth=2, label=f'Class {i}')

# Plot the perfect calibration line.
plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
plt.xlabel("Mean predicted probability", fontsize=14)
plt.ylabel("Fraction of positives", fontsize=14)
plt.title("Calibration Curves for Each Class (One-vs-Rest)", fontsize=16)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# --- 2. Threshold Tuning ---
# Here we choose a range of thresholds and compute the F1 score for each.
thresholds = np.linspace(0, 1, 101)
f1_scores = []


best_thresholds = {}

# Iterate over each class
for target_class in range(n_classes):
    y_true_bin = (true_labels == target_class).astype(int)
    probs_for_class = probs[:, target_class]
    
    thresholds = np.linspace(0, 1, 101)
    f1_scores = []
    
    for threshold in thresholds:
        y_pred_bin = (probs_for_class >= threshold).astype(int)
        f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
        f1_scores.append(f1)
    
    f1_scores = np.array(f1_scores)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_thresholds[target_class] = (best_threshold, f1_scores[best_idx])
    
    # Optionally, plot for each class
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, f1_scores, marker="o")
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.title(f"Threshold Tuning for Class {target_class}")
    plt.axvline(x=best_threshold, color='r', linestyle='--', 
                label=f"Best threshold: {best_threshold:.2f}")
    plt.legend()
    plt.grid(True)
    plt.show()

print("Best thresholds per class:")
for cls, (thr, f1_val) in best_thresholds.items():
    print(f"  Class {cls}: Threshold = {thr:.2f}, F1 Score = {f1_val:.4f}")

f1_scores = np.array(f1_scores)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

# Plot F1 scores versus thresholds.
plt.figure(figsize=(8, 6))
plt.plot(thresholds, f1_scores, marker="o")
plt.xlabel("Threshold")
plt.ylabel("F1 Score")
plt.title("Threshold Tuning (F1 Score)")
plt.grid(True)
plt.axvline(x=best_threshold, color='r', linestyle='--', label=f"Best threshold: {best_threshold:.2f}")
plt.legend()
plt.show()

print(f"Best threshold: {best_threshold:.2f} with F1 score: {best_f1:.4f}")


# %%
# Convert true labels to a one-hot encoded format.
true_labels_binarized = label_binarize(true_labels, classes=list(range(num_labels)))

plt.figure(figsize=(12, 10))

# Plot ROC curve for each class.
for i in range(num_labels):
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
# Determine the unique classes
classes = np.unique(true_labels)

# Dictionary to store metrics for each class.
metrics_dict = {}

for cls in classes:
    # Create binary arrays for the current class:
    # 1 if the instance belongs to the class, 0 otherwise.
    y_true_bin = (true_labels == cls).astype(int)
    y_pred_bin = (predicted_labels == cls).astype(int)
    
    # Compute metrics
    acc = accuracy_score(y_true_bin, y_pred_bin)
    prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    
    # Compute the confusion matrix for this class (binary: class vs. rest)
    cm = confusion_matrix(y_true_bin, y_pred_bin)
    
    # Compute AUC-ROC.
    # Note: AUC-ROC is best computed on continuous scores, so using binary predictions
    # might result in a degenerate ROC curve. We'll attempt to compute it,
    # but if only one class is present, roc_auc_score will raise an error.
    try:
        auc_val = roc_auc_score(y_true_bin, y_pred_bin)
    except ValueError as e:
        auc_val = None
        print(f"Could not compute AUC-ROC for class {cls}: {e}")
    
    # Store the metrics in a dictionary for the current class.
    metrics_dict[cls] = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'auc_roc': auc_val,
        'confusion_matrix': cm
    }

# Print the computed metrics for each class.
for cls, metrics in metrics_dict.items():
    print(f"Metrics for class {cls}:")
    print(f"  Accuracy       : {metrics['accuracy']:.4f}")
    print(f"  Precision      : {metrics['precision']:.4f}")
    print(f"  Recall         : {metrics['recall']:.4f}")
    print(f"  F1 Score       : {metrics['f1_score']:.4f}")
    if metrics['auc_roc'] is not None:
        print(f"  AUC-ROC        : {metrics['auc_roc']:.4f}")
    else:
        print("  AUC-ROC        : Not computable (possibly only one class present)")
    print("  Confusion Matrix:")
    print(metrics['confusion_matrix'])
    print("-" * 40)

# %%
