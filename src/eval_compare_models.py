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
save_directory = "../artifacts/trained_bert"
bert_model = AutoModelForSequenceClassification.from_pretrained(save_directory)
bert_tokenizer = AutoTokenizer.from_pretrained(save_directory)

save_directory = "../artifacts/trained_modernbert"
modernbert_model = AutoModelForSequenceClassification.from_pretrained(save_directory)
modernbert_tokenizer = AutoTokenizer.from_pretrained(save_directory)

max_length = 50

# %%
def eval_model(tokenizer, newsgroups, max_length, model, model_name):
    # Tokenization with Dynamic Padding (no fixed padding here)
    newsgroups = newsgroups.map(
        lambda x: helpers.compute_length(x, tokenizer)
    )
    # Apply the tokenize_and_trim function to both splits (train and test)
    newsgroups_trimmed = newsgroups.map(
        lambda x: helpers.tokenize_and_trim(x, tokenizer, max_length=max_length),
        batched=True
    )
    train_dataset = newsgroups_trimmed['train']
    eval_dataset = newsgroups_trimmed['test']

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

    # 6. Data Collator for Dynamic Padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

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
    # -------------------------------
    # Run Predictions on the Entire Validation (Test) Set
    # -------------------------------
    print("\nRunning predictions on the validation set...")
    predictions_output = trainer.predict(eval_dataset)
    logits = predictions_output.predictions
    true_labels = predictions_output.label_ids  # True labels
    predicted_labels = np.argmax(logits, axis=1)

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
    print("---------------------------------------")
    print(f"\n----------Evaluation Metrics for {model_name}: -------------")
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

    return f1, accuracy, precision, recall, conf_mat, auc_roc

# %%
# BERT
tokenizer = bert_tokenizer
model = bert_model
model_name = "BERT-base"
f1_bert, accuracy_bert, precision_bert, recall_bert, conf_mat_bert, auc_roc_bert = eval_model(tokenizer, newsgroups, max_length, model, model_name)

# %%
# ModernBERT
tokenizer = modernbert_tokenizer
model = modernbert_model
model_name = "ModernBERT-base"
f1_modernbert, accuracy_modernbert, precision_modernbert, recall_modernbert, conf_mat_modernbert, auc_roc_modernbert = eval_model(tokenizer, newsgroups, max_length, model, model_name)

# %%
