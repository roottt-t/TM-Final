import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments
from datasets import load_dataset, Dataset

# Step 1: Load and preprocess the dataset
def preprocess_data(file_path):
    # Read and process your dataset file
    texts = []
    labels = []
    with open(file_path, 'r') as f:
        while True:
            try:
                text = next(f).strip()  # Read the current line for text
                label_line = next(f).strip()  # Read the next line for labels
                texts.append(text)
                labels.append(1 if 'ADR' in label_line else 0)  # Simplified label extraction
            except StopIteration:
                break  # End of file reached, exit the loop

    return Dataset.from_dict({'text': texts, 'label': labels})



# Load datasets
train_dataset = preprocess_data("train.txt")
val_dataset = preprocess_data("val.txt")
test_dataset = preprocess_data("test.txt")  

# Step 2: Tokenization
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')

def tokenize(batch):
    return tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True)


train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

# Set input columns for Hugging Face Trainer
train_dataset = train_dataset.rename_columns({"label": "labels"})
val_dataset = val_dataset.rename_columns({"label": "labels"})

# Set format for PyTorch
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Step 3: Load Model

model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-large")

# Step 4: Define Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# Step 5: Trainer
def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


if __name__ == "__main__":
    # Step 6: Train the Model
    trainer.train()
    # Step 7: Save the Model
    model.save_pretrained("./fine_tuned_xlm_roberta")
    tokenizer.save_pretrained("./fine_tuned_xlm_roberta")



    # Step 8: Evaluate the Model on Test Set
    test_results = trainer.evaluate(eval_dataset=test_dataset)
    print("Test results:", test_results)
    print(test_results)
