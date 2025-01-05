import os
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW, get_scheduler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from accelerate import Accelerator


# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")

# Define function to load and preprocess the dataset
def load_dataset(file_path):
    sentences = []
    labels = []
    with open(file_path, "r", encoding="utf-8") as file:
        sentence = []
        label = []
        for line in file:
            if line.strip() == "":  # Sentence separator
                if sentence:
                    sentences.append(sentence)
                    labels.append(label)
                    sentence = []
                    label = []
            else:
                token, tag = line.strip().split()
                sentence.append(token)
                label.append(tag)
    if sentence:  # Add the last sentence if file does not end with a blank line
        sentences.append(sentence)
        labels.append(label)
    return sentences, labels

# Load the dataset

# sentences, tags = load_dataset( "train.txt")

sentences,tags = load_dataset( "train_single_tag.txt")

# Split into training,  validation, test sets  72 % 8% 20%

train_sentences, temp_sentences, train_tags, temp_tags = train_test_split(sentences, tags, test_size=0.28, random_state=42)
val_sentences, test_sentences, val_tags, test_tags = train_test_split(temp_sentences, temp_tags, test_size=0.71, random_state=42)


def tokenize_and_align_labels(sentences, labels, tokenizer, label_to_id):
    tokenized_inputs = tokenizer(
        sentences,
        is_split_into_words=True,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    aligned_labels = []
    for i, label in enumerate(labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Ignore special tokens
            elif word_idx != previous_word_idx:  # Assign label to the first subword only
                label_ids.append(label_to_id[label[word_idx]])
            else:
                label_ids.append(-100)  # Other subwords get -100
            previous_word_idx = word_idx
        aligned_labels.append(label_ids)
    return tokenized_inputs, aligned_labels

# Define label to ID mapping
unique_labels = sorted(set(tag for tags in train_tags for tag in tags))
print("unique_labels",unique_labels)
label_to_id = {label: i for i, label in enumerate(unique_labels)}
id_to_label = {i: label for label, i in label_to_id.items()}

# Tokenize and align labels
train_inputs, train_label_ids = tokenize_and_align_labels(train_sentences, train_tags, tokenizer, label_to_id)
val_inputs, val_label_ids = tokenize_and_align_labels(val_sentences, val_tags, tokenizer, label_to_id)


import torch
from torch.utils.data import Dataset

class NERDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": self.inputs["input_ids"][idx],
            "attention_mask": self.inputs["attention_mask"][idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

# Create datasets
train_dataset = NERDataset(train_inputs, train_label_ids)
val_dataset = NERDataset(val_inputs, val_label_ids)

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4)

# Initialize Accelerator
accelerator = Accelerator()

from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

# Load pre-trained model with the number of unique labels
model = AutoModelForTokenClassification.from_pretrained(
    "xlm-roberta-base", 
    num_labels=len(label_to_id))

optimizer = AdamW(model.parameters(), lr=5e-5)
num_training_steps = len(train_dataloader) * 3  # 3 epochs
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)


model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler
)

import evaluate
metric = evaluate.load("seqeval")


def compute_metrics(pred):
    predictions, labels = pred
    predictions = predictions.argmax(axis=-1)

    # Remove ignored index (-100)
    true_labels = [[id_to_label[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [id_to_label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def compute_metrics_opt(predictions, labels):
    predictions = predictions.argmax(dim=-1).detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    true_predictions = [[id_to_label[pred] for pred, label in zip(preds, labs) if label != -100] for preds, labs in zip(predictions, labels)]
    true_labels = [[id_to_label[label] for label in labs if label != -100] for labs in labels]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return results

def evaluate(dataloader, model, accelerator):
    model.eval()  # Set model to evaluation mode
    all_predictions = []
    all_labels = []

    for batch in dataloader:
        with torch.no_grad():
            outputs = model(**batch)  # Forward pass
        predictions = accelerator.gather(outputs.logits)  # Gather predictions from all devices
        labels = accelerator.gather(batch["labels"])      # Gather true labels

        all_predictions.append(predictions)
        all_labels.append(labels)

    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)

    # Compute metrics
    metrics = compute_metrics_opt(all_predictions, all_labels)
    return metrics

# # Training arguments
# training_args = TrainingArguments(
#     output_dir="./results",
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     learning_rate=5e-5,
#     per_device_train_batch_size=4,
#     per_device_eval_batch_size=4,
#     num_train_epochs=3,
#     weight_decay=0.01,
#     save_total_limit=2,
#     load_best_model_at_end=True,
# )

# # Define trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics,
# )



if __name__ == "__main__":

    # # Train the model
    # trainer.train()

    # model.save_pretrained("./fine_tuned_xlm_roberta")
    # tokenizer.save_pretrained("./fine_tuned_xlm_roberta")

    # # Evaluate the model on the test set
    # test_sentences, test_tags = load_dataset( "test.txt")
    # test_inputs, test_label_ids = tokenize_and_align_labels(test_sentences, test_tags, tokenizer, label_to_id)
    # test_dataset = NERDataset(test_inputs, test_label_ids)
    # test_result = trainer.evaluate(test_dataset=test_dataset)
    # print(test_result)
    # # Predict on a sample sentence  
    # sample_sentence = "The quick brown fox jumps over the lazy dog"
    # tokenized_sample = tokenizer(sample_sentence, return_tensors="pt")
    # sample_prediction = trainer.predict(tokenized_sample)
    # print(sample_prediction.predictions)
    # print(sample_prediction.label_ids)
    # print(sample_prediction.metrics)
    

    model.train()
    for epoch in range(3):
        total_loss = 0
        for batch in train_dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
    
    # Save the model and tokenizer
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained("./fine_tuned_xlm_roberta")
    tokenizer.save_pretrained("./fine_tuned_xlm_roberta")

    # Evaluate at the end of training
    validation_metrics = evaluate(val_dataloader, model, accelerator)
    print(f"Validation Metrics: {validation_metrics}")

