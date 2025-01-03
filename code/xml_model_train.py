import os
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

import torch
torch.cuda.empty_cache()

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

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

sentences, tags = load_dataset( "train.txt")

# Split into training,  validation, test sets  72 % 8% 20%

train_sentences, temp_sentences, train_tags, temp_tags = train_test_split(sentences, tags, test_size=0.28, random_state=42)
<<<<<<< HEAD
val_sentences, test_sentences, val_tags, test_tags = train_test_split(temp_sentences, temp_tags, test_size=0.71, random_state=42)

print("train", len(train_sentences))
print("val", len(val_sentences))
print("test", len(test_sentences))
=======
val_sentences, test_sentences, val_tags, test_tags = train_test_split(temp_sentences, temp_tags, test_size=0.7, random_state=42)

>>>>>>> 427790c (save xml temp files)

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
test_inputs, test_label_ids = tokenize_and_align_labels(test_sentences, test_tags, tokenizer, label_to_id)

<<<<<<< HEAD

=======
>>>>>>> 427790c (save xml temp files)
import torch
from torch.utils.data import Dataset

class NERDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs["input_ids"])

    def __getitem__(self, idx):
<<<<<<< HEAD
=======
        if idx >= len(self.inputs["input_ids"]) or idx >= len(self.labels):
            return None
>>>>>>> 427790c (save xml temp files)
        return {
            "input_ids": self.inputs["input_ids"][idx],
            "attention_mask": self.inputs["attention_mask"][idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

# Create datasets
train_dataset = NERDataset(train_inputs, train_label_ids)
val_dataset = NERDataset(val_inputs, val_label_ids)
test_dataset = NERDataset(test_inputs, test_label_ids)


from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

# Load pre-trained model with the number of unique labels
model = AutoModelForTokenClassification.from_pretrained(
    "xlm-roberta-base", 
    num_labels=len(label_to_id))

<<<<<<< HEAD
=======

>>>>>>> 427790c (save xml temp files)
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

# Training arguments
training_args = TrainingArguments(
<<<<<<< HEAD
    #fp16=True,
=======
>>>>>>> 427790c (save xml temp files)
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
)

# Define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

if __name__ == "__main__":

    # Train the model
    trainer.train()

    model.save_pretrained("./fine_tuned_xlm_roberta")
    tokenizer.save_pretrained("./fine_tuned_xlm_roberta")

    # Evaluate on the test set
    print("test_dataset", len(test_dataset))
    test_results = trainer.evaluate(test_dataset)
    print(test_results)

    import torch

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

<<<<<<< HEAD
    # Load model and tokenizer
    from transformers import AutoTokenizer, AutoModelForTokenClassification

    tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_xlm_roberta")
    model = AutoModelForTokenClassification.from_pretrained("./fine_tuned_xlm_roberta").to(device)
    
    text = "I feel headache."
=======
    text = "I feel pain."
>>>>>>> 427790c (save xml temp files)
    encoded_input = tokenizer(text, return_tensors="pt").to(device)

    # forward pass
    output = model(**encoded_input).logits
<<<<<<< HEAD
=======
    print(output)
>>>>>>> 427790c (save xml temp files)

    # get predictions
    predictions = torch.argmax(output, dim=2)
    # get predicted labels
    predicted_labels = [id_to_label[label] for label in predictions[0].tolist()]
    print(predicted_labels)
<<<<<<< HEAD
    
=======

>>>>>>> 427790c (save xml temp files)
    tokens = tokenizer.convert_ids_to_tokens(encoded_input["input_ids"][0])
    for token, label in zip(tokens, predicted_labels):
        print(f"{token}: {label}")

<<<<<<< HEAD
=======





>>>>>>> 427790c (save xml temp files)
