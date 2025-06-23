# BERT Fine-Tuning for Named Entity Recognition (NER) Task

## Overview

This notebook demonstrates how to fine-tune the BERT model for a Named Entity Recognition (NER) task using the CoNLL-2003 dataset. The process includes data preprocessing, tokenization, model training, and evaluation.

## Dataset

The dataset used is **CoNLL-2003**, which contains annotated text for named entities such as:
- **PER** (Person)
- **ORG** (Organization)
- **LOC** (Location)
- **MISC** (Miscellaneous)

The dataset is loaded using the `datasets` library from Hugging Face.

## Key Steps

### 1. Data Loading and Exploration
- The dataset is loaded using `load_dataset("conll2003")`.
- The structure of the dataset is explored, including the features like `tokens` and `ner_tags`.

### 2. Tokenization
- The BERT tokenizer (`bert-base-cased`) is used to tokenize the input text.
- Special handling is applied to align labels with subword tokens, ensuring labels are correctly assigned even when words are split into subwords.

### 3. Label Alignment
- A custom function `align_labels_with_tokens` is implemented to handle cases where tokens are split into subwords. Labels for subwords are adjusted to maintain consistency with the original entity tags.

### 4. Model Initialization
- The `AutoModelForTokenClassification` class is used to initialize a BERT model for token classification.
- The model is configured with the label mappings (`id2label` and `label2id`).

### 5. Training Setup
- **TrainingArguments** are defined to configure the training process, including learning rate, epochs, and evaluation strategy.
- A **DataCollatorForTokenClassification** is used for dynamic padding of inputs and labels within batches.

### 6. Evaluation
- The `seqeval` metric is used to compute precision, recall, F1-score, and accuracy during evaluation.
- A custom `compute_metrics` function processes model predictions and ground truth labels for evaluation.

### 7. Training
- The model is trained using the `Trainer` class from Hugging Face.
- Alternatively, training is also demonstrated using `accelerate` for faster processing with distributed training capabilities.

### 8. Results
- The model achieves high precision, recall, and F1-scores on the validation set, demonstrating effective fine-tuning for the NER task.

## Dependencies

To run this notebook, ensure the following libraries are installed:
- `transformers`
- `datasets`
- `torch`
- `evaluate`
- `seqeval`
- `accelerate`

Install them using:
```bash
pip install transformers datasets torch evaluate seqeval accelerate
```


## Results

After training for 3 epochs, the model achieves the following metrics on the validation set:
- **Precision**: ~94.6%
- **Recall**: ~92.8%
- **F1-score**: ~93.7%
- **Accuracy**: ~98.6%
