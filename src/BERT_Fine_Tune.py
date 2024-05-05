import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from transformers import EarlyStoppingCallback, IntervalStrategy
from transformers import AutoTokenizer
from transformers import AutoModel
import torch
from sklearn.linear_model import LogisticRegression
import numpy as np
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification


# Load Distilbert Tokenizer
model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
# Encode Our Example Text
encoded_text = tokenizer("The movie was not good")
tokens = tokenizer.convert_ids_to_tokens(encoded_text.input_ids)
print(encoded_text, len(encoded_text.input_ids))
print(tokens)

tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)
model = AutoModel.from_pretrained(model_ckpt).to(device)

# Use whole dataset in Huggingface dataset format
df_source = pd.DataFrame()  # TO-DO
batch = df_source
# Send inputs from CPU to GPU
inputs = {k: v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names}
# Extract last hidden states
# Disable gradient calculation on PyTorch Side
with torch.no_grad():
    last_hidden_state = model(**inputs).last_hidden_state
# Return latest hidden state as numpy matrix
df_hidden = last_hidden_state[:, 0].cpu().numpy()

X_train = np.array(df_hidden["train"]["hidden_state"])
X_valid = np.array(df_hidden["valid"]["hidden_state"])
X_test = np.array(df_hidden["test"]["hidden_state"])

y_train = np.array(df_hidden["train"]["label"])
y_valid = np.array(df_hidden["valid"]["label"])
y_test = np.array(df_hidden["test"]["label"])

clf_lr = LogisticRegression(max_iter=3000)
clf_lr.fit(X_train, y_train)
clf_lr.score(X_valid, y_valid)


###
#
# Code for fine tunning BERT
#
###
# Set Batch Size
batch_size = 64
logging_steps = len(df_hidden["train"]) // batch_size
num_train_epochs = 30
lr_initial = 2e-5
weight_decay = 1e-3
output_dir = ""
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    learning_rate=lr_initial,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=weight_decay,
    evaluation_strategy="epoch",
    disable_tqdm=False,
    logging_steps=logging_steps,
    push_to_hub=False,
    log_level="error",
)


args = TrainingArguments(
    evaluation_strategy=IntervalStrategy.STEPS,  # "steps"
    eval_steps=50,  # Evaluation and Save happens every 50 steps
    save_total_limit=5,  # Only last 5 models are saved. Older ones are deleted.
    metric_for_best_model="f1",  # Metric to pick the best model
    load_best_model_at_end=True,
)


num_labels: int = None  # TO-DO
compute_metrics: str = None
df_encoded = pd.DataFrame()  # TO-DO
# Create Model
model = AutoModelForSequenceClassification.from_pretrained(
    model_ckpt, num_labels=num_labels
).to(device)
model_name = f"models/{model_ckpt}-finetuned-bert-emotion-tweets"
training_args.output_dir = model_name
trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=df_encoded["train"],
    eval_dataset=df_encoded["validation"],
    tokenizer=tokenizer,
)
trainer.train()
