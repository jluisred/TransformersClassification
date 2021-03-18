import pandas as pd
import torch
import time
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup
import random
import numpy as np
from src.utils.training_utils import format_time, flat_accuracy

seed_val = 42
batch_size = 32
num_labels = 6
learning_rate = 2e-5
output_dir = '../../model_folder/'

df = pd.read_csv("../../data/queries2categories.csv", delimiter=',', header=None,
                 names=['sentence', 'label'])


if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print(f'We will use the GPU:{torch.cuda.get_device_name(0)}')
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


# Report the number of sentences.
print(f"Number of training sentences: {df.shape[0]}")
# Display 10 random rows from the data.
# print(df.sample(10))


# Get the lists of sentences and their labels.
sentences = df.sentence.tolist()
labels = df.label.values

# Load the BERT tokenizer.
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Tokenize all of the sentences and map the tokens to their word IDs.
encoded_sentences = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
# print(encoded_sentences)


# Use 90% for training and 10% for validation.
train_inputs, val_inputs, train_labels, val_labels, train_att, val_att = train_test_split(
    encoded_sentences['input_ids'],
    labels,
    encoded_sentences['attention_mask'],
    random_state=2020,
    test_size=0.1)

print(len(train_labels))
print(train_labels)

train_inputs = torch.as_tensor(train_inputs)
validation_inputs = torch.as_tensor(val_inputs)
train_labels = torch.as_tensor(train_labels)
validation_labels = torch.as_tensor(val_labels)
train_masks = torch.as_tensor(train_att)
validation_masks = torch.as_tensor(val_att)

# Create the DataLoaders
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_data_loader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
val_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
val_sampler = SequentialSampler(val_data)
val_data_loader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)


model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=num_labels
)

optimizer = AdamW(
    model.parameters(),
    lr=learning_rate,
    eps=1e-8
)

epochs = 4
total_steps = len(train_data_loader) * epochs

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps)


random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

training_stats = []

total_t0 = time.time()

# For each epoch...
for epoch_i in range(0, epochs):

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    t0 = time.time()

    total_train_loss = 0
    model.train()

    for step, batch in enumerate(train_data_loader):
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)

            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(
                step,
                len(train_data_loader),
                elapsed)
            )
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()

        output = model(b_input_ids,
                             token_type_ids=None,
                             attention_mask=b_input_mask,
                             labels=b_labels)

        total_train_loss += output["loss"].item()
        output["loss"].backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

    avg_train_loss = total_train_loss / len(train_data_loader)
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(training_time))
    print("")
    print("Running Validation...")

    t0 = time.time()
    model.eval()

    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    for batch in val_data_loader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        with torch.no_grad():
            output = model(b_input_ids,
                                   token_type_ids=None,
                                   attention_mask=b_input_mask,
                                   labels=b_labels)
        loss = output["loss"]
        logits = output["logits"]
        total_eval_loss += loss.item()
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        total_eval_accuracy += flat_accuracy(logits, label_ids)

    avg_val_accuracy = total_eval_accuracy / len(val_data_loader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    avg_val_loss = total_eval_loss / len(val_data_loader)

    validation_time = format_time(time.time() - t0)

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

print("")
print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

print(f"Saving model to {output_dir}")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

