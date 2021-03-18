import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import numpy as np


seed_val = 42
batch_size = 32
num_labels = 6
learning_rate = 2e-5
output_dir = '../../model_folder/'

# df = pd.read_csv("../../data/queries2categories.csv", delimiter=',', header=None,
#                 names=['sentence', 'label'])

d = {'sentence': ["shoes for my mum with boots", "fancy dress to buy", "shirt for party"], 'label': [3, 1, 0]}
df = pd.DataFrame(data=d)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print(f'We will use the GPU:{torch.cuda.get_device_name(0)}')
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


# Report the number of sentences.
print(f"Number of testing sentences: {df.shape[0]}")


# Get the lists of sentences and their labels.
sentences = df.sentence.tolist()
labels = df.label.values

# Load the BERT tokenizer.
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


# Tokenize all of the sentences and map the tokens to their word IDs.
encoded_sentences = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
# print(encoded_sentences)

test_inputs = torch.as_tensor(encoded_sentences['input_ids'])
test_labels = torch.as_tensor(labels)
test_masks = torch.as_tensor(encoded_sentences['attention_mask'])

# Create the DataLoaders
test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = SequentialSampler(test_data)
test_data_loader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)


model = BertForSequenceClassification.from_pretrained(
    output_dir)

print(f'Predicting labels for {len(sentences)} test sentences...')
model.eval()

predictions, true_labels = [], []

# Predict
for batch in test_data_loader:
    batch = tuple(t.to(device) for t in batch)

    b_input_ids, b_input_mask, b_labels = batch

    with torch.no_grad():
        # Forward pass, calculate logit predictions
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

    logits = outputs[0]

    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    # Store predictions and true labels
    predictions.append(logits)
    true_labels.append(label_ids)

predictions = [np.argmax(pred, axis=1).flatten() for pred in predictions]
print(predictions)
print(true_labels)

print('DONE.')
