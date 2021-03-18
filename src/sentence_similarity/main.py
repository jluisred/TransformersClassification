from transformers import AutoTokenizer, AutoModel
import torch
import csv
from torch import Tensor


def pytorch_cos_sim(a: Tensor, b: Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    This function can be used as a faster replacement for 1-scipy.spatial.distance.cdist(a,b)
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    return torch.mm(a_norm, b_norm.transpose(0, 1))


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")
model = AutoModel.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")

query_missing_list = []
with open('./top100_missing.csv', newline='') as csvfile:
    missing_query_reader = csv.reader(csvfile)
    for row in missing_query_reader:
        query_missing_list.append(row[1])


queries_with_asin_list = []
with open('./queries_with_asins.csv', newline='') as csvfile:
    queries_with_asin_reader = csv.reader(csvfile)
    for row in queries_with_asin_reader:
        queries_with_asin_list.append(row[0])


queries_with_asin_list = queries_with_asin_list[1:5000]
print("tokenizing")
queries_with_asin_list_encoded_input = tokenizer(queries_with_asin_list,
                                                 padding=True,
                                                 truncation=True,
                                                 max_length=256,
                                                 return_tensors='pt')
print(len(queries_with_asin_list_encoded_input['attention_mask']))
print("tokenized")

# Compute token embeddings
with torch.no_grad():
    queries_with_asin_model_output = model(**queries_with_asin_list_encoded_input)
print("embeddings computed")


queries_with_asin_embeddings = mean_pooling(
    queries_with_asin_model_output,
    queries_with_asin_list_encoded_input['attention_mask']
)

for query_missing in query_missing_list:
    print(f"Analysing query {query_missing}")

    query_missing_sentence = [query_missing, ]

    # Tokenize sentences
    query_missing_encoded_input = tokenizer(query_missing_sentence,
                                            padding=True,
                                            truncation=True,
                                            max_length=256,
                                            return_tensors='pt')
    # Compute token embeddings
    with torch.no_grad():
        query_missing_model_output = model(**query_missing_encoded_input)

    # Perform pooling. In this case, mean pooling
    query_missing_embedding = mean_pooling(
        query_missing_model_output,
        query_missing_encoded_input['attention_mask']
    )

    print(f"    Missing query's embedding created.")

    #Compute cosine-similarits
    cosine_scores = pytorch_cos_sim(query_missing_embedding, queries_with_asin_embeddings)
    print(f"    Distances calculated.")

    # Find the pairs with the highest cosine similarity scores
    pairs = []
    for j in range(len(cosine_scores[0])):
        pairs.append({'index': [1, j], 'score': cosine_scores[0][j]})

    # Sort scores in decreasing order
    pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)

    for pair in pairs[0:10]:
        i, j = pair['index']
        if pair['score'] >= 0.9:
            print("         {} \t\t {} \t\t Score: {:.4f}".format(query_missing, queries_with_asin_list[j], pair['score']))


