import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np


import logging
logging.basicConfig(level=logging.INFO)

output_dir = '../../model_folder/'


class CategoryPredictorClassifier:
    """
    Given a query Q from user, predicts the intent of query Q and maps them to categories
    """

    def __init__(self):

        # Load pre-trained model tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.model = BertForSequenceClassification.from_pretrained(output_dir)

        self.model.eval()

    def predict(self, query: str) -> int:
        """
        :param query: input query from user
        :return: int categories with prediction score
            0 => OOD
            1 => dress
            2 => shoes
            3 => men shoes
            4 => women shoes
            5 => women boots
        """

        # Tokenize and include mask
        encoded_sentence = self.tokenizer(
            [query, ], padding=True, truncation=True, return_tensors="pt")

        test_inputs = torch.as_tensor(encoded_sentence['input_ids'])
        test_masks = torch.as_tensor(encoded_sentence['attention_mask'])

        with torch.no_grad():
            outputs = self.model(test_inputs, token_type_ids=None, attention_mask=test_masks)

        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        return np.argmax(logits, axis=1).flatten()[0]


# Tokenize input
queries = [
    "men shoes",
    "summer dress",
    "shirt for party", # OOD
    "the boots for women",
]

category_predictor_ns = CategoryPredictorClassifier()
for query in queries:
    print(f"{query}:    {category_predictor_ns.predict(query)}")

