import torch
from transformers import BertTokenizer, BertForNextSentencePrediction
from torch.nn.functional import softmax

import logging
logging.basicConfig(level=logging.INFO)


class CategoryPredictorNextSentence:
    """
    Given a query Q from user, predicts top n categories to refine the product intent in Q
    """

    query_pattern = "I want {query}"
    category_pattern = "the {category}"

    def __init__(self, categories: list):
        # load pretrained BERT
        self.categories = categories

        # Load pre-trained model tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')

        self.model.eval()

    def predict(self, query: str) -> list:
        """
        :param query: input query from user
        :return: ordered list of categories with prediction score
        """
        query_sentence = self.query_pattern.format(query=query)

        prob_per_category = []

        with torch.no_grad():
            for category in self.categories:
                category_sentence = self.category_pattern.format(category=category)

                # Tokenize and include separators
                query_tokens = ["[CLS]"] + self.tokenizer.tokenize(query_sentence) + ["[SEP]"]
                category_tokens = self.tokenizer.tokenize(category_sentence) + ["[SEP]"]
                indexed_tokens = self.tokenizer.convert_tokens_to_ids(
                    query_tokens + category_tokens)
                segments_ids = [0] * len(query_tokens) + [1] * len(category_tokens)
                tokens_tensor = torch.tensor([indexed_tokens])
                segments_tensors = torch.tensor([segments_ids])

                prediction = self.model(tokens_tensor, token_type_ids=segments_tensors)
                probabilities = softmax(prediction[0], dim=1)
                prob_per_category.append((category, probabilities[0][0]))

        return sorted(prob_per_category, key=lambda cat_tup: cat_tup[1], reverse=True)


# Tokenize input
queries = [
    "men shoes",
    "summer dress",
    "shirt for party",
]

categories = ["size", "department", "style", "sleeve", "neckline", "occasion", "pattern"]

category_predictor_ns = CategoryPredictorNextSentence(categories)

for query in queries:
    print(f"{query}:    {category_predictor_ns.predict(query)}")











