import torch
from transformers import BertTokenizer, BertForMaskedLM
from torch.nn.functional import softmax

import logging
logging.basicConfig(level=logging.INFO)


class CategoryPredictorMaskedWord:
    """
    Given a query Q from user, predicts top n categories to refine the product intent in Q
    """

    # pattern = "[CLS] I want {query} [SEP] Select the [MASK] of {query2} [SEP]"
    pattern = "[CLS] I want {query} with [MASK] category for {query2} [SEP] [SEP]"

    def __init__(self, categories: list):
        # load pretrained BERT
        self.categories = categories

        # Load pre-trained model tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForMaskedLM.from_pretrained('bert-base-uncased')

        self.model.eval()

    def predict(self, query: str) -> list:
        """
        :param query: input query from user
        :return: ordered list of categories with prediction score
        """

        category_sentences = self.pattern.format(query=query, query2=query)

        # Tokenize and include mask
        tokenized_sentences = self.tokenizer.tokenize(category_sentences)
        masked_index = tokenized_sentences.index('[MASK]')
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_sentences)

        separator_index = tokenized_sentences.index('[SEP]')
        segments_ids = [0] * (separator_index + 1) + [1] * (
                len(indexed_tokens) - separator_index - 1)

        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        with torch.no_grad():
            outputs = self.model(tokens_tensor, token_type_ids=segments_tensors)
            predictions = outputs[0]
            predictions_softmax = softmax(predictions[0, masked_index], dim=0)
            # predictions_softmax = predictions[0, masked_index]


        prob_per_category = []

        for category in self.categories:
            category_vocab_id = self.tokenizer.convert_tokens_to_ids(category)
            category_score = predictions_softmax[category_vocab_id].item()
            prob_per_category.append((category, category_score))

        return sorted(prob_per_category, key=lambda cat_tup: cat_tup[1], reverse=True)


# Tokenize input
queries = [
    "men shoes",
    "summer dress",
    "shirt for party",
]

categories = ["size", "department", "style", "sleeve", "neckline", "occasion", "pattern"]

category_predictor_ns = CategoryPredictorMaskedWord(categories)

for query in queries:
    print(f"{query}:    {category_predictor_ns.predict(query)}")











