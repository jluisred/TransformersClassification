# TransformersClassification

This package implements different methods of classification using Transformers. 
It intends to demonstrate the capabilities or pre-trained BERT transformer, with no additional
re-training, versus fine-tuning on a small set of samples.

1) Based on Next Sentence Prediction, over a pre-trained vanilla BERT
  See: `TransformersClassification/src/classifiers/category_prediction_next_sentence.py`
2) Based on Masked Word Prediction, over a pre-trained vanilla BERT
  See: `TransformersClassification/src/classifiers/category_prediction_mask.py`
3) Fine-tuned on downstream task.
  See: `TransformersClassification/src/classifiers/category_classifier_prediction.py`
  See: `TransformersClassification/src/classifiers/category_classifier_test.py`
  See: `TransformersClassification/src/classifiers/category_classifier_train.py`

In addition, we also include a proof of concept for semantic sentence similarity based on:
  See: `TransformersClassification/src/classifiers/category_prediction_mask.py`
