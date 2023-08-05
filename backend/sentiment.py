"""https://huggingface.co/learn/nlp-course/en/chapter2/2?fw=pt"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class Sentiment:
    def __init__(self) -> None:
        # A checkpoint is a dictionary of pretrained weights and biases that is
        # ready to be loaded into a model architecture.
        self.checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

        # A model generally refers to a neural network architecture and its
        # associated weights and biases.  Here we are loading an architecture
        # and loading it with the weights and biases above.
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.checkpoint)

        # A tokenizer splits input into easily digestible pieces for the model
        # to intake.
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)

    def analyze(self, inputs):

        # Here we apply the tokenizer to the input.
        tokenized_input = self.tokenizer(inputs, return_tensors="pt")

        # Here we run the tokenized input through the model to get the
        # probability distribution of the text having a positive or negative
        # sentiment.
        output = self.model(**tokenized_input)

        # The raw output of the model is non-normalized. This step results in
        # the probabilities that will ultimately be reported.
        predictions = torch.nn.functional.softmax(output.logits, dim=1)

        return predictions[0].tolist()
