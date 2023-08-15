from transformers import BertTokenizer
import torch
from torch import nn


# NN copied over from myModel
class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, hidden_dim=256, num_classes=1, num_lstm_layers=2, dropout_prob=0.2):
        super(TextClassificationModel, self).__init__()
        # Embedding layers convert word values into more complicated vectors.
        # This is a type of feature extraction or definition.
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Long short-term memory layer. An abstracted layer that contains
        # several gates that perform different tasks on sequential data.
        # Common in natural language processing tasks.
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_lstm_layers,
                            bidirectional=True, dropout=dropout_prob)

        # Fully connected layer, maps NN output thusfar to final classification.
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

        # Initialize the NN with random weights and biases.
        self.init_weights()

    def init_weights(self):
        # Sets weights and biases of NN layers to random values to start.
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, attention_mask):
        # Defines the way data moves through the NN. Utilizes the attention mask
        # that was developed for each sentence in the tokenizing step.

        # Pass text through the embedding layer.
        embedded = self.embedding(text)

        # Pass text through lstm layers.
        output, (hidden, cell) = self.lstm(embedded)

        # Apply attention mask to output of embedding and lstm layers.
        # Output has dimension of 512, so unsqueeze to match this.
        masked_output = output * attention_mask.unsqueeze(-1)
        attention_output = torch.sum(masked_output * output, dim=1)

        # Pass attention weighted output through fc layer to produce final logits.
        output = self.fc(attention_output)
        return output.squeeze()


class myModelSentiment:
    def __init__(self) -> None:

        # Ppre-built tokenizer from HuggingFace.
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Init model.
        self.model = TextClassificationModel(len(self.tokenizer))

        # Load previously trained parameters.
        self.model.load_state_dict(torch.load("modelCheckpoint.pt"))

    def process_input(self, text):
        # Standardize input to lowercase.
        text = text.lower()

        # Apply tokenizer.
        tokenized_input = self.tokenizer.tokenize(text)

        # Create input tensor with text input. Process similar to myModel.
        MAX_LEN = 256
        input_id = self.tokenizer.convert_tokens_to_ids(tokenized_input)
        input_id = torch.tensor(
            input_id[:MAX_LEN] + [0] * (MAX_LEN - len(input_id)))
        attention_mask = torch.tensor(
            [1 if token > 0 else 0 for token in input_id])

        return input_id, attention_mask

    def analyze(self, text):

        # Process input text.
        input_id, attention_mask = self.process_input(text)

        # Call model to classify.
        output = self.model(input_id, attention_mask)

        return output
