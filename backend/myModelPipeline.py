from transformers import BertTokenizer
import torch
from torch import nn


# NN copied over from myModel notebook.
class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size=30522, embed_dim=512, hidden_dim=256, num_classes=1, num_lstm_layers=4, dropout_prob=0.2):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_lstm_layers, bidirectional=True, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, attention_mask):
        embedded = self.embedding(text)       
        output, (hidden, cell) = self.lstm(embedded)
        masked_output = output * attention_mask.unsqueeze(-1)
        attention_output = torch.sum(masked_output * output, dim=1)
        output = self.fc(attention_output)
        return output.squeeze()


class myModelSentiment:
    def __init__(self) -> None:

        # Pre-built tokenizer from HuggingFace.
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Init model.
        self.model = TextClassificationModel(len(self.tokenizer))

        # Load previously trained parameters.
        self.model.load_state_dict(torch.load("modelCheckpoint_4LSTM_10epochs.pt"))

    def process_input(self, text):
        # Standardize input to lowercase.
        text = text.lower()

        # Apply tokenizer.
        tokenized_input = self.tokenizer.tokenize(text)

        # Create input tensor with text input. Process similar to myModel.
        MAX_LEN = 512
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

        # Round output to either 1 or 0 (postive or negative)
        rounded_output = torch.round(torch.sigmoid(output))

        return rounded_output
    