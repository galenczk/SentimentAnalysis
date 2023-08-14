from transformers import BertTokenizer
import torch
from torch import nn


class TextClassificationModel(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_classes):
        super(TextClassificationModel, self).__init__()

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        vocab_size = len(self.tokenizer)

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
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
        attention_weights = torch.softmax(masked_output, dim=1)
        attention_output = torch.sum(attention_weights * output, dim=1)
        output = self.fc(attention_output)
        return output.squeeze()


class myModelSentiment:
    def __init__(self) -> None:

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        embedded_dim = 512
        hidden_dim = 256

        self.model = TextClassificationModel(
            embed_dim=embedded_dim, hidden_dim=hidden_dim,
            num_classes=1)

    def process_input(self, text):
        text = text.lower()

        tokenized_input = self.tokenizer.tokenize(text)

        MAX_LEN = 256
        input_id = self.tokenizer.convert_tokens_to_ids(tokenized_input)
        input_id = torch.tensor(input_id[:MAX_LEN] + [0] * (MAX_LEN - len(input_id)))
        attention_mask = torch.tensor([1 if token > 0 else 0 for token in input_id])

        return input_id, attention_mask

    def analyze(self, text):

        input_id, attention_mask = self.process_input(text)

        output = self.model(input_id, attention_mask)

        return output


model = TextClassificationModel(512, 256, 1)

analyzer = myModelSentiment()

print(analyzer.analyze("bad worst worse sad evil bad very bad"))