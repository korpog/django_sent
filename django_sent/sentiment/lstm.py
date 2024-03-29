import numpy as np
import re
import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedding_dim = 64
output_dim = 1

class LSTM_NN(nn.Module):
    def __init__(self, no_layers, vocab_size, hidden_dim, emdedding_dim, drop_prob=0.5):
        super(LSTM_NN, self).__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.no_layers = no_layers
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=emdedding_dim, hidden_size=self.hidden_dim,
                            num_layers=no_layers, batch_first=True)
        self.droput = nn.Dropout(0.3)
        self.fc = nn.Linear(self.hidden_dim, output_dim)
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size(0)
        embeddings = self.embedding(x)
        lstm_out, hidden = self.lstm(embeddings, hidden)

        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        out = self.droput(lstm_out)
        out = self.fc(out)

        sig_out = self.sig(out)
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]  # get last batch

        return sig_out, hidden

    def init_hidden(self, batch_size):
        # create two new tensors with sizes n_layers x batch_size x hidden_dim
        h0 = torch.zeros((self.no_layers, batch_size,
                         self.hidden_dim)).to(device)
        c0 = torch.zeros((self.no_layers, batch_size,
                         self.hidden_dim)).to(device)
        hidden = (h0, c0)
        return hidden
    
def preprocess_string(s):
    # remove non-word characters, whitespaces and digits
    re.sub(r"[^\w\s]", '', s)
    re.sub(r"[\s+]", '', s)
    re.sub(r"[\d]", '', s)

    return s

def padding(sentences, length):
    features = np.zeros((len(sentences), length), dtype=int)
    for i, review in enumerate(sentences):
        if len(review) != 0:
            features[i, -len(review):] = np.array(review)[:length]
    return features

def predict_text(text, model, vocab):
        word_seq = np.array([vocab[preprocess_string(word)] for word in text.split()
                         if preprocess_string(word) in vocab.keys()])
        word_seq = np.expand_dims(word_seq,axis=0)
        pad =  torch.from_numpy(padding(word_seq,500))
        inputs = pad.to(device)
        batch_size = 1
        h = model.init_hidden(batch_size)
        h = tuple([each.data for each in h])
        output, h = model(inputs, h)
        outcome = "positive" if output.item() > 0.5 else "negative"
        return outcome