import os
import torch
from lstm import LSTM_NN
from django.shortcuts import render
from django_sent.settings import STATIC_ROOT

no_layers = 2
vocab_size = 1001
embedding_dim = 64
output_dim = 1
hidden_dim = 128

PATH = os.path.join(STATIC_ROOT, "sentiment/model_state_dict.pt")

model = LSTM_NN(no_layers, vocab_size, hidden_dim,
                embedding_dim, drop_prob=0.5)
model.load_state_dict(torch.load(PATH))
model.eval()


