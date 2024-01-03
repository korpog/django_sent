import os
import pickle
import torch
from . import lstm
from django.shortcuts import render
from django_sent.settings import STATIC_ROOT

no_layers = 2
vocab_size = 1001
embedding_dim = 64
output_dim = 1
hidden_dim = 128

PATH_MODEL = os.path.join(STATIC_ROOT, "sentiment/model_state_dict.pt")
PATH_VOCAB = os.path.join(STATIC_ROOT, "sentiment/vocabulary.pickle")

model = lstm.LSTM_NN(no_layers, vocab_size, hidden_dim,
                embedding_dim, drop_prob=0.5)
with open(PATH_VOCAB, 'rb') as file:
    vocab = pickle.load(file)
model.load_state_dict(torch.load(PATH_MODEL))
model.eval()
text = "Absolutely epic! The scale is truly mind blowing. Every filmmaking aspect is beyond incredible. Especially the score, set design & dialogue. The emotional weight & amazing battles seal it."

pred = lstm.predict_text(text, model, vocab)
print(pred)

def index(request):
    return render(request, "sentiment/index.html")



