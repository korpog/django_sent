import os
import pickle
import torch
from .lstm import LSTM_NN, predict_text
from .forms import ReviewForm
from django.shortcuts import render
from django_sent.settings import STATIC_ROOT

no_layers = 2
vocab_size = 1001
embedding_dim = 64
output_dim = 1
hidden_dim = 128

PATH_MODEL = os.path.join(STATIC_ROOT, "sentiment/model_state_dict.pt")
PATH_VOCAB = os.path.join(STATIC_ROOT, "sentiment/vocabulary.pickle")

model = LSTM_NN(no_layers, vocab_size, hidden_dim,
                embedding_dim, drop_prob=0.5)
with open(PATH_VOCAB, 'rb') as file:
    vocab = pickle.load(file)
model.load_state_dict(torch.load(PATH_MODEL))
model.eval()

def index(request):
    return render(request, "sentiment/index.html")

def result(request):
    prediction = ""
    text = ""
    if request.method == "POST":
        form = ReviewForm(request.POST)
        print(form)
        if form.is_valid():
            text = form.cleaned_data["review"]
            prediction = predict_text(text, model, vocab)
   
    else:
       form = ReviewForm()

    context = {
        'form': form,
        'text': text,
        'prediction': prediction
    }

    return render(request, 'sentiment/result.html', context)





