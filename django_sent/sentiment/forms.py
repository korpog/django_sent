from django import forms


class ReviewForm(forms.Form):
    review = forms.CharField(label="Review", max_length=500)