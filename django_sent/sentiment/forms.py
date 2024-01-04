from django import forms


class ReviewForm(forms.Form):
    review = forms.CharField(widget=forms.Textarea, label="Review", max_length=500,
                             help_text="Review in English. Up to 500 characters.")
