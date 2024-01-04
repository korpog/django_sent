from django.urls import path

from . import views

app_name = "sentiment"
urlpatterns = [
    path("", views.index, name="index"),
    path("result/", views.result, name="result"),
]