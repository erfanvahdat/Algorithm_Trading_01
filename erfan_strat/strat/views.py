from django.shortcuts import render
from django.http import HttpResponse
from .tasks import *
# Create your views here.

def index(request):
    data=cryptoForex.deley()
    return HttpResponse("done")