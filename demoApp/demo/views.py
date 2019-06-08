
import cv2, os ,pickle, keras, sqlite3, requests, Augmentor, sys
import numpy as np
from PIL import Image
from tqdm import tqdm
from keras.models import model_from_json
from django.shortcuts import render
from IPython.core.display import HTML
from keras.models import load_model
import os.path
from .models import *
from lxml import html
from time import sleep as slp
from sklearn.model_selection import train_test_split
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from .DishDetection import display_sweets
from .DishDetection import trainModel
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
import json
import base64
import random
import string
from django.core.files.base import ContentFile
from django.http import JsonResponse
from demoApp.settings import MEDIA_ROOT

BASE = os.path.dirname(os.path.abspath(__file__))

def index(request):
    context ={}
    if request.method == 'POST':
       # print(request.POST)
        #x=request.POST['files']
        #x = json.loads(x)
        #print(x)
        data = display_sweets()
        print(data)
        context['data']= data
        #return HttpResponse({'data':'work'})
    return render(request, 'index.html', context)


@csrf_exempt
def upload(request):
    context ={}
    k=0
    #TEST_URL='C:/Users/SHUBHAM PANDEY/GRAFFERSID_PROJECTS/SWEETS_RECOG/demoApp/media/'
    filelist = [ f for f in os.listdir(MEDIA_ROOT) if f.endswith(".png") ]

    from .models import Images
    #instance = Images.objects.get()
    for x in Images.objects.all():
        x.delete()

 #   Images.objects.get(id=1).values('file').delete()
    # Deletes all sized images and cache entries associated with instance.imag

    for f in filelist:
        os.remove(os.path.join(MEDIA_ROOT, f))
    if request.method == 'POST':
        data = request.body
        data = json.loads(data[0:len(data)])
        #c = Sweets.objects.get(pk=1)
        #print(data)
        temp = len('data:image/jpeg;base64,')
        for d in data:
            print("D LENGTH PREVIOUS",len(d))
            format, imgstr = d.split(';base64,') 
            ext = format.split('/')[-1] 

            img = ContentFile(base64.b64decode(imgstr), name=str(k)+"." + ext)
            #d += "=" * ((4 - len(d) % 4) % 4)
            #print("D LENGTH AFTER",len(d)) 
            #d = d[temp:len(d)]

            #imgdata = base64.b64decode(d[temp:len(d)]+"===")
            #filename = str(k)+'.jpg'  # I assume you have a way of picking unique filenames
            #with open('media/'+filename, 'wb') as f:
                #f.write(img)
                #file_name = "'myphoto." + ext
                #f.save(file_name, img, save=True)
            i = Images.objects.create(file=img)
            i.save()
            k=k+1
        print("MY NAME IS SHUBHAM PANDEY, I AM AN INTERN AT GRAFFERSID")
        context ={ 'data':"Image Captured",
                'abcdef':"Go to Home page to Analyse your Dish."
        }
        return JsonResponse({'data': 'Success'})
    return render(request, 'Upload.html')
 
    
def addSweet(request):
    context ={}
    if request.method == 'POST':
        s = Sweets(name=request.POST['sweet'])
        s.save()
        a1 = AttrValue(sweet_name=s, attr_name='Carbohydrate', value=request.POST['carbo'])
        a1.save()
        a1 = AttrValue(sweet_name=s, attr_name='Fat', value=request.POST['Fat'])
        a1.save()
        a1 = AttrValue(sweet_name=s, attr_name='Calories', value=request.POST['Calories'])
        a1.save()
        a1 = AttrValue(sweet_name=s, attr_name='Protein', value=request.POST['protein'])
        a1.save()

        trainModel(s.name)
        context ={ 'data':"Data Saved",
                'abcdef':"Training the System to automatically detect the dish. Please wait till the magic completes."
        }
    return render(request, 'form.html', context)
    





