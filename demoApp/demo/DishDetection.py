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
from demoApp.settings import MEDIA_ROOT

global update_counter

#TEST_URL="C:/Users/SHUBHAM PANDEY/GRAFFERSID_PROJECTS/SWEETS_RECOG/demoApp/test/"

def display_sweets():
    IMG_SIZE=50
    BASE = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("BASE_DIR IS ",BASE_DIR)
    print("BASE IS",BASE)



    try:
        new_model=load_model(os.path.join(BASE,'latest_one.h5'))
        print("shubham pandey")
    except Exception as e:
        print("the Exception is ",e)
    
    # camera = cv2.VideoCapture(0)
    # for i in range(22):
    #     return_value, image = camera.read()
    #     cv2.imshow('frame',image)
    #     cv2.imwrite('C:/Users/SHUBHAM PANDEY/GRAFFERSID_PROJECTS/SWEETS_RECOG/demoApp/test/'+str(i)+'.jpg', image)
    #     if(i==21):
    #         cv2.imwrite('C:/Users/SHUBHAM PANDEY/GRAFFERSID_PROJECTS/SWEETS_RECOG/demoApp/media/'+str(i)+'.jpg', image)
    #     key = cv2.waitKey(300)
    # # Stop video
    # camera.release()

    # # Close all started windows
    # cv2.destroyAllWindows()


    testing_X=[]
    IMG_SIZE=50
    #TEST_URL=os.path.join(BASE,"media")

    for img in tqdm(os.listdir(MEDIA_ROOT)):
        abcd=os.path.join(MEDIA_ROOT,img)
        img=cv2.resize(cv2.imread(abcd,cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE),interpolation = cv2.INTER_LINEAR)
        testing_X.append(np.array(img))
        
    testing_X=np.array(testing_X)
    testing_X=testing_X.astype('float32')
    testing_X=testing_X/ 255
    testing_X=testing_X.reshape(-1,50,50,1)
    print(testing_X.shape)
    print(new_model.predict(testing_X))

    predicted_classes=[]
    try:
        predicted_classes.append(new_model.predict(testing_X))
    except Exception as e:
        print("Tjsi is Exception: ", e)
    print(predicted_classes)


    BASE = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(BASE,"test.txt"), "rb") as fp:  # Unpickling
        b = pickle.load(fp)
        print(b)



    w=[]
    for i in range(0,4):
        abcd=predicted_classes[0][i]
        result=np.where(abcd == np.amax(abcd))
        w.append(result[0])

    newSet = np.unique(w)
    print(newSet)
    s=len(newSet)
    print(s)
    count = [0]*s
    i = 0
    for y in newSet:
        for x in w:
            if x==y:
                count[i] = count[i] +1
        i = i + 1
    
    print(count)

    maxpos=count.index(max(count))

    t=newSet[maxpos]

    print(t)


    final=b[t]


    print(final)
        

    filelist = [ f for f in os.listdir(MEDIA_ROOT) if f.endswith(".jpg") ]

    for f in filelist:
        os.remove(os.path.join(MEDIA_ROOT, f))


    filelist = [ f for f in os.listdir(MEDIA_ROOT) if f.endswith(".jpeg") ]

    for f in filelist:
        os.remove(os.path.join(MEDIA_ROOT, f))

    filelist = [ f for f in os.listdir(MEDIA_ROOT) if f.endswith(".cms") ]

    for f in filelist:
        os.remove(os.path.join(MEDIA_ROOT, f))


    #filelist = [ f for f in os.listdir(TEST_URL) if f.endswith(".png") ]

    #for f in filelist:
        #os.remove(os.path.join(TEST_URL, f))


    print("ALL FILES ARE REMOVED SUCCESSFULLY")
    sweet_obj = Sweets.objects.filter(name=final)
    print('sweeeeeeee',sweet_obj)
    attr_obj = AttrValue.objects.filter(sweet_name=sweet_obj[0])
    print('attrrrrrrr', attr_obj)
    # = {'Calories': 'fruit', 'Carbohydrate': 'vegetable', 'Protine': 'dessert','Sugar':'sweet'}
    #BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    #imgs =os.path.join(BASE_DIR,"media")+"\\Cake21.jpg"
    #print(imgs)

    payload = {
        'sweet': sweet_obj[0].name,
        'img': "/media/3.png",
        'attrbutes': attr_obj 
    }
    return payload



def label_img(img):
    word_label=img.split('.')[-2]
    print(word_label)
    a=len(b)
    returning_list = [0] *a
    for i in b:
        if i in word_label:
            q=b.index(i)
            print("q is",q)
            returning_list[q]=1
            return returning_list



def trainModel(keyword):

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    temp=os.path.join(BASE_DIR,"temp/")
    print("TEMP IS",temp)
    TRAIN_DIR=os.path.join(BASE_DIR,"train/")
    print("TRAIN_DIR IS",TRAIN_DIR)

    print(keyword)

    update_counter=0
    for q in range(0,6):
        if q==0:
            abc=10
            lmn=300
        if q==1:
            abc=310
            lmn=600
        if q==2:
            abc=610
            lmn=900
        if q==3:
            abc=910
            lmn=1200
        if q==4:
            abc=1210
            lmn=1500
        if q==5:
            abc=1510
            lmn=1800
        for page_no in range(abc, lmn, 20):
            slp(10)
            r = requests.get('https://www.google.co.in/search?q=' + keyword +'&gbv=1&tbm=isch&start=' + str(page_no))
            img_urls = html.fromstring(r.content).xpath('//table[@class="images_table"]//img/@src')
            for i, img_url in enumerate(img_urls):
                print(img_url)
                j=update_counter
                f = open((temp)+ str(j) + ".jpg", "wb")
                f.write(requests.get(img_url).content)
                j=j+1
                update_counter=j


    i=0
    for filename in os.listdir(temp): 
        dst =str(keyword) +str(i) + ".jpg"
        src =(temp)+ filename 
        dst =(TRAIN_DIR)+ dst 

        os.rename(src, dst) 
        i += 1
    

    
    IMG_SIZE=100

    BASE = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(BASE,"test.txt"), "rb") as fp:  # Unpickling
        b = pickle.load(fp)
        b.append(str(keyword))
        print(b)

    with open("test.txt", "wb") as fp:   #Pickling
        pickle.dump(b, fp)


    training_X=[]
    training_Y=[]
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label=label_img(img)
        path=os.path.join(TRAIN_DIR,img)
        img=cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE),interpolation = cv2.INTER_LINEAR)
        training_X.append(np.array(img))
        training_Y.append(np.array(label))

    training_Y=np.array(training_Y)

    training_X=np.array(training_X)
    training_X = training_X.astype('float32')

    train_X,valid_X,train_label,valid_label = train_test_split(training_X, training_Y, test_size=0.2, random_state=13)

    train_X = train_X.reshape(-1, 100,100, 1)

    valid_X = valid_X.reshape(-1, 100,100, 1)

    batch_size = 64
    epochs = 20
    a=len(b)

    fashion_model = Sequential()
    fashion_model.add(Conv2D(64, kernel_size=(3, 3),activation='linear',input_shape=(100,100,1),padding='same'))
    fashion_model.add(LeakyReLU(alpha=0.01))
    fashion_model.add(MaxPooling2D((2, 2),padding='same'))
    fashion_model.add(Dropout(0.15))
    fashion_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
    fashion_model.add(LeakyReLU(alpha=0.01))
    fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    fashion_model.add(Dropout(0.15))
    fashion_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
    fashion_model.add(LeakyReLU(alpha=0.01))                  
    fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    fashion_model.add(Dropout(0.2))
    fashion_model.add(Flatten())
    fashion_model.add(Dense(128, activation='linear'))
    fashion_model.add(LeakyReLU(alpha=0.01)) 
    fashion_model.add(Dropout(0.15))
    fashion_model.add(Dense(a, activation='softmax'))

    fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

    abcd_trained=fashion_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))

    fashion_model.save('Pizza_Rasmalai.h5')