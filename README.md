# Coding2-Coding2-Tensorflow-ObjectDetection-HandGesture

#### What is public speaking phobia?

#### Public speaking fear and anxiety is a form of social phobia called glossophobia  – an intense and irrational fear of being judged by others when speaking in front of them – or of making mistakes, being embarrassed or humiliated in such situations – causing dread, panic and avoidance.

#### Sufferers recognise that their fear and anxiety is excessive or unreasonable but they feel powerless to do anything to change their responses. So the feared situations – such as presentations, wedding speeches, meetings or even one-to-ones – are avoided or else endured with intense anxiety or distress.


After study term 2 in CCI i found that i have high curiosity and passion in OpenCV. So i start explore how does in final project.


For more graphic details: https://www.notion.so/Coding2-Final-61efd8ab0c8a4891b369fa37d4011cc2


This README file basicly explain the whole work, and I divide the project process into two parts：

PART 1: Image Collection

PART 2: Training and Object Detection



(Note: with huge greatful to those person who develops valuable open source code, and also i learned a lot from Nicholas Renotte www.nicholasrenotte.com.)


## PART 1: Image Collection

### This part is eqasier than part 2. I complete the work in: 
### 1 Clone repo
### 2 Collect Images(and code)
### 3 Setup labelImg
### 4 Label Images

!pip install opencv-python

import cv2 
import uuid
import os
import time

labels = ['hello','thumbsup', 'thumbsdown', 'thanku', 'missu']
number_imgs = 15

IMAGES_PATH = os.path.join('Tensorflow', 'workspace', 'images', 'collectedimages')

if not os.path.exists(IMAGES_PATH):
    if os.name == 'posix':
        !mkdir -p {IMAGES_PATH}
    if os.name == 'nt':
         !mkdir {IMAGES_PATH}
for label in labels:
    path = os.path.join(IMAGES_PATH, label)
    if not os.path.exists(path):
        !mkdir {path}
for label in labels:
    cap = cv2.VideoCapture(0)
    print('Collecting images for {}'.format(label))
    time.sleep(5)
    for imgnum in range(number_imgs):
        print('Collecting image {}'.format(imgnum))
        ret, frame = cap.read()
        imgname = os.path.join(IMAGES_PATH,label,label+'.'+'{}.jpg'.format(str(uuid.uuid1())))
        cv2.imwrite(imgname, frame)
        cv2.imshow('frame', frame)
        time.sleep(2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()







## PART 2: Training and Object Detection

### This part is very tough to me. i met lots of different errors. Basically cause of wrong syntax, wrong file path and different version.






