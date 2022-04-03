import os
import tensorflow
import numpy as np
import cv2
from tensorflow import keras
def getCalssName(classNo):
    if   classNo == 0: return 'Speed Limit 20 km/h'
    elif classNo == 1: return 'Speed Limit 30 km/h'
    elif classNo == 2: return 'Speed Limit 50 km/h'
    elif classNo == 3: return 'Speed Limit 60 km/h'
    elif classNo == 4: return 'Speed Limit 70 km/h'
    elif classNo == 5: return 'Speed Limit 80 km/h'
    elif classNo == 6: return 'End of Speed Limit 80 km/h'
    elif classNo == 7: return 'Speed Limit 100 km/h'
    elif classNo == 8: return 'Speed Limit 120 km/h'
    elif classNo == 9: return 'No passing'
    elif classNo == 10: return 'No passing for vechiles over 3.5 metric tons'
    elif classNo == 11: return 'Right-of-way at the next intersection'
    elif classNo == 12: return 'Priority road'
    elif classNo == 13: return 'Yield'
    elif classNo == 14: return 'Stop'
    elif classNo == 15: return 'No vechiles'
    elif classNo == 16: return 'Vechiles over 3.5 metric tons prohibited'
    elif classNo == 17: return 'No entry'
    elif classNo == 18: return 'General caution'
    elif classNo == 19: return 'Dangerous curve to the left'
    elif classNo == 20: return 'Dangerous curve to the right'
    elif classNo == 21: return 'Double curve'
    elif classNo == 22: return 'Bumpy road'
    elif classNo == 23: return 'Slippery road'
    elif classNo == 24: return 'Road narrows on the right'
    elif classNo == 25: return 'Road work'
    elif classNo == 26: return 'Traffic signals'
    elif classNo == 27: return 'Pedestrians'
    elif classNo == 28: return 'Children crossing'
    elif classNo == 29: return 'Bicycles crossing'
    elif classNo == 30: return 'Beware of ice/snow'
    elif classNo == 31: return 'Wild animals crossing'
    elif classNo == 32: return 'End of all speed and passing limits'
    elif classNo == 33: return 'Turn right ahead'
    elif classNo == 34: return 'Turn left ahead'
    elif classNo == 35: return 'Ahead only'
    elif classNo == 36: return 'Go straight or right'
    elif classNo == 37: return 'Go straight or left'
    elif classNo == 38: return 'Keep right'
    elif classNo == 39: return 'Keep left'
    elif classNo == 40: return 'Roundabout mandatory'
    elif classNo == 41: return 'End of no passing'
    elif classNo == 42: return 'End of no passing by vechiles over 3.5 metric tons'
def preprocessing(img):
    #grayscale
    gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #equalize histogram
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))
    hist_equal_image=clahe.apply(gray_image)
    #normalize
    normalized_image = np.array(hist_equal_image, dtype=np.float32) / 255
    #img = img / 255
    return normalized_image

#load model
new_model = tensorflow.keras.models.load_model('my_model.h5')
new_model.load_weights("VGG_GermanSigns_classification.h5")
#new_model.summary()

#read and pre process
path="F:/GitHub Projects/Traffic-Sign-Recognition/TestData/"
img_original = cv2.imread(path+"5.png")
img = np.asarray(img_original)
img = cv2.resize(img, (32, 32))
img = preprocessing(img)
img = img.reshape(1, 32, 32, 1)
#predict class
prediction =new_model.predict(img)
prediction_class=np.argmax(prediction)
probabilityValue=np.amax(prediction)
print(prediction)
print(prediction_class)
print(probabilityValue)

#show result
threshold = 0.75
font = cv2.FONT_HERSHEY_SIMPLEX
img_original=cv2.resize(img_original,(800,800))
cv2.putText(img_original, "CLASS: " , (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
cv2.putText(img_original, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

cv2.putText(img_original,str(prediction_class)+" "+str(getCalssName(prediction_class)), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
cv2.putText(img_original, str(round(probabilityValue*100,2) )+"%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
cv2.imshow("Result", img_original)

cv2.waitKey(0)