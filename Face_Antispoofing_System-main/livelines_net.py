import pickle
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
import os
from random import randint
import numpy as np
from tensorflow.keras.models import model_from_json
root_dir = os.getcwd()
classColors = [(randint(0,255), randint(0,255), randint(0,255)) for _ in range(91)]
# Load Face Detection Model
face_cascade = cv2.CascadeClassifier(r"C:\Users\hp\Downloads\smoofing\Face_Antispoofing_System-main\models\haarcascade_frontalface_default.xml")
# Load Anti-Spoofing Model graph
json_file = open(r'C:\Users\hp\Downloads\smoofing\Face_Antispoofing_System-main\antispoofing_models\antispoofing_model.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load antispoofing model weights 
model.load_weights(r'C:\Users\hp\Downloads\smoofing\Face_Antispoofing_System-main\antispoofing_models\antispoofing_model.h5')
with open(r"C:\Users\hp\Downloads\project_computervision\project_computervision\models\trained_reconizer&embeddings.pickle","rb") as pf:
    data = pickle.load(pf)
faceRecognizer = data["recognizer"]
labelEncoder = data["labelencoder"]
FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN = (0,255,0)
print("[COMPLETED] models has been Loaded.")
class FaceDetector:
    def __init__(self, cuda=None):
        modelparamters = r"C:\Users\hp\Downloads\project_computervision\project_computervision\models\res10_300x300_ssd_iter_140000_fp16.caffemodel" # model paramters
        modelconfig= r'C:\Users\hp\Downloads\project_computervision\project_computervision\models\res10_structure_paramters.prototxt' # model layers configs
        self.facenetModel = DiNet(modelconfig,modelparamters, cuda=cuda) # load the pre-trained model
        self.detectedfaces = []
        self.AppendFace = self.detectedfaces.append
    def detect(self, frame, threshold=0.6, Width=640, Height=480):
        self.detectedfaces.clear()
        imBlob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)), 1.0,
            size=(300,300),
            mean=(104.0, 177.0, 123.0))
        predictedfaces = self.facenetModel.passInput(imBlob)
        for i in range(predictedfaces.shape[2]):
            preds_confdence = predictedfaces[0,0,i,2]
            if preds_confdence >= threshold:
                boundingBox = predictedfaces[0,0,i,3:7] * np.array([Width,Height,Width,Height])
                (x, y, boxwidth, boxheight) = boundingBox.astype('int')
                self.AppendFace([x,y,boxwidth,boxheight])
        return self.detectedfaces
##################################################################
class FaceEmbedder:
    def __init__(self, cuda=None):
        openfacepath=r"C:\Users\hp\Downloads\project_computervision\project_computervision\models\nn4.small2.v1.t7"
        self.embedder = DiNet(modelpath=openfacepath, framework="PyTorch", cuda=cuda)
    def embedFace(self, faceRoI):
        ''':faceRoI: RGBimage'''
        imBlob = cv2.dnn.blobFromImage(faceRoI, 1.0/255,
            (96, 96), (0,0,0),swapRB=True, crop=False)
        return self.embedder.passInput(imBlob)
class DiNet:
    def __init__(self, modelpath, modelconfig=None, framework="Caffe", cuda=False):
        self.net = cv2.dnn.readNet(model=modelpath,config=modelconfig,
            framework=framework)
        if cuda: self.useCuda()
    def useCuda(self): 
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    def passInput(self ,netInput):
        self.net.setInput(netInput)
        return self.net.forward()
    def setInputSize(self, Width, Height):
        self.net.setInputSize(Width, Height)
    def setInputScale(self, Scale: float):
        self.net.setInputScale(Scale)
    def setInputMean(self, R, G, B):
        self.net.setInputMean(R, G, B)
    def setInputSwapRB(self, boolean: bool):
        self.net.setInputSwapRB(boolean)
face_emmedd=FaceEmbedder()
#facedembedder = utils.FaceEmbedder()
print("Model loaded from disk")
# video.open("http://192.168.1.101:8080/video")
# vs = VideoStream(src=0).start()
# time.sleep(2.0)
import random
def faceRecognize(frame) -> str:
    """ function that process the frame and predict faces """
    faces = ["yousef", "mohamed", "ali"]
    return random.choice(faces)
attendance_table = {}
import datetime
from datetime import datetime
def markaddtend(name):
    with open('sheet.csv','r+') as f:
        datatimelist=f.readlines()
        names=[]
        for Line in datatimelist:
            entry=Line.split(',')
            names.append(entry[0])
            if name not in names:
                currunt=datetime.now()
                datastring=currunt.strftime('%H:%M:%S')
                #name_data.append([name,datastring])
                f.writelines(f'{name},{datastring}\n')
video = cv2.VideoCapture(0)
breakcounter=0
attendance={}
while True:
    try:
        ret,frame = video.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in faces:  
            face = frame[y-5:y+h+5,x-5:x+w+5]
            faceVecId = face_emmedd.embedFace(face)
            predictedsvmnames = faceRecognizer.predict_proba(faceVecId)[0]
            nameId_index = np.argmax(predictedsvmnames)
            faceName = labelEncoder.classes_[nameId_index]
            print(faceName)
            #reconized_face=faceRecognize(faceName)
            #if reconized_face  not in attendance_table:
            #   attendance_table[reconized_face] = (True, datetime.datetime.now())    
            # logic to break the loop
            #print(faceName)
            #cv2.putText(frame, faceName, (x, y), FONT, 0.5, GREEN , 2)
            resized_face = cv2.resize(face,(160,160))
            resized_face = resized_face.astype("float") / 255.0
            # resized_face = img_to_array(resized_face)
            resized_face = np.expand_dims(resized_face, axis=0)
            # pass the face ROI through the trained liveness detector
            # model to determine if the face is "real" or "fake"
            preds = model.predict(resized_face)[0]
            #print(preds)
            if preds> 0.5:
                label = 'spoof'
                cv2.putText(frame, label+" "+faceName , (x,y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,255),1)
                #print(faceName)
            else:
                label = 'real'
                cv2.putText(frame, label+" "+faceName, (x,y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
                cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,255),1)  
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    except Exception as e:
        pass
video.release()        
cv2.destroyAllWindows()
print(attendance) 