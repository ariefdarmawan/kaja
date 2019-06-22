from helper import config
import sys
import glob
import cv2
import numpy as np
import face_recognition as fr
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from keras.models import load_model

cfg = config.Config()
trains = {}

cfg.testfolder = "./images/test/*"
cfg.trainfolder = "./images/train/*"
cfg.dumpfolder = "./images/temp/"
cfg.model = "./models/fer-03.hdf5"
cfg.input_shape = (48,48)
#cfg.facefolder = "./images/faces/"

# run the train first
for imgf in glob.glob(cfg.trainfolder):
    fileid = Path(imgf).name.split(".")[0]
    fimg = fr.load_image_file(imgf)
    locs = fr.face_locations(fimg)
    fencs = fr.face_encodings(fimg, known_face_locations=locs)
    number_of_faces = len(locs)
    print("train {}, found {} images".format(imgf, number_of_faces))

    locindex = 0
    for face_location in locs:
        fenc = fencs[locindex]
        trains[fileid + "_" + str(locindex)] = fenc
        locindex +=1

print("trains {} faces".format(len(trains)))

# run test
train_encodings = list(trains.values())
train_keys = list(trains.keys())
#print("keys are: {}".format(train_keys))

# emotion recod
emotions = ["fear","disgust","neutral", "excited", "sad", "surprised","happy"]
model = load_model(cfg.model)

fnt = ImageFont.truetype("./assets/Hack-Regular.ttf", 12)
testindex = 0
for imgf in glob.glob(cfg.testfolder):
#print("test image {}".format(imgf))
    fileid = Path(imgf).name.split(".")[0]
    fimg = fr.load_image_file(imgf)
    locs = fr.face_locations(fimg)
    number_of_faces = len(locs)
    fencs = fr.face_encodings(fimg, known_face_locations=locs)
    print("test image {}, found {} faces".format(imgf, number_of_faces))

    pimg = Image.fromarray(fimg)
    locindex = 0
    for loc in locs:
        fenc = fencs[locindex]
        results = fr.compare_faces(train_encodings, fenc, tolerance=0.4)
        distances = fr.face_distance(train_encodings, fenc)

        top, right, bottom, left = loc
        cropped = pimg.crop([left, top, right, bottom])

        draw = ImageDraw.Draw(pimg)
        found = False
        index = 0
        selected = -1
        distance = 0
        for result in results:
            if result:
                #print("image {}[{}] is match with image {}, distance {}".format(imgf, locindex, train_keys[index], distances[index]))
                found = True
                if distance==0:
                    distance = distances[index]
                    selected = index
                elif distances[index] < distance:
                    distance = distances[index]
                    selected = index
            
            index +=1

        if selected >= 0:
            print("image {}[{}] is match with image {}".format(imgf, locindex, train_keys[selected]))
            draw.rectangle([left, top, right, bottom], outline="green", width=2)
            draw.text([left, bottom+4], "[{}]".format(locindex) + " is " + train_keys[selected], font=fnt, fill=(0,255,0,0))    
        locindex +=1
            
        if not found:
            draw.rectangle([left, top, right, bottom], outline="yellow", width=2)
            print("Image {} has no match with any trains data".format(imgf))

        # convert to cv2
        cropped = np.array(cropped)  
        cropped = cv2.resize(cropped, cfg.input_shape)
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        cropped = np.reshape(cropped, [1, cropped.shape[0], cropped.shape[1], 1])

        predicted_class = np.argmax(model.predict(cropped))
        predicted_label = emotions[predicted_class]
        draw.text([left, bottom+16], predicted_label, font=fnt, fill=(255,255,0,100))   

    pimg.save(cfg.dumpfolder + "{}.png".format(fileid))
    testindex +=1