from helper import config
import sys
import glob
import face_recognition as fr
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

cfg = config.Config()
trains = {}

cfg.testfolder = "./images/test/*"
cfg.trainfolder = "./images/train/*"
cfg.dumpfolder = "./images/temp/"

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

        top, right, bottom, left = loc
        draw = ImageDraw.Draw(pimg)
        draw.rectangle([left, top, right, bottom], outline="green", width=3)
        
        found = False
        index = 0
        #print("image {} compare results: {}".format(imgf, results))
        for result in results:
            if result:
                print("image {} is match with image {}".format(imgf, train_keys[index]))
                #fnt = ImageFont.truetype("arial")
                #fnt.size = 14
                draw.text([left, bottom+4], train_keys[index])
                found = True
                break
            else:
                index +=1
        locindex +=1
            
        if not found:
            print("Image {} has no match with any trains data".format(imgf))

    pimg.save(cfg.dumpfolder + "{}_{}_{}.png".format(fileid, testindex, locindex))
    testindex +=1