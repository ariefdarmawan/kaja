from helper import config
import sys
import glob
import face_recognition as fr

cfg = config.Config()
trains = {}

if __name__ == "__main__":
    cfg.testfolder = "./images/test"
    cfg.trainfolder = "./images/train/*.jpg"
  
    # run the train first
    index = 0
    for imgf in glob.glob(cfg.trainfolder):
        index+=1
        print("train image {} {}".format(index, imgf))
        fimg = fr.load_image_file(imgf)
        locs = fr.face_locations(fimg)
        number_of_faces = len(locs)
        print("found {} images".format(number_of_faces))
        
    '''
    index = 0
    for imgf in glob.glob(cfg.trainfolder):
        index+=1
        fimg = fr.load_image_file(imgf)
        fenc = fr.face_encodings(fimg)[0]
        trains[index] = fenc

    print("trains {} images".format(len(trains)))
    for index in trains:
        print("data {}: {}".format(index, trains[index]))
    '''