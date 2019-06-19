from helper import config
import sys
import glob
import face_recognition as fr
import PIL.Image 
import PIL.ImageDraw

cfg = config.Config()
trains = {}

if __name__ == "__main__":
    cfg.testfolder = "./images/test/*"
    cfg.trainfolder = "./images/train/*"
  
    # run the train first
    for imgf in glob.glob(cfg.trainfolder):
        fimg = fr.load_image_file(imgf)
        locs = fr.face_locations(fimg)
        fencs = fr.face_encodings(fimg, known_face_locations=locs)
        number_of_faces = len(locs)
        print("train {}, found {} images".format(imgf, number_of_faces))

        locindex = 0
        for face_location in locs:
            fenc = fencs[locindex]
            trains[imgf + "_" + str(locindex)] = fenc
            locindex +=1

    print("trains {} faces".format(len(trains)))
    
    # run test
    train_encodings = list(trains.values())
    train_keys = list(trains.keys())
    #print("keys are: {}".format(train_keys))
    
    for imgf in glob.glob(cfg.testfolder):
        #print("test image {}".format(imgf))
        fimg = fr.load_image_file(imgf)
        locs = fr.face_locations(fimg)
        number_of_faces = len(locs)
        fencs = fr.face_encodings(fimg, known_face_locations=locs)
        print("test image {}, found {} faces".format(imgf, number_of_faces))

        locindex = 0
        for loc in locs:
            fenc = fencs[locindex]
            results = fr.compare_faces(train_encodings, fenc, tolerance=0.4)

            found = False
            index = 0
            #print("image {} compare results: {}".format(imgf, results))
            for result in results:
                if result:
                    print("image {} is match with image {}".format(imgf, train_keys[index]))
                    found = True
                    break
                else:
                    index +=1
            locindex +=1
                
            if not found:
                print("Image {} has no match with any trains data".format(imgf))
                