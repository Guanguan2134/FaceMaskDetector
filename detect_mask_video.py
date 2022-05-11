from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from screeninfo import get_monitors
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import yaml


def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            if face.any():
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                face = np.expand_dims(face, axis=0)

                # add the face and bounding boxes to their respective
                # lists
                faces.append(face)
                locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        face = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return locs, preds


# construct the argument parser and parse the arguments
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--face", type=str, help="The path of face detector model directory")
    ap.add_argument("-m", "--model", type=str, help="The path of trained face mask detector model")
    ap.add_argument("-c", "--confidence", default="auto", 
                    help="Minimum probability to filter weak detections, default is 'auto' which means the result depend on the bigger probability between w/ mask and w/o mask")
    
    args = vars(ap.parse_args())
    if args['face'] is not None:
        config['Test']['face_model_dir'] = args['face']
    if args['model'] is not None:
        config['Test']['mask_model'] = args['model']
    if args['confidence'] is not None:
        config['Test']['confidence'] = args['confidence']

with open("config.yml", "w") as f:
    yaml.safe_dump(config, f)


if __name__ == '__main__':
    # load our serialized face detector model from disk
    print("[INFO] loading face detector model...")
    prototxtPath = os.path.sep.join([config['Test']['face_model_dir'], "deploy.prototxt"])
    weightsPath = os.path.sep.join([config['Test']['face_model_dir'], "res10_300x300_ssd_iter_140000.caffemodel"])
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model from disk
    print("[INFO] loading face mask detector model...")
    maskNet = load_model(config['Test']['mask_model'])

    # initialize the video stream and allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    vs = VideoStream(src=1).start()
    time.sleep(2.0)

    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        try:
            (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

            # loop over the detected face locations and their corresponding
            # locations
            for (box, pred) in zip(locs, preds):
                # unpack the bounding box and predictions
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred

                # determine the class label and color we'll use to draw
                # the bounding box and text
                if config['Test']['confidence'] == "auto" or config['Test']['confidence'] is None:
                    label = "Mask" if mask > withoutMask else "No Mask"
                else:
                    label = "Mask" if mask > config['Test']['confidence'] else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                # include the probability in the label
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                # display the label and bounding box rectangle on the output
                # frame
                cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        except:
            pass

        # show the output frame
        # define the screen resulation
        screen_res =  [[i.width, i.height] for i in get_monitors() if i.is_primary==True][0]
        scale_width = screen_res[0] / frame.shape[1]
        scale_height = screen_res[1] / frame.shape[0]
        scale = min(scale_width, scale_height)
        
        # resized window width and height
        window_width = int(frame.shape[1] * scale)
        window_height = int(frame.shape[0] * scale)
        
        # cv2.WINDOW_NORMAL makes the output window resizealbe
        cv2.namedWindow('Resized Window', cv2.WINDOW_NORMAL)
        
        # resize the window according to the screen resolution
        cv2.resizeWindow('Resized Window', window_width, window_height)
        cv2.imshow('Resized Window', frame)

        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()
