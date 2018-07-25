import os
import glob
import argparse
import pandas as pd

import cv2
import numpy as np
from skimage import io, img_as_float

#delete
import dlib


# Basic model parameters as external flags.
FLAGS = None


def getFaceKeypoints(frame_cnt, openFace_landmarks):
    shapes2D = []
    frame = openFace_landmarks[openFace_landmarks['frame'] == frame_cnt]

    for i in range(0, 68):
        x = frame[' x_' + str(i)].values[0]
        y = frame[' y_' + str(i)].values[0]
        shapes2D.append([x, y])

    return shapes2D

def getdlibKeypoints(img, detector, predictor, maxImgSizeForDetection=320):
    imgScale = 1
    scaledImg = img
    if max(img.shape) > maxImgSizeForDetection:
        imgScale = maxImgSizeForDetection / float(max(img.shape))
        scaledImg = cv2.resize(img, (int(img.shape[1] * imgScale), int(img.shape[0] * imgScale)))

    dets = detector(scaledImg, 1)

    if len(dets) == 0:
        return None

    shapes2D = []
    for det in dets:
        faceRectangle = dlib.rectangle(int(det.left() / imgScale), int(det.top() / imgScale), int(det.right() / imgScale), int(det.bottom() / imgScale))
        dlibShape = predictor(img, faceRectangle)
        shape2D = np.array([[p.x, p.y] for p in dlibShape.parts()])
        shape2D = shape2D.T
        shapes2D.append(shape2D)

    return shapes2D


def drawPoints(img, points, color=(0, 255, 0)):
    for point in points:
        cv2.circle(img, (int(point[0]), int(point[1])), 2, color)

def main():
    # Landmark detector
    landmarks = pd.read_csv(FLAGS.openFace_landmarks)
    # cap = cv2.VideoCapture(FLAGS.video)
    # Landmark detector
    os.chdir('./data')
    predictor_path = "../models/shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    cap = cv2.VideoCapture(FLAGS.video)

    frame_cnt = 1
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            # Load the source video frame and convert to 64-bit float
            # b,g,r = cv2.split(frame)
            # img_org = cv2.merge([r,g,b])
            # img = img_as_float(img_org)

            # dlib2D = getdlibKeypoints(frame, detector, predictor)
            # dlib2D = np.asarray(dlib2D)[0].T 
            # drawPoints(frame, dlib2D, (255, 0, 0))

            shape2D = getFaceKeypoints(frame_cnt, landmarks)
            shape2D = np.asarray(shape2D)

            #frame = cv2.resize(frame,(1080, 720), interpolation = cv2.INTER_CUBIC)
            drawPoints(frame, shape2D)
            cv2.imshow('frame', frame)


            cv2.imwrite(os.path.join(FLAGS.output_dir, str(frame_cnt) + ".png"), frame)
            frame_cnt = frame_cnt + 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
        


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Extract Keyframes for initalization')
    parser.add_argument('--video', help = 'Path to input video')
    parser.add_argument('--openFace_landmarks', help = 'Path to openface landmarks')
    parser.add_argument('--output_dir', help = 'Output directory')

    FLAGS, unparsed = parser.parse_known_args()

    main()