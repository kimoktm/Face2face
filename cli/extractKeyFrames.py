import os
import glob
import argparse

import cv2
import dlib
import numpy as np
from skimage import io, img_as_float

# Basic model parameters as external flags.
FLAGS = None


def getFaceKeypoints(img, detector, predictor, maxImgSizeForDetection=320):
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
        # shape2D = np.array([[(det.left() / imgScale), (det.top() / imgScale)], \
        #                     [(det.right() / imgScale), (det.top() / imgScale)], \
        #                     [(det.right() / imgScale), (det.bottom() / imgScale)],
        #                     [(det.left() / imgScale), (det.bottom() / imgScale)]])

        shape2D = np.array([[p.x, p.y] for p in dlibShape.parts()])
        shape2D = shape2D.T
        shapes2D.append(shape2D)

    return shapes2D


def drawPoints(img, points, color=(0, 255, 0)):
    for point in points:
        cv2.circle(img, (int(point[0]), int(point[1])), 2, color)

def main():
    # Landmark detector
    os.chdir('./data')
    predictor_path = "../models/shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    cap = cv2.VideoCapture(FLAGS.video)

    frame_cnt = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            # Load the source video frame and convert to 64-bit float
            # b,g,r = cv2.split(frame)
            # img_org = cv2.merge([r,g,b])
            # img = img_as_float(img_org)

            shape2D = getFaceKeypoints(frame, detector, predictor)
            shape2D = np.asarray(shape2D)[0].T 

            # nose = shape2D[28:31]
            # heading = np.cross(nose[0] - nose[1], nose[0] - nose[2], axisa = 0, axisb = 0)
            # print(heading)
            # a = (int(frame.shape[1] / 2), int(frame.shape[0] / 2))
            # b = (int(frame.shape[1] / 2), int(frame.shape[0] / 2) + int(heading))
            # cv2.line(frame, a, b, (255,0,0))
            
            # 27 - 8
            p1_id = 27
            p2_id = 8
            x2 = shape2D[p1_id, 0]
            x1 = shape2D[p2_id, 0]
            y2 = shape2D[p1_id, 1]
            y1 = shape2D[p2_id, 1]
            face_dist = ( (x2 - x1)**2 + (y2 - y1)**2 )**(1/2)
            print(face_dist)
            # shape2D = [shape2D[p1_id], shape2D[p2_id]]

            # drawPoints(frame, shape2D)
            # cv2.imshow('frame', frame)

            # cv2.imwrite(os.path.join(FLAGS.output_dir, str(frame_cnt) + ".png"), frame)
            frame_cnt = frame_cnt + 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
        


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Extract Keyframes for initalization')
    parser.add_argument('--video', help = 'Path to input video')
    parser.add_argument('--num_frames', help = 'Number of key frames to extract', type = int, default = 3)
    parser.add_argument('--output_dir', help = 'Output directory')

    FLAGS, unparsed = parser.parse_known_args()

    main()