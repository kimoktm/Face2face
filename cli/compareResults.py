#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from face2face.models import MeshModel

import cv2
import numpy as np
from skimage import img_as_float, img_as_ubyte
import matplotlib.pyplot as plt

import os
import glob
import argparse
from tqdm import tqdm


def saveImage(path, img):
    b,g,r = cv2.split(img)
    img = cv2.merge([r,g,b])
    img = img_as_ubyte(img)
    cv2.imwrite(path, img)

def loadImage(path):
    b,g,r = cv2.split(cv2.imread(path))
    img_org = cv2.merge([r,g,b])
    img = img_as_float(img_org)
    return img

def main():
    # Change directory to the folder that holds the VRN data, OpenPose landmarks, and original images (frames) from the source video
    os.chdir('./data')

    # Load 3DMM
    m = MeshModel('../models/bfm2017.npz')

    samples = [250, 500, 1000, 5000, 10000, 20000, 30000]
    # samples = [5000, 10000, 20000, 30000]
    # samples = [250, 1000, 30000]

    data_path = os.path.join(FLAGS.params_gt, '*.png')
    keyframes = glob.glob(data_path)

    pix_lst = []
    sh_lst = []
    exp_lst = []
    pose_lst = []
    flt_samples = []

    for s in tqdm(samples):
        pix_axis = []
        sh_axis = []
        exp_axis = []
        pose_axis = []
        params_src = os.path.join(FLAGS.params_folders, 'rendering_' + str(s))

        if not os.path.isdir(params_src):
            continue

        flt_samples.append(s)

        for i in (range(FLAGS.start_frame, len(keyframes), 10)):
            # Load rendered image
            fNameImgOrig = os.path.join(params_src, str(i) + '.png')
            syn_img = loadImage(fNameImgOrig)

            # Load GT image
            GtNameImgOrig = os.path.join(FLAGS.params_gt, str(i) + '.png')
            gt_img = loadImage(GtNameImgOrig)

            # Load syn parameters
            syn_params = np.load(os.path.join(params_src, str(i) + "_params.npy"))
            syn_shCoef = syn_params[:27]
            syn_exp    = syn_params[27 + m.numId: -6]
            syn_pose   = syn_params[-6:]

            # Load GT parameters
            gt_params = np.load(os.path.join(FLAGS.params_gt, str(i) + "_params.npy"))
            gt_shCoef = gt_params[:27]
            gt_exp    = gt_params[27 + m.numId: -6]
            gt_pose   = gt_params[-6:]

            sh_diff = np.sum((gt_shCoef - syn_shCoef)**2)
            sh_axis.append(sh_diff**(1/2))
            
            exp_diff = np.sum((gt_exp - syn_exp)**2)
            exp_axis.append(exp_diff**(1/2))

            pose_diff = np.sum((gt_pose - syn_pose)**2)
            pose_axis.append(pose_diff**(1/2))

            pix_diff = np.sum((gt_img - syn_img)**2)
            pix_axis.append(pix_diff**(1/2))
            # saveImage(os.path.join(FLAGS.output_dir, str(i) + ".png"), (gt_img - syn_img))

        pix_lst.append(pix_axis)
        sh_lst.append(sh_axis)
        exp_lst.append(exp_axis)
        pose_lst.append(pose_axis)


    # plot pix difference
    plt.figure("Pixel Difference", figsize=(14, 11.0))
    plt.title("Pixel Difference")
    plt.ylim([0,60])
    pd = []
    for s in range(len(flt_samples)):
        x, = plt.plot(pix_lst[s], label=str(flt_samples[s]))
        pd.append(x)
    plt.legend(handles = pd)
    plt.savefig(os.path.join(FLAGS.output_dir, 'pixel_difference.png'))

    # plot Exp difference
    plt.figure("Exp Difference", figsize=(14, 11.0))
    plt.title("Exp Difference")
    plt.ylim([0,400])
    ed = []
    for s in range(len(flt_samples)):
        x, = plt.plot(exp_lst[s], label=str(flt_samples[s]))
        ed.append(x)
    plt.legend(handles = ed)
    plt.savefig(os.path.join(FLAGS.output_dir, 'exp_difference.png'))

    # plot SH difference
    plt.figure("SH Difference", figsize=(14, 11.0))
    plt.title("SH Difference")
    plt.ylim([0,1.0])
    sd = []
    for s in range(len(flt_samples)):
        x, = plt.plot(sh_lst[s], label=str(flt_samples[s]))
        sd.append(x)
    plt.legend(handles = sd)
    plt.savefig(os.path.join(FLAGS.output_dir, 'sh_difference.png'))

    # plot Pose difference
    plt.figure("Pose Difference", figsize=(14, 11.0))
    plt.title("Pose Difference")
    plt.ylim([0,9])
    dd = []
    for s in range(len(flt_samples)):
        x, = plt.plot(pose_lst[s], label=str(flt_samples[s]))
        dd.append(x)
    plt.legend(handles = dd)
    plt.savefig(os.path.join(FLAGS.output_dir, 'pose_difference.png'))
    # plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'Render fitted parameters')
    parser.add_argument('--params_folders', help = 'Path to parameters folders')
    parser.add_argument('--params_gt', help = 'Path to ground truth params')
    parser.add_argument('--output_dir', help = 'Path to save comparison results')
    parser.add_argument('--start_frame', help = 'Frame to start tracking from (optional)',type = int, default = 0)

    FLAGS, unparsed = parser.parse_known_args()

    main()
