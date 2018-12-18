Facial Capture
===========

This is a Python package to fit 3D morphable models (3DMMs) to images of faces. It mainly provides classes to work with and render 3DMMs, and functions that use these classes to optimize the objective of fitting 3DMMs to a source RGB image.


Features
========

* Fit a 3DMM shape model to a source RGB map.
* Joint optimization of rendering pixel error and landmarks fitting
* Fit a 3DMM texture model with spherical harmonic lighting to a source RGB image.
* Recover the barycentric parameters of the underlying verticles from the 3DMM mesh triangles that contribute to each pixel of a person's face in an image.
* Extract per vertex texture
* Track expressions and spherical harmonic lighting over a sequence of images.


Prerequisites
=============

* Python 3
* Install all requirements with ``pip``: ``pip install -r requirements.txt .``
* Install via ``pip``: ``pip install -e .``

You need to download 2017 BFM model as we aren't allowed to share it:

* Create `models` folder under `Facial-Capture` 
* Download Basel model 2017 model from [here](https://faces.dmi.unibas.ch/bfm/bfm2017.html) to `models` folder
* Process via ``python processBFM2017.py``

Also you would need the trained landmark dlib predictor:

* Download `shape_predictor_68_face_landmarks` from [here](https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat) to `models` folder


Running
=======

* First create a face identity (use 1 to 3 images max) using

  ```
  python bin/initialize.py --input_dir path_to_init_images --output_dir path_to_save_identity
  ```


* After creating the identity, you can now track the expressions using:

  ```
  python bin/tracker.py --input_dir path_to_tracking_images --output_dir path_to_save_tracking --parameters path_to_save_identity/params.npy
  ```
