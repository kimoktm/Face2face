Facial Capture
===========

This is a Python package to fit 3D morphable models (3DMMs) to images of faces. It mainly provides classes to work with and render 3DMMs, and functions that use these classes to optimize the objective of fitting 3DMMs to a source RGB image or depth map. The package is tentatively named ``mm``, so to use it, follow the installation instructions below and import it like you would any other Python package. ::

	import mm
	mm.do_stuff()

Features
========

* Fit a 3DMM shape model to a source RGB map.
* Joint optimization of rendering pixel error and landmarks fitting
* Fit a 3DMM texture model with spherical harmonic lighting to a source RGB image.
* Recover the barycentric parameters of the underlying verticles from the 3DMM mesh triangles that contribute to each pixel of a person's face in an image.
* Fit a 3DMM shape model to a source RGB image.

Prerequisites
=============

* Python 3
* Install all requirements with ``pip``: ``pip install -r requirements.txt .``.
* Install via ``pip``: ``pip install -e .``.