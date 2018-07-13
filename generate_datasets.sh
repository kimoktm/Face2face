#!/bin/sh


#
# Barnie
python bin/initialize.py --input_dir '/home/karim/Documents/Development/FacialCapture/Facial-Capture/data/barnie/init' \
						 --output_dir '/home/karim/Documents/Development/FacialCapture/Facial-Capture/data/barnie'

python bin/tracker.py --input_dir '/home/karim/Documents/Development/FacialCapture/Facial-Capture/data/barnie/orig' \
					  --output_dir '/home/karim/Documents/Development/FacialCapture/Facial-Capture/data/barnie/rendering' \
					  --parameters '/home/karim/Documents/Development/FacialCapture/Facial-Capture/data/barnie/params.npy'

#
# Bush
python bin/initialize.py --input_dir '/home/karim/Documents/Development/FacialCapture/Facial-Capture/data/bush/init' \
						 --output_dir '/home/karim/Documents/Development/FacialCapture/Facial-Capture/data/bush'

python bin/tracker.py --input_dir '/home/karim/Documents/Development/FacialCapture/Facial-Capture/data/bush/orig' \
					  --output_dir '/home/karim/Documents/Development/FacialCapture/Facial-Capture/data/bush/rendering' \
					  --parameters '/home/karim/Documents/Development/FacialCapture/Facial-Capture/data/bush/params.npy' \
					  --start_frame 10


#
# Trump
python bin/initialize.py --input_dir '/home/karim/Documents/Development/FacialCapture/Facial-Capture/data/trump/init' \
						 --output_dir '/home/karim/Documents/Development/FacialCapture/Facial-Capture/data/trump'

python bin/tracker.py --input_dir '/home/karim/Documents/Development/FacialCapture/Facial-Capture/data/trump/orig' \
					  --output_dir '/home/karim/Documents/Development/FacialCapture/Facial-Capture/data/trump/rendering' \
					  --parameters '/home/karim/Documents/Development/FacialCapture/Facial-Capture/data/trump/params.npy'


#
# Jack
python bin/initialize.py --input_dir '/home/karim/Documents/Development/FacialCapture/Facial-Capture/data/jack/init' \
						 --output_dir '/home/karim/Documents/Development/FacialCapture/Facial-Capture/data/jack'

python bin/tracker.py --input_dir '/home/karim/Documents/Development/FacialCapture/Facial-Capture/data/jack/orig' \
					  --output_dir '/home/karim/Documents/Development/FacialCapture/Facial-Capture/data/jack/rendering' \
					  --parameters '/home/karim/Documents/Development/FacialCapture/Facial-Capture/data/jack/params.npy'


#
# Obama
python bin/initialize.py --input_dir '/home/karim/Documents/Development/FacialCapture/Facial-Capture/data/obama/init' \
						 --output_dir '/home/karim/Documents/Development/FacialCapture/Facial-Capture/data/obama'

python bin/tracker.py --input_dir '/home/karim/Documents/Development/FacialCapture/Facial-Capture/data/obama/orig' \
					  --output_dir '/home/karim/Documents/Development/FacialCapture/Facial-Capture/data/obama/rendering' \
					  --parameters '/home/karim/Documents/Development/FacialCapture/Facial-Capture/data/obama/params.npy'

#
# Putin
python bin/initialize.py --input_dir '/home/karim/Documents/Development/FacialCapture/Facial-Capture/data/putin/init' \
						 --output_dir '/home/karim/Documents/Development/FacialCapture/Facial-Capture/data/putin'

python bin/tracker.py --input_dir '/home/karim/Documents/Development/FacialCapture/Facial-Capture/data/putin/orig' \
					  --output_dir '/home/karim/Documents/Development/FacialCapture/Facial-Capture/data/putin/rendering' \
					  --parameters '/home/karim/Documents/Development/FacialCapture/Facial-Capture/data/putin/params.npy'


#
# Craig
python bin/initialize.py --input_dir '/home/karim/Documents/Development/FacialCapture/Facial-Capture/data/craig/init' \
						 --output_dir '/home/karim/Documents/Development/FacialCapture/Facial-Capture/data/craig'

python bin/tracker.py --input_dir '/home/karim/Documents/Development/FacialCapture/Facial-Capture/data/craig/orig' \
 					  --output_dir '/home/karim/Documents/Development/FacialCapture/Facial-Capture/data/craig/rendering' \
 					  --parameters '/home/karim/Documents/Development/FacialCapture/Facial-Capture/data/craig/params.npy' 


#
# Cage
python bin/initialize.py --input_dir '/home/karim/Documents/Development/FacialCapture/Facial-Capture/data/cage/init' \
						 --output_dir '/home/karim/Documents/Development/FacialCapture/Facial-Capture/data/cage'

python bin/tracker.py --input_dir '/home/karim/Documents/Development/FacialCapture/Facial-Capture/data/cage/orig' \
					  --output_dir '/home/karim/Documents/Development/FacialCapture/Facial-Capture/data/cage/rendering' \
					  --parameters '/home/karim/Documents/Development/FacialCapture/Facial-Capture/data/cage/params.npy'
