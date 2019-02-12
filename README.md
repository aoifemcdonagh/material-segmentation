# MINC_model_testing
Set of files for testing models pretrained on MINC dataset

# full_image_classify.py
Takes a single image and performs coarse material classification
Arguments: 
1. `-i` or `--image` : path to image to perform material classfication on
2. `-m` or `--model` : directory path containing `.caffemodel` and `.prototxt` files
3. `-p` or `--plot` : set to `False` to avoid plotting results. Default is `True`
