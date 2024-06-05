# CSA-Lanenet from NCKU MediaCore
This is a refernce code in “CSA-Lanenet: A Contiguous Spatial Attention Lane Detection Network with Vision Transformer Modules”. Our code is based on Lanenet.
# Environment
Python 3.7.15
CUDA 10.0.130
cuDNN 7.6.5
Tensorflow-gpu 2.0.0
others :
opencv-python, numpy,easydict,glog
# Folder construction
The data folder is for training images.
The encoder_decoder_model folder is basic structure for our network.
The lanenet_model folder contains the final network results after post-processing and loss computation.
# Training Tips
1. Change your training config such as batch size in /config/global_config.py.
2. Please place the training and testing images in the data folder, which should contain both the original images and the segmented images. Additionally, create train.txt and val.txt files that include the file paths of the images for the program's reference.
# Training
python train_lanenet_pool.py --net_flag mv3 --dataset_dir "D:\your_directory\data\training_data_example\tu_train\training"
# Testing 
python test_lanenet_pool.py --is_batch True --net_flag mv3  --use_gpu 1 --save_dir "save\your\testing\images\" --weights_path "your\model\weight\such_as_lanenet_2023-07-20-20-11-13.ckpt-324000" --image_path  data\training_data_example\tu_train\testing\gt_image


