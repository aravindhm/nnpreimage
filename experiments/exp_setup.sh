#/bin/bash

# This script will setup the data and networks required to replicate our results from the IJCV paper


# First setup pics. Everything except for the imagenet validation data is available on our server
if [ ! -d "data" ]; then
  mkdir "data";
fi

if [ ! -d "data/pics" ]; then
  mkdir "data/pics";
fi

if [ ! -r "data/pics/pics.zip" ]; then
  # CAUTION: I assume that if this zip file exists then all the images are already there.
  wget http://gandalf.robots.ox.ac.uk/ijcv2015/data/pics.zip -O data/pics/pics.zip
  cd data/pics/
  unzip pics.zip
  cd ../../
fi

# Next setup all the network models
cd networks/
if [ ! -r "imagenet-caffe-ref.mat" ]; then
  wget www.vlfeat.org/matconvnet/models/imagenet-caffe-ref.mat 
fi

if [ ! -r "imagenet-caffe-alex.mat" ]; then
  wget www.vlfeat.org/matconvnet/models/imagenet-caffe-alex.mat 
fi

if [ ! -r "imagenet-vgg-m.mat" ]; then
  wget www.vlfeat.org/matconvnet/models/imagenet-vgg-m.mat
fi

if [ ! -r "imagenet-vgg-verydeep-16.mat" ]; then
  wget www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-16.mat
fi
cd ../
