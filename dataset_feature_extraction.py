'''
Created on Oct 5, 2017

@author: jedrzej
'''
import sys
CAFFE_ROOT = '/home/jedrzej/work/caffe/python'
sys.path.insert(0, CAFFE_ROOT) 
import caffe
import h5py
import os
import numpy as np
import cPickle as pickle
from glob import glob
from time import time
from caffe_feature_extractor import CaffeFeatureExtractor
	
DATA_ROOT = "/media/jedrzej/Seagate/DATA/"

def run_caffe_feature_extractor_on_list_of_images(imageList):
	#Create caffe feature extractor object
    	extractionNet = CaffeFeatureExtractor(
		model_path=DATA_ROOT+"ILSVRC2012/googlenet/googlenet_deploy.prototxt",
		pretrained_path=DATA_ROOT+"ILSVRC2012/googlenet/bvlc_googlenet.caffemodel",
		blob="pool5/7x7_s1",
		crop_size=224,
		mean_values=[104.0, 117.0, 123.0]
		)
	model_path=DATA_ROOT+"ILSVRC2012/googlenet/googlenet_deploy.prototxt"
	pretrained_path=DATA_ROOT+"ILSVRC2012/googlenet/bvlc_googlenet.caffemodel"
	blob="pool5/7x7_s1"
	crop_size=224
	mean_values=[104.0, 117.0, 123.0]
	
	caffe.set_mode_gpu()
	caffe.set_device(1)

	extractionNet = caffe.Net(model_path, pretrained_path, caffe.TEST)
	
	mean = np.zeros((3, crop_size, crop_size))
	mean[0] = mean_values[0]
	mean[1] = mean_values[1]
   	mean[2] = mean_values[2]
	transformer = caffe.io.Transformer({"data": extractionNet.blobs["data"].data.shape}) # for cropping
	transformer.set_transpose("data", (2,0,1)) # (H,W,C) => (C,H,W)
	transformer.set_mean("data", mean) # subtract by mean
	transformer.set_raw_scale("data", 255) # [0.0, 1.0] => [0.0, 255.0].
	transformer.set_channel_swap("data", (2,1,0)) # RGB => BGR
	print "Transformer ready"
	images = []
	for img_path in imageList:
		try:
			start_time = time()
			img = caffe.io.load_image(img_path)		# read the image
			img = caffe.io.resize_image(img,(224,224))	# resize to correct dimensions
			images.append(transformer.preprocess("data", img))
			del img
		except:
			print img_path
	print "Images transformed"
	images = np.asarray(images)
	n = 50 		# number of splits so to not get 'out of memory'
	feats = np.zeros((1,1024))
	ims = np.array_split(images, n)
	for i in xrange(n):
		extractionNet.blobs['data'].reshape(ims[i].shape[0],ims[i].shape[1],ims[i].shape[2],ims[i].shape[3])
		out = extractionNet.forward_all(**{extractionNet.inputs[0]: ims[i], "blobs": [blob]})[blob].copy()
		feats = np.concatenate((feats, out.reshape(out.shape[0],out.shape[1])),axis = 0)
		del out
	del images, ims
	feats = feats[1:,...]
	
	return feats


def run_caffe_feature_extractor_on_dataset(datasetPath, pathToSave, imgExtension):

	#Create caffe feature extractor object
    	extractionNet = CaffeFeatureExtractor(
		model_path=DATA_ROOT+"ILSVRC2012/googlenet/googlenet_deploy.prototxt",
		pretrained_path=DATA_ROOT+"ILSVRC2012/googlenet/bvlc_googlenet.caffemodel",
		blob="pool5/7x7_s1",
		crop_size=224,
		mean_values=[104.0, 117.0, 123.0]
		)
	model_path=DATA_ROOT+"ILSVRC2012/googlenet/googlenet_deploy.prototxt"
	pretrained_path=DATA_ROOT+"ILSVRC2012/googlenet/bvlc_googlenet.caffemodel"
	blob="pool5/7x7_s1"
	crop_size=224
	mean_values=[104.0, 117.0, 123.0]
	
	caffe.set_mode_gpu()
	caffe.set_device(1)

	extractionNet = caffe.Net(model_path, pretrained_path, caffe.TEST)
	
	mean = np.zeros((3, crop_size, crop_size))
	mean[0] = mean_values[0]
	mean[1] = mean_values[1]
   	mean[2] = mean_values[2]
	transformer = caffe.io.Transformer({"data": extractionNet.blobs["data"].data.shape}) # for cropping
	transformer.set_transpose("data", (2,0,1)) # (H,W,C) => (C,H,W)
	transformer.set_mean("data", mean) # subtract by mean
	transformer.set_raw_scale("data", 255) # [0.0, 1.0] => [0.0, 255.0].
	transformer.set_channel_swap("data", (2,1,0)) # RGB => BGR
	print "Transformer ready"


    	if not os.path.exists(pathToSave):
    		os.makedirs(pathToSave)
	# Extracting features from datasePath and saving them as HDF5 files in pathToSave
	folderList = glob(datasetPath+'*/')
	for folder_i, folder in enumerate(folderList):
		start_time = time()
		category_name = folder.split("/")[-2]
		print category_name
		if(os.path.isfile(pathToSave+category_name+".hdf5")==False):
			img_path_list = glob(folder+'*'+imgExtension)
			del folder
			images = []
			for img_path_i, img_path in enumerate(img_path_list):
				try:	
					img = caffe.io.load_image(img_path)		# read the image
					img = caffe.io.resize_image(img,(224,224))	# resize to correct dimensions
					images.append(transformer.preprocess("data", img))

					#img_mirror = img[:, ::-1, :]  			# flip for mirrors
					#images.append(transformer.preprocess("data", img_mirror))

					del img#, img_mirror
				except:
					print "Probable error"
			del img_path_list
			images = np.asarray(images)
			
			n = 10 		# number of splits so to not get 'out of memory'
			feats = np.zeros((1,1024))
			ims = np.array_split(images, n)
			for i in xrange(n):
				extractionNet.blobs['data'].reshape(ims[i].shape[0],ims[i].shape[1],ims[i].shape[2],ims[i].shape[3])
				out = extractionNet.forward_all(**{extractionNet.inputs[0]: ims[i], "blobs": [blob]})[blob].copy()
				feats = np.concatenate((feats, out.reshape(out.shape[0],out.shape[1])),axis = 0)
				del out
			del images, ims
			feats = feats[1:,...]

			f = h5py.File(pathToSave+category_name+'.hdf5', 'w')
			f.create_dataset('data', (feats.shape[0], feats.shape[1]), dtype='double')
			f['data'][...] = feats
			f.close()
			del feats, f
		end_time = time()
		print "Processed", folder_i + 1, "folder in",str(round(end_time-start_time,2)),"seconds"


def main():
    run_caffe_feature_extractor_on_dataset(DATA_ROOT+"Animals_with_Attibutes_2/Animals_with_Attributes2/attribute_images/",DATA_ROOT+"Animals_with_Attibutes_2/Animals_with_Attributes2/inception_features/", "jpg")

if __name__ == "__main__":
    main()
