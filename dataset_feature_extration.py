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

def run_caffe_feature_extractor_on_data(datasetPath, pathToSave, imgExtension):
	#Create caffe feature extractor object
    	extractionNet = CaffeFeatureExtractor(
		model_path=DATA_ROOT+"ILSVRC2012/googlenet/googlenet_deploy.prototxt",
		pretrained_path=DATA_ROOT+"ILSVRC2012/googlenet/bvlc_googlenet.caffemodel",
		blob="pool5/7x7_s1",
		crop_size=224,
		mean_values=[104.0, 117.0, 123.0]
		)
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
			feats = []
			for img_path_i, img_path in enumerate(img_path_list):
				try:
					img = caffe.io.load_image(img_path)		# read the image
					img = caffe.io.resize_image(img,(224,224))	# resize to correct dimensions
					# Features extraction on image:
					feat = extractionNet.extract_feature(img)	# extract features
					feats.append(feat)
					del feat
					# Features extraction on mirrored image:
					img_mirror = img[:, ::-1, :]  			# flip for mirrors
					feat = extractionNet.extract_feature(img_mirror)	# extract features from mirrored image
					feats.append(feat)
					del feat, img, img_mirror
				except:
					print "Probable error"
			print len(feats)
			feats = np.asarray(feats)
			feats = feats.reshape(feats.shape[0],-1)
			f = h5py.File(pathToSave+category_name+'.hdf5', 'w')
			f.create_dataset('data', (feats.shape[0], feats.shape[1]), dtype='double')
			f['data'][...] = feats
			f.close()
			del feats, f
		end_time = time()
		print "Processed", folder_i + 1, "folder in",str(round(end_time-start_time,2)),"seconds"

#run_caffe_feature_extractor_on_data("/media/jedrzej/SAMSUNG/DATA/SUN_attribute/images/","/media/jedrzej/SAMSUNG/DATA/SUN_attribute/inception_features/", "jpg")
