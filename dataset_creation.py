'''
Created on Oct 4, 2017

@author: jedrzej
'''
import h5py
import numpy as np
from glob import glob
import os
from time import time
import cPickle as pickle
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import shutil
from scipy.spatial import distance

from shuffle_in_unison import shuffle_in_unison

DATA_ROOT = "/media/jedrzej/SAMSUNG/DATA/"

def divideClusterIntoCenterAndSurface(features):
	clusterCenter = np.mean(features,axis = 0)
	dst = distance.cdist(features,clusterCenter.reshape(1,1024))

	cutOffDistanceMin = np.percentile(dst,25)

	cutOffDistanceMax = np.percentile(dst,90)

	centralFeatures = features[np.where(dst<=cutOffDistanceMin)[0],:]
	surfaceFeatures = features[np.where(dst>=cutOffDistanceMax)[0],:]
	return centralFeatures, surfaceFeatures

def get_k_features_dict(featuresPath, datasetPath, k = 5):
	# k = no of examples per category
	if(os.path.isfile(datasetPath+str(k)+"featuresDict.pickle")==False):
		featuresFileList = glob(featuresPath+'*hdf5')
		print "Creating",k,"features dictionary"
		random_features_dict = dict()
		for feat_file_i, feat_file in enumerate(featuresFileList):
			start_time = time()
			category_name = feat_file.split("/")[-1].split(".")[0]
			with h5py.File(feat_file,'r') as f:
				feats = np.array(f.get('data'))
				r = np.random.choice(feats.shape[0], k, replace=False)
				chosen_feats = feats[r,:]
				print np.shape(chosen_feats)
				random_features_dict[category_name] = chosen_feats
			end_time = time()
			print "Processed", feat_file_i + 1, "file in",str(round(end_time-start_time,2)),"seconds"
		pickle.dump(random_features_dict, open(datasetPath+str(k)+"featuresDict.pickle", 'wb'))
	else:
		print "Opening",k,"features dictionary"
		with open(datasetPath+str(k)+"featuresDict.pickle", 'rb') as handle:
        		random_features_dict = pickle.load(handle)
	return random_features_dict

def get_negative_data(categoryName, featuresDictionary):
	negativeData = []
	for key, value in featuresDictionary.iteritems():
		if(key!=categoryName):
			for v in value:
        			negativeData.append(v)
	return np.asarray(negativeData)

def split_dataset_to_finetune(featuresPath, featuresPathFT, featuresPathTest):
	if not os.path.exists(featuresPathFT):
    		os.makedirs(featuresPathFT)
	if not os.path.exists(featuresPathTest):
    		os.makedirs(featuresPathTest)

	fileList = np.asarray(glob(featuresPath+'*hdf5'))
	n = len(fileList)
	FT_indices = np.random.choice(n, int(0.7*n), replace=False)
	FT_data = fileList[FT_indices]
	test_data = np.delete(fileList,[FT_indices],axis=0)
	
	for i in FT_data:
		shutil.copyfile(i, featuresPathFT+i.split("/")[-1])
	for j in test_data:
		shutil.copyfile(j, featuresPathTest+j.split("/")[-1])
	
def createRegularTrainingData(featuresPath, datasetPath, pathToSave, howManyToCreate = 5):
	inShape = 1024
	outShape = 1025
	# create saving folder if does not exist
	if not os.path.exists(pathToSave):
    		os.makedirs(pathToSave)
	random_features_dict =  get_k_features_dict(featuresPath, datasetPath, 7)	# get at random k features from every category
	fileList = glob(featuresPath+'*hdf5')
	val_input = np.zeros((1,inShape))
	train_input = np.zeros((1,inShape))
	val_output = np.zeros((1,outShape))
	train_output = np.zeros((1,outShape))
	# Iterate through all .hdf5 files (all categories):
	print len(fileList)
	for fFile_i, fFile in enumerate(fileList):
		sTime = time()
		with h5py.File(fFile,'r') as f:
			dataT = np.array(f.get('data'))

		#pos, neg = divideClusterIntoCenterAndSurface(dataT)

		train_data = dataT
		del f
		n_val = int(0.3*len(train_data))
		val_indices = np.random.choice(len(train_data), n_val, replace=False)	# choose 30% of images at random for validation
		val_data = train_data[val_indices,...]
		train_data = np.delete(train_data,[val_indices],axis=0)
		#print len(val_data), len(train_data)
		

		# CREATE REGULAR SVM BOUNDS:
		
		#print "Got input data"
		category_name = fFile.split("/")[-1].split(".")[0]			# get category name
		negatives = get_negative_data(category_name, random_features_dict)
		#negatives = np.concatenate((negatives, neg),axis = 0)	# add far away negatives from same category to negatives coming from other categories
		pos_labels = np.ones((len(dataT)))
		neg_labels = -1*np.ones((len(negatives)))
		svm_data = np.concatenate((dataT,negatives),axis = 0)
		svm_labels = np.concatenate((pos_labels, neg_labels))

		svm_data, svm_labels = shuffle_in_unison(svm_data, svm_labels)
		clf = LogisticRegression()				# define Logistic Regression classifier
		#clf = SVC(0.1,kernel = 'linear')			# define SVM classifier
		clf.fit(svm_data, svm_labels)
		optimal_hyperplane = np.concatenate((np.asarray(clf.intercept_).copy().reshape((1,1)),clf.coef_.copy()),axis=1)	# concatenate [bias,weight]
		optimal_hyperplane = optimal_hyperplane.reshape(optimal_hyperplane.shape[1])
		#print np.shape(val_input), np.shape(val_data)		

		val_input = np.concatenate((val_input,val_data),axis=0)
		train_input = np.concatenate((train_input,train_data),axis=0)
		val_output = np.concatenate((val_output,[optimal_hyperplane,]*val_data.shape[0]),axis=0)
		train_output = np.concatenate((train_output,[optimal_hyperplane,]*train_data.shape[0]),axis=0)
		del optimal_hyperplane, clf, train_data, val_data, pos_labels, neg_labels, negatives, svm_data,svm_labels, dataT,val_indices
		

		print "Category:",fFile_i+1,"took:",str(round((time()-sTime),2))
		#print np.shape(val_input), np.shape(train_input), np.shape(val_output), np.shape(train_output)
		if((fFile_i>0 and (fFile_i+1)%100==0)or(fFile_i==len(fileList)-1)):
			print "Saving files",fFile_i+1
			val_input = val_input[1:,:]
			train_input = train_input[1:,:]
			val_output = val_output[1:,:]
			train_output = train_output[1:,:]
			val_input, val_output = shuffle_in_unison(val_input,val_output)
			train_input, train_output = shuffle_in_unison(train_input,train_output)

			f1 = h5py.File(pathToSave+str((fFile_i)/100+1)+'_val_temp.hdf5', 'w')				# create dataset for validation
			f1.create_dataset('data', (val_input.shape[0], val_input.shape[1]), dtype='double')
			f1.create_dataset('label', (val_output.shape[0], val_output.shape[1]), dtype='double')
			f1['data'][...] = val_input
			f1['label'][...] = val_output
			f1.close()
			f2 = h5py.File(pathToSave+str((fFile_i)/100+1)+'_train_temp.hdf5', 'w')				# create dataset for training
			f2.create_dataset('data', (train_input.shape[0], train_input.shape[1]), dtype='double')
			f2.create_dataset('label', (train_output.shape[0], train_output.shape[1]), dtype='double')
			f2['data'][...] = train_input
			f2['label'][...] = train_output
			f2.close()
			del val_input,train_input,val_output,train_output,f1, f2
			val_input = np.zeros((1,inShape))
			train_input = np.zeros((1,inShape))
			val_output = np.zeros((1,outShape))
			train_output = np.zeros((1,outShape))
	
	print "Data created. Shuffling files..."
	if os.path.exists(datasetPath+"val.txt"):
		os.remove(datasetPath+"val.txt")
	print "Validation files"
	howManyCreated = len(glob(pathToSave+'*_val_temp.hdf5'))
	for i in xrange(howManyToCreate):
		print i
		val_input = np.zeros((1,inShape))
		val_output = np.zeros((1,outShape))

		for j in xrange(howManyCreated):
			
			with h5py.File(pathToSave+str(j+1)+'_val_temp.hdf5','r') as f:
				dataT = np.array(f.get('data'))
				labelT = np.array(f.get('label'))
				n = len(dataT)/howManyToCreate
			print "Val file read", j+1, n
			if(i!=(howManyToCreate-1)):
				val_input = np.concatenate((val_input,dataT[i*n:(i+1)*n,:]),axis=0)
				val_output = np.concatenate((val_output,labelT[i*n:(i+1)*n,:]),axis=0)
			elif(i==(howManyToCreate-1)):
				val_input = np.concatenate((val_input,dataT[i*n:,:]),axis=0)
				val_output = np.concatenate((val_output,labelT[i*n:,:]),axis=0)
			del dataT, labelT
		val_input = val_input[1:,:]
		val_output = val_output[1:,:]
		val_input, val_output = shuffle_in_unison(val_input, val_output)
		f = h5py.File(pathToSave+str(i+1)+'_val.hdf5', 'w')				# create dataset for validation
		f.create_dataset('data', (val_input.shape[0], val_input.shape[1]), dtype='double')
		f.create_dataset('label', (val_output.shape[0], val_output.shape[1]), dtype='double')
		assert len(val_input)==len(val_output)
		f['data'][...] = val_input
		f['label'][...] = val_output
		f.close()
		with open(datasetPath+"val.txt", 'a') as f1:
	    		f1.writelines(pathToSave+str(i+1)+'_val.hdf5'+"\n")
		del f,f1, val_input, val_output
	for j in xrange(howManyCreated):
		os.remove(pathToSave+str(j+1)+'_val_temp.hdf5')	


	print "Training files"
	if os.path.exists(datasetPath+"train.txt"):
		os.remove(datasetPath+"train.txt")
	howManyCreated = len(glob(pathToSave+'*_train_temp.hdf5'))
	for i in xrange(howManyToCreate):
		print i
		train_input = np.zeros((1,inShape))
		train_output = np.zeros((1,outShape))
		for j in xrange(howManyCreated):
			with h5py.File(pathToSave+str(j+1)+'_train_temp.hdf5','r') as f:
				dataT = np.array(f.get('data'))
				labelT = np.array(f.get('label'))
				n = len(dataT)/howManyToCreate
			print "Train file read", j+1, n
			if(i!=(howManyToCreate-1)):
				train_input = np.concatenate((train_input,dataT[i*n:(i+1)*n,:]),axis=0)
				train_output = np.concatenate((train_output,labelT[i*n:(i+1)*n,:]),axis=0)
			if(i==(howManyToCreate-1)):
				train_input = np.concatenate((train_input,dataT[i*n:,:]),axis=0)
				train_output = np.concatenate((train_output,labelT[i*n:,:]),axis=0)
			del dataT, labelT
		train_input = train_input[1:,:]
		train_output = train_output[1:,:]
		train_input, train_output = shuffle_in_unison(train_input, train_output)
		f = h5py.File(pathToSave+str(i+1)+'_train.hdf5', 'w')				# create dataset for validation
		f.create_dataset('data', (train_input.shape[0], train_input.shape[1]), dtype='double')
		f.create_dataset('label', (train_output.shape[0], train_output.shape[1]), dtype='double')
		assert len(train_input)==len(train_output)
		f['data'][...] = train_input
		f['label'][...] = train_output
		f.close()
		with open(datasetPath+"train.txt", 'a') as f1:
	    		f1.writelines(pathToSave+str(i+1)+'_train.hdf5'+"\n")
		del f,f1, train_input, train_output
	for j in xrange(howManyCreated):
		os.remove(pathToSave+str(j+1)+'_train_temp.hdf5')	
	
#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#
def main():
	d_name = "CUB_200_2011"

	#split_dataset_to_finetune("/media/jedrzej/Seagate/DATA/"+d_name+"/inception_features/", "/media/jedrzej/Seagate/DATA/"+d_name+"/inception_features_FT/", "/media/jedrzej/Seagate/DATA/"+d_name+"/inception_features_TEST/")


	createRegularTrainingData("/media/jedrzej/Seagate/DATA/ILSVRC2012/inception_features/", "/media/jedrzej/Seagate/DATA/ILSVRC2012/", "/media/jedrzej/Seagate/DATA/ILSVRC2012/img2bound_data_logistic/")
	#createRegularTrainingData("/media/jedrzej/Seagate/DATA/"+d_name+"/inception_features_FT/", "/media/jedrzej/Seagate/DATA/"+d_name+"/", "/media/jedrzej/Seagate/DATA/"+d_name+"/img2bound_data/")

if __name__ == "__main__":
    main()

