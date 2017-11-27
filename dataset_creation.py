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

class dataset_creator():
	def __init__(self, datasetPath, featuresPath, inshape = 1024, outShape = 1025):
		self.datasetPath = datasetPath
		self.featuresPath = featuresPath
		self.inShape = inShape
		self.outShape = outShape

	def set_dataset_path(self, datasetPath):
		self.datasetPath = datasetPath

	def set_features_path(self, featuresPath):
		self.featuresPath = featuresPath

	def get_k_features_dict(self, k = 5):
		# k = no of examples per category
		if(os.path.isfile(self.datasetPath+str(k)+"featuresDict.pickle")==False):
			# if the features dictionary does not exist - create one
			featuresFileList = glob(self.featuresPath+'*hdf5')
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
			pickle.dump(random_features_dict, open(self.datasetPath+str(k)+"featuresDict.pickle", 'wb'))	# save the dictionary
		else:
			# if the features dictionary does exist - open it
			print "Opening",k,"features dictionary"
			with open(self.datasetPath+str(k)+"featuresDict.pickle", 'rb') as handle:
				random_features_dict = pickle.load(handle)
		return random_features_dict

	def get_negative_data(self, categoryName, featuresDictionary):
		negativeData = []
		for key, value in featuresDictionary.iteritems():
			if(key!=categoryName):
				for v in value:
					negativeData.append(v)
		return np.asarray(negativeData)

	def save_training_pairs(self, saveFileName, networkInput, networkOutput):
		networkInput, networkOutput = shuffle_in_unison(networkInput, networkOutput)		# Shuffle 100 categories together
		f = h5py.File(saveFileName, 'w')				# create dataset
		f.create_dataset('data', (networkInput.shape[0], networkInput.shape[1]), dtype='double')
		f.create_dataset('label', (networkOutput.shape[0], networkOutput.shape[1]), dtype='double')
		f['data'][...] = networkInput
		f['label'][...] = networkOutput
		f.close()
		del f

	def reshuffle_all_files(self, howManyToCreate, pathToSave, type_of_data='val'):
		
		# if the txt file exists, remove it
		if os.path.exists(self.datasetPath+type_of_data+".txt"):
			os.remove(self.datasetPath+type_of_data+".txt")

		howManyCreated = len(glob(pathToSave+'*_'+type_of_data+'_temp.hdf5'))	# read how many files should we include in reshuffling procedure
		# Create new file one by one:
		for i in xrange(howManyToCreate):
			print "Creating new hdf5 file number:",i+1
			net_input = np.zeros((1,self.inShape))
			net_output = np.zeros((1,self.outShape))

			# Iterate through old hdf5 files one by one:
			for j in xrange(howManyCreated):
				with h5py.File(pathToSave+str(j+1)+'_'+type_of_data+'_temp.hdf5','r') as f:
					dataT = np.array(f.get('data'))		# inputs
					labelT = np.array(f.get('label'))	# labels (outputs)
					n = len(dataT)/howManyToCreate 	# how much data to include in the new file
				print "Old file number", j+1,"read"
				if(i!=(howManyToCreate-1)):
					# if it's not the last file to be created
					# include respecting part of the old file in the new one
					net_input = np.concatenate((net_input,dataT[i*n:(i+1)*n,:]),axis=0)
					net_output = np.concatenate((net_output,labelT[i*n:(i+1)*n,:]),axis=0)
				elif(i==(howManyToCreate-1)):
					# if it is the last file to be created
					# include respecting part of the old file in the new one
					net_input = np.concatenate((net_input,dataT[i*n:,:]),axis=0)
					net_output = np.concatenate((net_output,labelT[i*n:,:]),axis=0)
				del dataT, labelT

			# save the data:
			self.save_training_pairs(pathToSave+str(i+1)+'_'+type_of_data+'.hdf5',net_input[1:,:], net_output[1:,:])
			# save hdf5 file path to txt file:
			with open(self.datasetPath+type_of_data+".txt", 'a') as f:
		    		f.writelines(pathToSave+str(i+1)+'_'+type_of_data+'.hdf5'+"\n")
			del f, net_input, net_output
	
		# after the resfhuffling, remove old hdf5 files:
		for j in xrange(howManyCreated):
			os.remove(pathToSave+str(j+1)+'_'+type_of_data+'_temp.hdf5')		

	def createTrainingData(self, pathToSave, howManyFilesToCreate = 5, classifier_type = 'SVM'):
		# classifier_type: 'SVM' or 'LOGISTIC_REGRESSION'


		# create folder to save the hdf5 files if one does not exist
		if not os.path.exists(pathToSave):
	    		os.makedirs(pathToSave)
		self.random_features_dict =  get_k_features_dict(7)	# get at random k features from every category
		fileList = glob(self.featuresPath+'*hdf5')
		
		# create structures to hold input/output pairs
		val_input = np.zeros((1,self.inShape))
		train_input = np.zeros((1,self.inShape))
		val_output = np.zeros((1,self.outShape))
		train_output = np.zeros((1,self.outShape))

		# Iterate through all .hdf5 files (all categories):
		for fFile_i, fFile in enumerate(fileList):
			sTime = time()	#measure time

			# read the features data inside the hdf5 file:
			with h5py.File(fFile,'r') as f:
				dataT = np.array(f.get('data'))
			del f
			
			# divide data into training/validation parts (70%/30%):
			data_length = len(dataT)
			n_val = int(0.3*data_length)
			val_indices = np.random.choice(data_length, n_val, replace=False)	# choose 30% of images at random for validation
			val_data = dataT[val_indices,...]
			train_data = np.delete(dataT,[val_indices],axis=0)

			# Prepare the data to create optimal decision boundary:
			category_name = fFile.split("/")[-1].split(".")[0]			# get category name
			negatives = self.get_negative_data(category_name, random_features_dict)	# get negative data for this cateogory
			pos_labels = np.ones((data_length))					# create positive labels (+1)				
			neg_labels = -1*np.ones((len(negatives)))				# create negative labels (-1)
			svm_data = np.concatenate((train_data,val_data,negatives),axis = 0)	# concatenate positive & negative training examples
			svm_labels = np.concatenate((pos_labels, neg_labels))			# concatenate positive & negative labels
			svm_data, svm_labels = shuffle_in_unison(svm_data, svm_labels)		# shuffle the data
			
			# Create oprimal decision boundary:
			assert classifier_type in ['SVM','LOGISTIC_REGRESSION']
			if(classifier_type = 'SVM'):
				clf = SVC(0.1,kernel = 'linear')			# define SVM classifier
			elif(classifier_type = 'LOGISTIC_REGRESSION'):
				clf = LogisticRegression()				# define Logistic Regression classifier
			clf.fit(svm_data, svm_labels)					# fit data to a classifier
			optimal_hyperplane = np.concatenate((np.asarray(clf.intercept_).copy().reshape((1,1)),clf.coef_.copy()),axis=1)	# concatenate [bias,weight]
			optimal_hyperplane = optimal_hyperplane.reshape(optimal_hyperplane.shape[1])	# reshape the output

			# Concatenate new input/output pairs
			val_input = np.concatenate((val_input,val_data),axis=0)
			train_input = np.concatenate((train_input,train_data),axis=0)
			val_output = np.concatenate((val_output,[optimal_hyperplane,]*val_data.shape[0]),axis=0)
			train_output = np.concatenate((train_output,[optimal_hyperplane,]*train_data.shape[0]),axis=0)
			del optimal_hyperplane, clf, train_data, val_data, pos_labels, neg_labels, negatives, svm_data,svm_labels, dataT,val_indices
			print "Category:",fFile_i+1,"took:",str(round((time()-sTime),2))	# measure time per category

			# Save every 100 categories:
			if((fFile_i>0 and (fFile_i+1)%100==0)or(fFile_i==len(fileList)-1)):
				print "Saving files",fFile_i+1
				self.save_training_pairs(pathToSave+str((fFile_i)/100+1)+'_val_temp.hdf5', val_input[1:,:], val_output[1:,:])		# save validation data
				self.save_training_pairs(pathToSave+str((fFile_i)/100+1)+'_train_temp.hdf5', train_input[1:,:], train_output[1:,:])	# save training data
				
				# Release memory and initialize new structures for next 100 categories
				del val_input,train_input,val_output,train_output
				val_input = np.zeros((1,self.inShape))
				train_input = np.zeros((1,self.inShape))
				val_output = np.zeros((1,self.outShape))
				train_output = np.zeros((1,self.outShape))
	

		# All data has been created. Now shuffle all together for better training:
		print "Data created. Shuffling files..."
		print "Validation files"
		self.reshuffle_all_files(howManyFilesToCreate, pathToSave, type_of_data='val')
		print "Training files"
		self.reshuffle_all_files(howManyFilesToCreate, pathToSave, type_of_data='train')
		
	
#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#
def main():
	DATA_ROOT = "/media/jedrzej/Seagate/DATA/"
	creator = dataset_creator(datasetPath = DATA_ROOT+"ILSVRC2012/", featuresPath = DATA_ROOT+"ILSVRC2012/inception_features/")
	creator.createTrainingData(DATA_ROOT+"ILSVRC2012/CLEAR_data/")
	

if __name__ == "__main__":
    main()

