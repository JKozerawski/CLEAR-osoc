'''
Created on Oct 4, 2017

@author: jedrzej
'''
import numpy as np
import sys
CAFFE_ROOT = '/home/jedrzej/work/caffe/python'
sys.path.insert(0, CAFFE_ROOT) 
import caffe
from glob import glob
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve, f1_score
from sklearn.svm import OneClassSVM as ocsvm
import pickle

from dataset_feature_extraction import run_caffe_feature_extractor_on_list_of_images


class tester():
	def __init__(self, mainPath, imagesPath, networkModel, networkWeigths, datasetName, finalLayerName = 'dmt3_3', testIter = 10, noOfExamples = 20, inShape = 1024, outShape = 1025):
		self.network = caffe.Net(networkModel,networkWeigths, caffe.TEST)
		self.finalLayerName = finalLayerName
		self.testIter = testIter
		self.noOfExamples = noOfExamples
		self.inShape = inShape
		self.outShape = outShape
		self.noOfCategories = len(glob(self.featuresPath+'*hdf5'))
		self.datasetName = datasetName
		self.imagesPath = imagesPath
		self.mainPath = mainPath

	def set_main_path(self, mainPath):
		self.mainPath = mainPath
	
	def set_images_path(self, imagesPath):
		self.imagesPath = imagesPath

	def get_query_and_testing_data(self):
		imageList, categories = ([] for i in xrange(2)) # initialize lists
		categoryToImage = pickle.load( open( self.mainPath+"category-to-image.p", "rb" ) )
		for key, values in categoryToImage.iteritems():
			indices = np.random.choice(len(values), self.noOfExamples+1, replace=False)
			imgList.append([self.imagesPath+values[i] for i in indices])
			categories.append(key)
		imgList = [item for sublist in imgList for item in sublist]
	return imgList, categories

	def split_train_test(self, featureList, imageList):
		trainImages, testImages, trainFeatures, testFeatures = ([] for i in xrange(4))	# initialize lists
		for i in xrange(len(featureList)):
			if(i%(self.noOfExamples+1)==0):
				trainImages.append(imageList[i])
				trainFeatures.append(featureList[i,:])
			else:
				testImages.append(imageList[i])
				testFeatures.append(featureList[i,:])
		return trainImages, testImages, np.asarray(trainFeatures), np.asarray(testFeatures)

	def get_ground_truth_data(self, images, categories):
		imageToCategory = pickle.load( open( self.mainPath+"image-to-category.p", "rb" ) )
		groundTruth = []
		for i in xrange(len(categories)):
			categoryGroundTruth = []
			for j in xrange(len(images)):
				categoryGroundTruth.append(categories[i] in imageToCategory[images[j]])
			groundTruth.append(np.asarray(categoryGroundTruth).astype(int))
		return np.asarray(groundTruth)

	def get_query_OCSVM(self, queryImages):
		qOCSVM = []
		queryImages = queryImages.reshape(queryImages.shape[0],queryImages.shape[2])
		for i in xrange(len(queryImages)):
			clf = ocsvm(kernel = 'linear')
			clf.fit(queryImages[i,:].reshape(1,self.inShape))
			qOCSVM.append(clf)
			del clf
		return qOCSVM

	def get_logistic_regression_prob(self, points, weights, bias):
		logit = np.exp(bias+np.sum(np.multiply(points,weights),axis=1))
		prob = np.divide(logit,(np.ones(len(points))+logit))
		return prob

	def get_SVM_scores(self, points, weights, bias):
		return np.transpose(np.dot(weights,np.transpose(points))+np.transpose(bias))

	def get_MAP(self, yTrue, yScore):
		positives = np.where(yScore>=0)[0]
		MAP = 0.
		if(len(positives)>0):
			yTest  = [yTest[p] for p in positives]
			yScore  = yScore[np.where(yScore>0)[0]]
			if (np.isnan(average_precision_score(yTest, yScore))==False): 
				MAP = average_precision_score(yTest, yScore)
		return MAP

	def predict(self, trainFeatures, testFeatures, groundTruth, classifiersToTest = ['CLEAR_SVM', 'OCSVM', 'RANDOM']):
		MAP = dict()
		F1 = dict()
		for classifier in classifiersToTest:
			MAP[classifier] = []
			F1[classifier] = []

		if('CLEAR_SVM' in classifiersToTest):
			self.network.blobs['data'].reshape(trainFeatures.shape[0],self.inShape)
			CLEAR_boundaries = self.network.forward(data = np.asarray(trainFeatures).reshape(trainFeatures.shape[0],self.inShape))[self.finalLayerName].copy()
		if('OCSVM' in classifiersToTest):
			OCSVM_classifiers = self.get_query_OCSVM(trainFeatures)
		for i in xrange(len(trainFeatures)):
			# get preditions
			if('CLEAR_SVM' in classifiersToTest):
				CLEAR_score = self.get_SVM_scores(testFeatures,CLEAR_boundaries[i,1:],out[i,0])	# get classification score for CLEAR with SVM as classifier
				F1['CLEAR_SVM'] += f1_score(groundTruth[i,:], np.sign(CLEAR_score))		# get F1 score accuracy measure
				MAP['CLEAR_SVM'] += self.get_MAP(groundTruth[i,:], CLEAR_score)			# get Mean Average Precision score accuracy measure
			if('OCSVM' in classifiersToTest):
         			OCSVM_score = OCSVM_classifiers[i].decision_function(testFeatures).flatten()	# get classification score for One-Class SVM
				F1['OCSVM'] += f1_score(groundTruth[i,:], np.sign(OCSVM_score))			# get F1 score accuracy measure
				MAP['OCSVM'] += self.get_MAP(groundTruth[i,:], OCSVM_score)			# get Mean Average Precision score accuracy measure
			if('RANDOM' in classifiersToTest):
				RANDOM_score = np.random.random_sample((testFeatures.shape[0]))-.5		# get classification score for random choice
				F1['RANDOM'] += f1_score(groundTruth[i,:], np.sign(RANDOM_score))		# get F1 score accuracy measure
				MAP['RANDOM'] += self.get_MAP(groundTruth[i,:], RANDOM_score)			# get Mean Average Precision score accuracy measure
		
		# Print out the results
		for classifier in classifiersToTest:
			print "Accuracy for classification method:", classifier
			print "Mean Average Precision:", MAP[classifier]/trainFeatures.shape[0] 
			print "F1:", F1[classifier]/trainFeatures.shape[0] 

		# Return the results
		return np.asarray([MAP[i]/trainFeatures.shape[0], F1[i]/trainFeatures.shape[0]  for i in classifiersToTest])
			

	def get_average_accuracy(self):
		classifiers = ['CLEAR_SVM', 'OCSVM', 'RANDOM']
		accuracy = np.zeros((2*len(classifiers))
		for i in xrange(self.testIter):
			imagesList, categories = self.get_training_testing_images()
			features = run_caffe_feature_extractor_on_list_of_images(imagesList)
			trainImages, testImages, trainFeatures, testFeatures = self.split_train_test(features, imagesList)
			groundTruth = datasetTester.get_ground_truth_data(testImages, categories)
			accuracy+=self.predict(trainFeatures, testFeatures, groundTruth, classifiersToTest = classifiers)
		print self.testIter,"iterations are over. The average results are:"
		for i in xrange(len(classifiers)):
			print "Accuracy for classification method:", classifiers[i]
			print "Mean Average Precision:", accuracy[i*2]/self.testIter
			print "F1:", accuracy[1+i*2]/self.testIter


#------------------------------------------------------------------------------------------------------------------------


def main():
	DATA_ROOT = "/media/jedrzej/Seagate/DATA/"
	MODELS_ROOT = "/media/jedrzej/Seagate/Python/models/"
	datasets = ["CALTECH_256", "102flowers", "CUB_200_2011", "SUN_attribute", "indoorCVPR_09"]#, "Cars-196"]
	dset = 0
	model = MODELS_ROOT+"img2bound_ILSVRC2012/deploy.prototxt"
	weights = MODELS_ROOT+"img2bound_ILSVRC2012/dmt_iter_150000.caffemodel"

	datasetTester = tester(DATA_ROOT+datasets[dset], DATA_ROOT+datasets[dset]+"/Images/", model, weights, datasetName = datasets[dset])
	imagesList, categories = datasetTester.get_training_testing_images()
	features = run_caffe_feature_extractor_on_list_of_images(imagesList)
	trainImages, testImages, trainFeatures, testFeatures = datasetTester.split_train_test(features, imagesList)
	groundTruth = datasetTester.get_ground_truth_data(testImages, categories)
	accuracyMeasures = datasetTester.predict(trainFeatures, testFeatures, groundTruth, classifiersToTest = ['CLEAR_SVM', 'OCSVM', 'RANDOM'])

	
if __name__ == "__main__":
    main()
