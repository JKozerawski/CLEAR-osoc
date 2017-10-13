'''
Created on Oct 4, 2017

@author: jedrzej
'''
import h5py
import numpy as np
import sys
CAFFE_ROOT = '/home/jedrzej/work/caffe/python'
sys.path.insert(0, CAFFE_ROOT) 
import caffe
from glob import glob
from shuffle_in_unison import shuffle_in_unison
from matplotlib import pyplot as plt

class tester():
	def __init__(self, featuresPath, network_model, network_weigths, datasetName, layerName = 'fc5', testIter = 10, noOfExamples = 15, inShape = 1024, outShape = 1025):
		self.featuresPath = featuresPath
		self.network = caffe.Net(network_model,network_weigths, caffe.TEST)
		self.layerName = layerName
		self.testIter = testIter
		self.noOfExamples = noOfExamples
		self.inShape = inShape
		self.outShape = outShape
		self.noOfCategories = len(glob(self.featuresPath+'*hdf5'))
		self.datasetName = datasetName

	def get_query_and_testing_data(self):
		featuresFileList = glob(self.featuresPath+'*hdf5')
		queryData = []
		testingData = []
		for feat_file_i, feat_file in enumerate(featuresFileList):
			with h5py.File(feat_file,'r') as f:
				feats = np.array(f.get('data'))	

			all_indices = np.random.choice(len(feats), self.noOfExamples+1, replace=False)	# choose 20 images at random for testing and 1 as a query
			queryData.append(feats[all_indices[:1],...])
			testingData.append(feats[all_indices[1:],...])
			del f,feats,all_indices
		return queryData, testingData, featuresFileList


	def test_SVM_hyperplane(self, points, weights, bias):
		y = np.sign(np.transpose(np.dot(weights,np.transpose(points))+np.transpose(bias)))
		return len(y), len(np.where(y==1)[0])

	def get_confusion_matrix(self, testingData, querySVMs):
		confusionMatrix = np.zeros((self.noOfCategories,self.noOfCategories))
		for i in xrange(self.noOfCategories):
			test = np.asarray(testingData[i])	
			for j in xrange(self.noOfCategories):
				out = np.asarray(querySVMs[j]).reshape(self.outShape)
				tot, pos = self.test_SVM_hyperplane(test,out[1:],out[0])
				confusionMatrix[i,j] = float(pos)/tot
				del out, tot, pos	
			del test		
		return confusionMatrix

	def test_DMT_on_data(self):
		confMatrix = np.zeros((self.noOfCategories,self.noOfCategories))
		for k in xrange(self.testIter):
			queryData, testingData, featuresFileList = self.get_query_and_testing_data()		# get query and testing data for all categories
			self.network.blobs['data'].reshape(self.noOfCategories,self.inShape)
			querySVMs = self.network.forward(data = np.asarray(queryData).reshape(self.noOfCategories,self.inShape))[self.layerName].copy()
			confMatrix += self.get_confusion_matrix(testingData, querySVMs)

		# calculate precision and recall:
		
		mAP = 0.
		mAR = 0.
		for j in xrange(confMatrix.shape[1]):
			true_positives = confMatrix[j,j]
			false_negatives = 1.0 - true_positives
			false_positives = np.sum(confMatrix[:,j])-true_positives
			true_negatives = confMatrix.shape[0]-1.0 - false_positives
			precision = true_positives / (true_positives + false_positives)
			if(np.isnan(precision)): precision = 0.
			recall = true_positives / (true_positives + false_negatives)
			mAP += precision
			mAR += recall
		mAP = round((mAP / confMatrix.shape[1]), 2)	# mean Average Precision
		mAR = round((mAR / confMatrix.shape[1]), 2)	# mean Average Recall
		if(np.isinf(mAP)): mAP = 0
		if(np.isinf(mAR)): mAR = 0
	    	plt.matshow(confMatrix/float(self.testIter), vmin=0., vmax=1.0, cmap=plt.cm.binary)
	    	plt.colorbar()
		plt.suptitle('Precision = '+str(mAP)+' , Recall = '+str(mAR), fontsize=14, fontweight='bold')
	    	#plt.show()
		#plt.figure(figsize=(20,20))
		plt.savefig('./confusion_matrices/'+self.datasetName+'.png', dpi=200)
		plt.clf()
		return mAP, mAR

#-----------------------------------------------------------------------------#	
def testAllSnapshotsOnDataSet(dataset, N):
	plotPrecision = np.zeros((N))
	plotRecall = np.zeros((N))	
	for i in xrange(1,N+1):
		model = MODELS_ROOT+"img2bound_ILSVRC2012_NEW/deploy.prototxt"
		weights = MODELS_ROOT+"img2bound_ILSVRC2012_NEW/img2bound_iter_"+str(i)+"0000.caffemodel"
	
		datasetTester = tester(DATA_ROOT+dataset+"/inception_features/", model, weights, datasetName = dataset+"_"+str(i), layerName = 'fc5', inShape=1024)
		plotPrecision[i-1], plotRecall[i-1] = datasetTester.test_DMT_on_data()

	plotF1score = 2*np.multiply(plotPrecision,plotRecall)/(plotPrecision+plotRecall)

	plt.plot(plotPrecision, 'r')
	plt.xlabel('iteration')
	plt.ylabel('precision')
	plt.savefig("./testing_results/"+dataset+"_precision.png") #save image as png
	plt.clf()
	plt.plot(plotRecall, 'b')
	plt.xlabel('iteration')
	plt.ylabel('recall')
	plt.savefig("./testing_results/"+dataset+"_recall.png") #save image as png
	plt.clf()
	plt.plot(plotF1score, 'g')
	plt.xlabel('iteration')
	plt.ylabel('f1 score')
	plt.savefig("./testing_results/"+dataset+"_f1_score.png") #save image as png
	plt.clf()
#-----------------------------------------------------------------------------#	

DATA_ROOT = "/media/jedrzej/Seagate/DATA/"
MODELS_ROOT = "/media/jedrzej/Seagate/Python/models/"
testAllSnapshotsOnDataSet("CALTECH_256", 28)

datasets = ["CALETECH_256", "102flowers", "CUB_200_2011", "SUN_attribute", "indoorCVPR_09"]


#ILSVRC2012_test = tester(DATA_ROOT+"ILSVRC2012/inception_features/", model, weights, "ILSVRC2012")	
#ILSVRC2012_test.test_DMT_on_data()
'''
indoorCVPR_test = tester(DATA_ROOT+"indoorCVPR_09/inception_features/", model, weights, "indoorCVPR")	
indoorCVPR_test.test_DMT_on_data()

CALTECH256_test = tester(DATA_ROOT+"CALTECH_256/inception_features_double/", model, weights, "caltech256_d", layerName = 'dmt3', inShape=2048)
CALTECH256_test.test_DMT_on_data()

flowers_test = tester(DATA_ROOT+"102flowers/inception_features/", model, weights, "102flowers")
flowers_test.test_DMT_on_data()

cars196_test = tester(DATA_ROOT+"Cars-196/inception_features/", model, weights, "cars196")
cars196_test.test_DMT_on_data()

BIRDS_test = tester(DATA_ROOT+"CUB_200_2011/inception_features_TEST/", model, weights, "birds_fg3")
BIRDS_test.test_DMT_on_data()

SUN_test = tester(DATA_ROOT+"SUN_attribute/inception_features/", model, weights, "SUN")
SUN_test.test_DMT_on_data()

DOUBLE_test = tester(DATA_ROOT+"CALTECH_256/inception_features_double/", model, weights, "caltech256_d", layerName = 'dmt3', inShape=2048)
DOUBLE_test.test_DMT_on_data()
'''


