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
	def __init__(self, featuresPath, network_model, network_weigths, datasetName, layerName = 'dmt3_3', testIter = 10, noOfExamples = 15, inShape = 1024, outShape = 1025):
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
				if(self.inShape==2048): test = test[:,:1024]
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
	    	plt.matshow(confMatrix/float(self.testIter), vmin=0., vmax=1.0, cmap=plt.cm.binary)
	    	plt.colorbar()
	    	#plt.show()
		#plt.figure(figsize=(20,20))
		plt.savefig('./confusion_matrices/'+self.datasetName+'.png', dpi=200)


#--------------------------------------#
'''
def get_precision_recall(network, testingData, queryData, featuresFileList, inShape=1024, outShape=1025):
		layerName = 'dmt3_3'
		totalPrecision = []
		totalRecall = []
		print "Done getting data"
		for i, test_file in enumerate(testingData):
			category_name = featuresFileList[i].split("/")[-1].split(".")[0]	# get the name of a category
			testCurrCategory = np.asarray(testingData[i])	
			queryCurrCategory = np.asarray(queryData[i])
			noOfExamples = len(testCurrCategory)
			testIter = len(queryCurrCategory)
			negatives = np.asarray(testingData)
			negatives = np.delete(negatives,[i],axis=0).reshape((len(featuresFileList)-1)*noOfExamples,inShape)

			print category_name
			precision = 0
			recall = 0

			for idx in xrange(testIter):
				f = queryCurrCategory[idx,:].reshape(1,inShape)
				out = network.forward(data = f)[layerName].copy().reshape(outShape)
				tot, pos = test_SVM_hyperplane(testCurrCategory,out[1:],out[0])
				tot2, pos2 = test_SVM_hyperplane(negatives,out[1:],out[0])
				if(pos>0 or pos2>0): precision += round((float(pos)/(pos+pos2)),2)
				else: precision += 0
				recall += round((float(pos)/tot),2)
			precision = precision/float(testIter)
			recall = recall/float(testIter)
			print "Precision:",str(precision)
			print "Recall:",str(recall)
			print "F1-score:",str(2*(precision*recall)/(precision+recall))
			totalPrecision.append(precision)
			totalRecall.append(recall)
		return totalPrecision, totalRecall
'''	
		
DATA_ROOT = "/media/jedrzej/SAMSUNG/DATA/"
MODELS_ROOT = "/media/jedrzej/SAMSUNG/Python/models/"

#model = MODELS_ROOT+"im2bound_ILSVRC2012/deploy.prototxt"
#weights = MODELS_ROOT+"im2bound_ILSVRC2012/dmt_iter_150000.caffemodel"

#model = MODELS_ROOT+"im2bound_cub_200_2011/deploy.prototxt"
#weights = MODELS_ROOT+"im2bound_cub_200_2011/im2bound_iter_140000.caffemodel"

model = MODELS_ROOT+"img2bound_ILSVRC2012_double/deploy.prototxt"
weights = MODELS_ROOT+"img2bound_ILSVRC2012_double/img2bound_iter_215000.caffemodel"

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
'''
DOUBLE_test = tester(DATA_ROOT+"CALTECH_256/inception_features_double/", model, weights, "caltech256_d", layerName = 'dmt3', inShape=2048)
DOUBLE_test.test_DMT_on_data()



