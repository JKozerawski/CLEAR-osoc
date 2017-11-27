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
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve, f1_score
from sklearn.svm import OneClassSVM as ocsvm
import itertools
import matplotlib

matplotlib.rcParams.update({'font.size': 10})

class tester():
	def __init__(self, featuresPath, network_model, network_weigths, datasetName, layerName = 'fc5', testIter = 10, noOfExamples = 15, inShape = 1024, outShape = 1025):
		self.featuresPath = featuresPath
		self.network = caffe.Net(network_model,network_weigths, caffe.TEST)
		self.layerName = layerName
		self.testIter = testIter
		self.noOfExamples = noOfExamples
		self.inShape = inShape
		self.outShape = outShape
		if(featuresPath): self.noOfCategories = len(glob(self.featuresPath+'*hdf5'))
		else: self.noOfCategories = 0
		self.datasetName = datasetName

	def get_query_and_testing_data(self):
		featuresFileList = list(np.random.choice(glob(self.featuresPath+'*hdf5'),self.noOfCategories, replace=False))
		#featuresFileList = glob(self.featuresPath+'*hdf5')
		queryData = []
		testingData = []
		for feat_file_i, feat_file in enumerate(featuresFileList):
			with h5py.File(feat_file,'r') as f:
				feats = np.array(f.get('data'))	
			all_indices = np.random.choice(len(feats), self.noOfExamples+1, replace=False)	# choose 20 images at random for testing and 1 as a query
			queryData.append(feats[all_indices[:1],...])
			testingData.append(feats[all_indices[1:],...])
			del f,feats,all_indices
		return queryData, testingData


	def test_SVM_hyperplane(self, points, weights, bias):
		y = np.sign(np.transpose(np.dot(weights,np.transpose(points))+np.transpose(bias)))
		return len(y), len(np.where(y==1)[0])

	def get_SVM_scores(self, points, weights, bias):
		return np.transpose(np.dot(weights,np.transpose(points))+np.transpose(bias))

	def get_logistic_regression_prob(self, points, weights, bias):
		#logit = np.exp(bias+np.sum(np.multiply(points, weights)))
		#prob.append(logit/(1+logit))
		logit = np.exp(bias+np.sum(np.multiply(points,weights),axis=1))
		prob = np.divide(logit,(np.ones(len(points))+logit))
		return prob

	def get_query_OCSVM(self, queryImages):
		qOCSVM = []
		queryImages = queryImages.reshape(queryImages.shape[0],queryImages.shape[2])
		for i in xrange(len(queryImages)):
			clf = ocsvm(kernel = 'linear')
			clf.fit(queryImages[i,:].reshape(1,1024))
			qOCSVM.append(clf)
			del clf
		return qOCSVM
	
	def get_mAP(self, y_test, y_score):
		positives = np.where(y_score>=0)[0]
		mAP = 0.
		if(len(positives)>0):
			y_test  = [y_test[p] for p in positives]
			y_score  = y_score[np.where(y_score>=0)[0]]
			if (np.isnan(average_precision_score(y_test, y_score))==False): 
				mAP = average_precision_score(y_test, y_score)
		return mAP

	def get_mAP_scores(self, testingData, querySVMs, query_ocsvms):
		testingData = np.asarray(testingData)
		testingData = testingData.reshape(testingData.shape[0]*testingData.shape[1],testingData.shape[2])
		mAP = dict()
		y_score = dict()
		mAP["img2bound"] = []
		mAP["random"] = []
		mAP["ocsvm"] = []

		for j in xrange(self.noOfCategories):
			# iterate through query images
			out = np.asarray(querySVMs[j]).reshape(self.outShape)	# get svm hyperplane
			
			y_score["img2bound"] = self.get_SVM_scores(testingData,out[1:],out[0])	# get svm-score for img2bound method
			#y_score["img2bound"] = self.get_logistic_regression_prob(testingData,out[1:],out[0])
			y_score["random"] = np.random.random_sample((len(testingData)))-.5	# get svm-score for random choice
			y_score["ocsvm"] = query_ocsvms[j].decision_function(testingData).flatten()
			y_test = np.zeros((len(testingData)))
			y_test[j*self.noOfExamples:(j+1)*self.noOfExamples] = 1
			mAP["img2bound"].append(self.get_mAP(y_test, y_score["img2bound"]))
			mAP["random"].append(self.get_mAP(y_test, y_score["random"]))
			mAP["ocsvm"].append(self.get_mAP(y_test, y_score["ocsvm"]))
		mAP["img2bound"] = np.asarray(mAP["img2bound"])
		mAP["random"] = np.asarray(mAP["random"])
		mAP["ocsvm"] = np.asarray(mAP["ocsvm"])

		return np.mean(mAP["img2bound"]), np.mean(mAP["random"]), np.mean(mAP["ocsvm"])

	def get_F1_scores(self, testingData, querySVMs, query_ocsvms):
		testingData = np.asarray(testingData)
		testingData = testingData.reshape(testingData.shape[0]*testingData.shape[1],testingData.shape[2])
		F1 = dict()
		y_pred = dict()
		F1["img2bound"] = 0.
		F1["random"] = 0.
		F1["ocsvm"] = 0.
		for j in xrange(self.noOfCategories):
			# iterate through query images
			out = np.asarray(querySVMs[j]).reshape(self.outShape)	# get svm hyperplane
			y_pred["img2bound"] = np.sign(self.get_SVM_scores(testingData,out[1:],out[0]))	# get svm-pred for img2bound method
			#y_pred["img2bound"] = np.sign(self.get_logistic_regression_prob(testingData,out[1:],out[0])-0.5)
			y_pred["random"] = np.sign(np.random.random_sample((len(testingData)))-.5)	# get svm-pred for random choice
			y_pred["ocsvm"] = np.sign(query_ocsvms[j].decision_function(testingData).flatten())
			y_test = np.zeros((len(testingData)))
			y_test[j*self.noOfExamples:(j+1)*self.noOfExamples] = 1
			F1["img2bound"] += f1_score(y_test, y_pred["img2bound"])
			F1["random"] += f1_score(y_test, y_pred["random"])
			F1["ocsvm"] += f1_score(y_test, y_pred["ocsvm"])
		F1["img2bound"] /= self.noOfCategories
		F1["random"] /= self.noOfCategories
		F1["ocsvm"] /= self.noOfCategories
		return np.asarray([F1["img2bound"], F1["random"], F1["ocsvm"]])

	def get_ROC_curve(self, testingData, querySVMs, query_ocsvms):
		testingData = np.asarray(testingData)
		testingData = testingData.reshape(testingData.shape[0]*testingData.shape[1],testingData.shape[2])
		fpr = dict()
		tpr = dict()
		roc_auc = dict()

		y_test = np.zeros((len(testingData),self.noOfCategories))
		y_score = np.zeros((len(testingData),self.noOfCategories))
		y_score_rand = np.zeros((len(testingData),self.noOfCategories))
		y_score_ocsvm = np.zeros((len(testingData),self.noOfCategories))

		for j in xrange(self.noOfCategories):
			# iterate through query images
			out = np.asarray(querySVMs[j]).reshape(self.outShape)	# get svm hyperplane
			y_score[:,j] = self.get_SVM_scores(testingData,out[1:],out[0])	# get svm-score for img2bound method
			#y_score[:,j] = self.get_logistic_regression_prob(testingData,out[1:],out[0])	# get logistic regression score for img2bound method
			y_score_rand[:,j] = np.random.random_sample((len(testingData)))-.5	# get svm-score for random choice
			y_score_ocsvm[:,j] = query_ocsvms[j].decision_function(testingData).flatten() # get svm-score for ocsvm
			y_test[j*self.noOfExamples:(j+1)*self.noOfExamples,j] = 1	# get true labels

		# Compute micro-average ROC curve and ROC area
		fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
		roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
		
		fpr["micro_rand"], tpr["micro_rand"], _ = roc_curve(y_test.ravel(), y_score_rand.ravel())
		roc_auc["micro_rand"] = auc(fpr["micro_rand"], tpr["micro_rand"])

		fpr["micro_ocsvm"], tpr["micro_ocsvm"], _ = roc_curve(y_test.ravel(), y_score_rand.ravel())
		roc_auc["micro_ocsvm"] = auc(fpr["micro_ocsvm"], tpr["micro_ocsvm"])
		
		# plot ROC curves
		'''
		plt.figure()
		lw = 2
		plt.plot(fpr["micro"], tpr["micro"], color='darkorange', lw=lw, label='auc = %0.2f' % roc_auc["micro"])
		#plt.plot(fpr["micro_rand"], tpr["micro_rand"], color='b', lw=lw, label='chance = %0.2f' % roc_auc["micro_rand"])
		#plt.plot(fpr["micro_ocsvm"], tpr["micro_ocsvm"], color='r', lw=lw, linestyle='--', label='ocsvm = %0.2f' % roc_auc["micro_ocsvm"])
		#plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('ROC curve')
		plt.legend(loc="lower right")
		#plt.savefig('./'+self.datasetName+'_ROC.png', dpi=200)
		#plt.show()
		'''
		return [fpr["micro"], tpr["micro"],roc_auc["micro"]]


	def get_confusion_matrix(self, testingData, querySVMs):
		confusionMatrix = np.zeros((self.noOfCategories,self.noOfCategories))
		for i in xrange(self.noOfCategories):
			# iterate through test categories
			test = np.asarray(testingData[i])	# get testing data	
			for j in xrange(self.noOfCategories):
				# iterate through query images
				out = np.asarray(querySVMs[j]).reshape(self.outShape)	# get svm hyperplane
				tot, pos = self.test_SVM_hyperplane(test,out[1:],out[0])
				confusionMatrix[i,j] = float(pos)/tot
				del out, tot, pos	
			del test		
		return confusionMatrix

	def test_img2bound_on_data(self):
		confMatrix = np.zeros((self.noOfCategories,self.noOfCategories))
		for k in xrange(self.testIter):
			queryData, testingData = self.get_query_and_testing_data()		# get query and testing data for all categories
			self.network.blobs['data'].reshape(self.noOfCategories,self.inShape)
			querySVMs = self.network.forward(data = np.asarray(queryData).reshape(self.noOfCategories,self.inShape))[self.layerName].copy()
			confMatrix += self.get_confusion_matrix(testingData, querySVMs)
		confMatrix = confMatrix/float(self.testIter)
		
		recall = 0
		precision = 0
		for i in xrange(self.noOfCategories):
			recall += confMatrix[i,i]
			precision += confMatrix[i,i] / np.sum(confMatrix[:,i])
		print "Recall", recall/self.noOfCategories
		print "Precision", precision/self.noOfCategories

	    	plt.matshow(confMatrix, vmin=0., vmax=1.0, cmap=plt.cm.binary)
	    	plt.colorbar()
	    	plt.xlabel('Classifier')
		plt.ylabel('Category')
		plt.title('MIT Indoor 67')
		plt.xticks([],[])
		plt.yticks([],[])
		#plt.figure(figsize=(20,20))
		#plt.savefig('./confusion_matrices/'+self.datasetName+'.png', dpi=300)
		#plt.clf()
		#plt.show()

	def get_accuracy_on_data(self, percentOfCategoriesToTest = 100):
		F1 = np.zeros((3))
		mAP = np.zeros((3))
		self.noOfCategories = int(self.noOfCategories * percentOfCategoriesToTest/100.)
		for k in xrange(self.testIter):
			queryData, testingData = self.get_query_and_testing_data()		# get query and testing data for all categories
			queryOCSVMS = self.get_query_OCSVM(np.asarray(queryData))
			self.network.blobs['data'].reshape(self.noOfCategories,self.inShape)
			querySVMs = self.network.forward(data = np.asarray(queryData).reshape(self.noOfCategories,self.inShape))[self.layerName].copy()
			mAP += self.get_mAP_scores(testingData, querySVMs, queryOCSVMS)
			F1 += self.get_F1_scores(testingData, querySVMs, queryOCSVMS)
			roc_info = self.get_ROC_curve(testingData, querySVMs, queryOCSVMS)
		return [mAP/self.testIter, F1/self.testIter, roc_info]
	
#-----------------------------------------------------------------------------#	

DATA_ROOT = "/media/jedrzej/Seagate/DATA/"
MODELS_ROOT = "/media/jedrzej/Seagate/Python/models/"
datasets = ["CALTECH_256", "102flowers", "CUB_200_2011", "SUN_attribute", "indoorCVPR_09"]#, "Cars-196"]

def main():
	dset = 0

	#model_main = MODELS_ROOT+"img2bound_ILSVRC2012_6_layers/deploy.prototxt"
	#weights_main = MODELS_ROOT+"img2bound_ILSVRC2012_6_layers/img2bound_iter_10000.caffemodel"
	'''
	datasetTester = tester(DATA_ROOT+datasets[dset]+"/inception_features/", model, weights, datasetName = datasets[dset], layerName = 'fc6', noOfExamples = 20, inShape=1024)
	datasetTester.get_accuracy_on_data()
	'''


	'''

	model_main = MODELS_ROOT+"img2bound_ILSVRC2012_logistic/deploy.prototxt"
	weights_main = MODELS_ROOT+"img2bound_ILSVRC2012_logistic/dmt_iter_320000.caffemodel"

	roc = []
	for dset in xrange(len(datasets)):
		datasetTester = tester(DATA_ROOT+datasets[dset]+"/inception_features/", model_main, weights_main, datasetName = datasets[dset], layerName = 'dmt3_3', testIter = 20, noOfExamples = 20, inShape=1024)
		mAP_main, F1_main, roc_info = datasetTester.get_accuracy_on_data()
		print datasets[dset]
		print mAP_main
		print F1_main
		roc.append(roc_info)
	'''



	#datasetTester = tester(DATA_ROOT+datasets[dset]+"/inception_features/", model_main, weights_main, datasetName = datasets[dset], layerName = 'dmt3_3', testIter = 20, noOfExamples = 20, inShape=1024)
	#mAP_main, F1_main, roc_info = datasetTester.get_accuracy_on_data()
	#print mAP_main, F1_main, roc_info
	'''
	plt.figure()
	lw = 2
	plt.plot(roc[0][0], roc[0][1], color='black', lw=lw, label='Caltech 256, auc = %0.2f' % roc[0][2])
	plt.plot(roc[2][0], roc[2][1], color='turquoise', lw=lw, label='CUB-200-2011, auc = %0.2f' % roc[2][2])
	plt.plot(roc[1][0], roc[1][1], color='red', lw=lw, label='Flowers 102, auc = %0.2f' % roc[1][2])
	plt.plot(roc[4][0], roc[4][1], color='darkorange', lw=lw, label='MIT Indoor 67, auc = %0.2f' % roc[4][2])
	plt.plot(roc[3][0], roc[3][1], color='green', lw=lw, label='SUN attributes, auc = %0.2f' % roc[3][2])
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC curves')
	plt.legend(loc="lower right")
	plt.savefig('./ROC.png', dpi=300)
	plt.show()
	'''


	'''
	model = MODELS_ROOT+"img2bound_ILSVRC2012/deploy.prototxt"
	weights = MODELS_ROOT+"img2bound_ILSVRC2012/dmt_iter_150000.caffemodel"
	mAP = []
	F1 = []
	r= [10,20,30,40,50,60,70,80,90,100]
	for i in xrange(1,11):
		datasetTester = tester(DATA_ROOT+datasets[dset]+"/inception_features/", model, weights, datasetName = datasets[dset], layerName = 'dmt3_3', testIter = 10, noOfExamples = 20, inShape=1024)
		mAP_temp, F1_temp = datasetTester.get_accuracy_on_data(percentOfCategoriesToTest = i*10)
		mAP.append(mAP_temp)
		F1.append(F1_temp)
	mAP = np.asarray(mAP)
	F1 = np.asarray(F1)

	plt.figure()
	lw = 2
	plt.plot(r, mAP[:,0], color='darkorange', lw=lw, label='img2bound [ours]')
	plt.plot(r, mAP[:,1], color='b', lw=lw, label='chance')
	plt.plot(r, mAP[:,2], color='r', lw=lw, linestyle='--', label='ocsvm')
	#plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0, 100])
	plt.ylim([0.0, 1.0])
	plt.xlabel('Percent of categories present')
	plt.ylabel('mAP')
	plt.title('mAP as function of # of categories')
	plt.legend(loc="lower right")
	plt.show()
	plt.clf()
	plt.close()

	plt.figure()
	lw = 2
	plt.plot(r, F1[:,0], color='darkorange', lw=lw, label='img2bound [ours]')
	plt.plot(r, F1[:,1], color='b', lw=lw, label='chance')
	plt.plot(r, F1[:,2], color='r', lw=lw, linestyle='--', label='ocsvm')
	#plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0, 100])
	plt.ylim([0.0, 1.0])
	plt.xlabel('Percent of categories present')
	plt.ylabel('F1')
	plt.title('F1 as function of # of categories')
	plt.legend(loc="lower right")
	plt.show()
	'''
if __name__ == "__main__":
    main()
