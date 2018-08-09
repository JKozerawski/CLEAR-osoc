import argparse
import caffe
from caffe_feature_extractor import CaffeFeatureExtractor

def preprocess(image_path, extractionNet):
	image = caffe.io.load(image_path)	# load the image
	return np.asarray(extractionNet.transformer.preprocess("data",image)/255.)	# preprocess the image
	
def run_net(clearNet, image_path, extractionNet):
	# read and preprocess images:
	image = preprocess(image_path, extractionNet)
	
	# extract feature witn Inception V1:
	extractionNet.blobs['data'].reshape(1,3,224,224)	# reshape input to make sure it matches size of the batch (in this example batch of size 1)
	feature = extractionNet.forward_all(**{extractionNet.inputs[0]: image, "blobs": ["pool/7x7_s1"]})["pool/7x7_s1"].copy()
	feature = feature.reshape(1,1024)

	# run the network:
	clearNet.blobs['data'].reshape(1,1024)	# reshape input to make sure it matches size of the batch (in this example batch of size 1)
	return clearNet.forward(data = np.asarray(feature))["dmt3_3"].copy()	# run the network and extract "dmt3_3" layer output


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--image',
		type=str,
		default="",
		help="Path to the image"
		)
	parser.add_argument(
		'--model_dir',
		type=str,
		default="./model/",
		help="Directory where the caffe model files are"
		)
	parser.add_argument(
		'--inception_model_dir',
		type=str,
		default="./model_inception/",
		help="Directory where the caffe model files are for the Inception V1 network"
		)

	FLAGS = parser.parse_args()
	extractionNet = CaffeFeatureExtractor(
		model_path = FLAGS.inception_model_dir+"googlenet_deploy.prototxt",
		pretrained_path = FLAGS.inception_model_dir+"bvlc_googlenet.caffemodel",
		blob = "pool/7x7_s1",
		crop_size = 224,
		mean_values = [104.0, 117.0, 123.0]		
		)
	clearNet = caffe.Net(FLAGS.model_dir+"deploy.prototxt", FLAGS.model_dir+"dmt_iter_150000.caffemodel", caffe.TEST)
    	CLEAR_boundary = run_net(clearNet, FLAGS.image, extractionNet)

