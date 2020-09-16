BATCH_SIZE = 8
MAX_PEPTIDE_LENGTH = 5000

ALPHABET = 'LAGVESIKRDTPNQFYMHCWXUBZO'
EPITOPE_LENGTH = 7


def getArgs():

	import argparse

	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('-_id', dest='_id', default=None, type=str)
	parser.add_argument('-gpu', default=0, type=int)



	parser.add_argument('-noise', default=0.1, type=float, help='noise applied as a dropout layer on top of genome sequence')
	parser.add_argument('-opt', default='adam', type=str)				# sgd is always the best it seems ... and I want to play with LR

	parser.add_argument('-antibodies', default=15, type=int)
	parser.add_argument('-bn1', default=1, type=int)

	parser.add_argument('-flat', default='max', type=str)



	parser.add_argument('-thresh', default=0.5, type=float, help='noise applied as a dropout layer on top of genome sequence')
	parser.add_argument('-p1', default=2, type=int)


	parser.add_argument('-p2', default=2, type=int)
	parser.add_argument('-f2', default=256, type=int)
	parser.add_argument('-bn2', default=1, type=int)


	parser.add_argument('-p3', default=2, type=int)
	parser.add_argument('-f3', default=256, type=int)
	parser.add_argument('-bn3', default=1, type=int)


	#parser.add_argument('-flat', default='max', type=str)

	parser.add_argument('-fc1', default=64, type=int)
	parser.add_argument('-fc2', default=16, type=int)

	parser.add_argument('-batchNorm', dest='batchNorm', default=1, type=int)
	parser.add_argument('-batchSize', dest='batchSize', default=128, type=int)

	parser.add_argument('-dropout', dest='dropout', default=0.5, type=float)

	parser.add_argument('-learningRate', default=0.02, type=float)


	# these parameters are only used for the custom encoder
	parser.add_argument('-convSize', dest='convSize', default=5, type=int)
	parser.add_argument('-filters', default=64, type=int)
	parser.add_argument('-doubleLayers', default=1, type=int)

	parser.add_argument("-dry", action="store_true", help="Dont train. Just assemble model, print summary, and quit")
	parser.add_argument("-debug", action="store_true", help="For dev purposes ....")


	#parser.add_argument('-preprocess', default='vgg', type=str, help='Type of preprocessing to use (vgg, inception)')


	args = parser.parse_args()

	return args


