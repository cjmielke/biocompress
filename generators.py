#!/usr/bin/python
import gzip
from glob import glob
from itertools import cycle
import pandas
import gc
import numpy as np
from Bio import SeqIO


from settings import MAX_PEPTIDE_LENGTH, getArgs, ALPHABET


# loops through dataframe forever, but with random order each time
def randomGenerator(dataFrame):
	'''
	for rowNum, row in cycle(dataFrame.iterrows()):
		yield row
	'''
	while True:
		dataFrame = dataFrame.sample(frac=1)		# reshuffle
		for rowNum, row in dataFrame.iterrows():
			yield row
		gc.collect()


# stratified subsampling between two piles of data
def splitGen(A, B):
	gens = cycle((randomGenerator(A), randomGenerator(B)))
	for gen in gens:
		row = next(gen)
		#print row
		yield row


class SequenceEncoder():
	def __init__(self):
		self.alphabet = ALPHABET
		self.AAtoIndex = {k: v for v, k in enumerate(self.alphabet)}  # now this is a dictionary that maps an amino acid to 0-24
		print(self.AAtoIndex)

	@property
	def alphabetSize(self): return len(self.alphabet)

	def encode(self, sequence):
		# one-hot encoded for peptide AA-s (including special, like selenocysteine)
		seqVec = np.zeros( (1, MAX_PEPTIDE_LENGTH, len(self.alphabet)))         # channels-last dimension ordering seems backwards for this!
		#seqVec = np.zeros((1, len(self.alphabet), MAX_PEPTIDE_LENGTH))

		for pos, residue in enumerate(sequence):
			if pos >= MAX_PEPTIDE_LENGTH: break
			n = self.AAtoIndex.get(residue)
			if n: seqVec[0, pos, n] = 1

		return seqVec

encoder = SequenceEncoder()

def prepSwissProt(debug=False):
	instances = []

	record_iterator = SeqIO.parse("../data/human.fasta", "fasta")
	idx = 0
	for protein in record_iterator:
		if '_HUMAN' not in protein.id: continue     # only want human! :)
		if len(protein.seq) > MAX_PEPTIDE_LENGTH:
			print(protein.id, ' is larger than the max peptide length (', len(protein.seq),')')

		# encoding is kinda slow, so might do it on the fly during training
		#instance = dict(id=protein.id, sequenceVector=encoder.encode(protein.seq))
		instance = dict(id=protein.id, sequence=protein.seq, idx=idx)
		instances.append(instance)
		idx += 1

		if debug==True and idx>10: break                # FASTER loading times

	instances = pandas.DataFrame(instances)             # I like using dataframes for many training tasks, for convenience

	return instances



def getBPAexpressionData():

	DIR = '../data/GEO/GSE38234/'

	dataFrame = []

	sampleNames = glob(DIR+'*/*')
	for fPath in sampleNames:
		print(fPath)
		if 'GEN' in fPath: continue

		sampleName, fileName = fPath.replace(DIR, '').split('/')
		_, gsm, cellLine, condition, replicate = sampleName.split('_') # Supp_GSM937166_T47D_BPA_R2

		fh = gzip.open(fPath, 'r')

		for l in fh.readlines():
			l=l.split()#.rstrip()
			#print len(l)
			#print fh, l
			symbol, nmCode, rpkm = l

			row = dict(cellLine=cellLine, condition=condition, replicate=replicate, symbol=symbol, nmCode=nmCode, rpkm=float(rpkm))
			dataFrame.append(row)

	dataFrame = pandas.DataFrame(dataFrame)

	print(dataFrame.columns)
	print(dataFrame.head())

	print(len(dataFrame))

	return dataFrame


def getData(args, forever=True):

	expressionVals = getBPAexpressionData()
	numInstances = len(expressionVals)

	print(expressionVals.head())

	# print TCF7L2

	DF = expressionVals

	DF = DF.dropna()
	# DF = DF[(DF != 0).all(1)].dropna()

	# DF = expressionVals[expressionVals.symbol=='TCF7L2']

	# p = TCF7L2.pivot(columns=['condition'])
	# DF = DF.groupby(by)
	DF = DF.groupby(['cellLine', 'symbol', 'nmCode', 'condition'], as_index=False).mean()

	print(DF)

	DF = DF.pivot_table(values='rpkm', index=['cellLine', 'symbol', 'nmCode'], columns=['condition']).reset_index()

	print(DF)

	DF['FC'] = DF['BPA'] / DF['E2']

	DF['BPA'] = DF.BPA.apply(np.log2)
	DF['E2'] = DF.E2.apply(np.log2)

	# DF['logFC'] = DF['FC'].apply(np.log2)
	DF['logFC'] = DF['BPA'] - DF['E2']

	DF = DF.replace([np.inf, -np.inf], np.nan)
	DF = DF.dropna()

	print(DF)

	print(len(DF))

	DFa = DF[DF.logFC<-args.thresh]
	DFb = DF[DF.logFC>args.thresh]

	print(len(DFa), DFa.logFC.max())
	print(len(DFb), DFb.logFC.min())

	#DFa = DF[DF.logFC<-1.0]
	#DFb = DF[DF.logFC>1.0]

	#DFa = DF[DF.logFC<-2.0]
	#DFb = DF[DF.logFC>2.0]

	#DFa = DF[DF.logFC<-1.5]
	#DFb = DF[DF.logFC>1.5]


	DF = pandas.concat([DFa,DFb]).sample(frac=1)
	print(len(DFa), len(DFb))

	print(len(DF))

	expressionVals=DF

	print(expressionVals.columns)

	#c


	from readPromoters import promoterSequences


	condMap = dict(E2=0, BPA=1, DMSO=2)


	pSeqA = promoterSequences[promoterSequences.nm.isin(DFa.nmCode)]
	pSeqB = promoterSequences[promoterSequences.nm.isin(DFb.nmCode)]


	trainA, valA = np.split(pSeqA.sample(frac=1), [int(.9*len(pSeqA))])
	trainB, valB = np.split(pSeqB.sample(frac=1), [int(.9*len(pSeqB))])

	print(len(pSeqA), len(pSeqB))

	G = instanceGen(trainA, trainB, args, expressionVals)
	V = instanceGen(valA, valB, args, expressionVals)


	return G, V, numInstances


def batchGen(instances, args):
	"""

	:type instances: pandas.DataFrame
	"""
	#promoterSeqGen = generator(promoterSequences)
	#promoterSeqGen = splitGen(pSeqA, pSeqB)                 # generator guaruntees equal representation from two classes

	numProteins = len(instances)

	instanceGen = randomGenerator(instances)

	while True:
		batch = np.zeros((args.batchSize, MAX_PEPTIDE_LENGTH, encoder.alphabetSize))
		targets = np.zeros((args.batchSize, numProteins))  # prediction target will be to guess the protein! Softmax!

		# build the training batch!
		for i, _ in enumerate(batch):
			protein = next(instanceGen)
			seqVec = encoder.encode(protein.sequence)
			batch[i] = seqVec
			targets[i][protein.idx] = 1

		yield batch, targets




def filterSwissProt():
	'''grab the human sequences and store them in a smaller file'''
	record_iterator = SeqIO.parse("../data/uniprot_sprot.fasta", "fasta")
	humanRecords = [rec for rec in record_iterator if '_HUMAN' in rec.id]
	print('Found ', len(humanRecords), ' human proteins. Saving ....')
	SeqIO.write(humanRecords, "human.fasta", "fasta")




if __name__=='__main__':

	#filterSwissProt()
	instances = prepSwissProt(debug=True)
	print(instances.head(15))

	args = getArgs()

	G = batchGen(instances, args=args)
	for n in G:
		#print n
		pass

		#print row






