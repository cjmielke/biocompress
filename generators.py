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






