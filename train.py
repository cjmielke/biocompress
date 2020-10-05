#from __future__ import print_function
import os

# first, build index mapping words in the embeddings set
# to their embedding vector
from models import buildModel
from settings import getArgs

from generators import prepSwissProt, batchGen

if __name__ == '__main__':

	args = getArgs()

	print(args)

	#assert args.fc2 <= args.fc1

	#if args.preprocess=='vgg': preprocess = preprocess_input_vgg
	#elif args.preprocess=='inception': preprocess = preprocess_input_inception
	#else: raise ValueError('Invalid preprocess function')

	os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)			# only use a specific GPU indicated on cmdline

	args = getArgs()

	instances = prepSwissProt()
	print(instances.head(15))


	G = batchGen(instances, args=args)


	numProteins = len(instances)

	model = buildModel(args, numProteins)

	#model.load_weights('model15_81p.h5', by_name=True)

	'''
	from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

	es = EarlyStopping(verbose=1, patience=10)
	lr = ReduceLROnPlateau(patience=5)
	fg = fgLogger(OUTDIR)
	'''

	#model.fit(x_train, y_train, batch_size=128, epochs=300, callbacks=[es, lr], validation_data=(x_val, y_val))

	#G, V, numInstances = getData(args)

	steps = int(len(instances)/args.batchSize)

	'''
	steps=100
	callbacks = [lr, es, fg]
	if not args._id or True:
		chk = ModelCheckpoint(filepath=OUTDIR+'/weights.{epoch:02d}-{val_loss:.2f}.hdf5', save_best_only=True)
		callbacks.append(chk)
	'''

	callbacks=[]

	try:
		model.fit_generator(G, steps_per_epoch=steps, epochs=10000, callbacks=callbacks, verbose=1)
	except KeyboardInterrupt:  # allow aborting but enable saving
		pass
	finally:
		model.save('model.h5')

#model.save('model.h5')
