from keras import Input, Model
from keras.constraints import NonNeg
from keras.initializers import Zeros
from keras.layers import Dropout, Conv1D, Concatenate, MaxPooling1D, BatchNormalization, Flatten, GlobalMaxPooling1D, \
	GlobalAveragePooling1D, Dense, Activation, GaussianNoise
from keras.utils import plot_model

from settings import MAX_PEPTIDE_LENGTH, ALPHABET, EPITOPE_LENGTH

from keras import backend as K


def binaryActivationFromTanh(x) :
	threshold = 0.0
	activated_x = K.tanh(x)
	#binary_activated_x = (activated_x > threshold)#*1.0
	binary_activated_x = K.round(activated_x)
	#binary_activated_x = K.cast_to_floatx(binary_activated_x)
	return binary_activated_x

def buildModel(args, numProteins):

	sequence = Input(shape=(MAX_PEPTIDE_LENGTH, len(ALPHABET)), name='sequence')
	x = sequence

	#x = Dropout(args.noise)(x)			# dropout on input might work ....

	constraint = NonNeg()
	#constraint=None
	initializer = Zeros()
	initializer='glorot_uniform'
	from keras import regularizers
	#reg=regularizers.l1(0.00001)
	reg=None
	x = Conv1D( args.antibodies, (EPITOPE_LENGTH),
	            kernel_constraint=constraint, kernel_initializer=initializer, kernel_regularizer=reg,
	            activation=None, padding='same')(x)

	#x = Activation('sigmoid', name='sigmoid')(x)
	x = Activation('tanh', name='tanh')(x)

	#x = Activation(binaryActivationFromTanh)(x)

	#if args.bn1: x = BatchNormalization()(x)


	# "flat" is my terrible lingo for different pooling types ....
	# for antibodies binding to proteins, I think Max Pooling makes sense!
	# IE, if an antibody binds anywhere to a peptide sequence, it bound to the protein

	flat = None
	if args.flat == 'flat':
		flat = Flatten(name='flatten')(x)
	elif args.flat == 'max':
		flat = GlobalMaxPooling1D(name='flatten')(x)
	elif args.flat == 'avg':
		flat = GlobalAveragePooling1D(name='flatten')(x)
	assert flat is not None

	x = flat

	# Now that pooling has occured, all we have remaining is a single vector
	# describing which antibody bound to a given protein. The activation function above
	# (either sigmoid or tanh) constrains the binding event to yes/no answers
	# but I am still concerned that teeny tiny values could be encoding more "information"
	# than I want the neural network to have.
	# Ideally, one would merely binarize the output of this layer, but this produces a
	# non-differentiable layer that cannot be trained.
	# For now, however, a simple way to fight this is to add some gaussian noise

	#x = GaussianNoise(0.1)(x)

	#x = Activation('sigmoid', name='sigmoid2')(x)


	#x = BatchNormalization()(flat)
	x = Dropout(args.dropout)(x)        # HMMMM ..... what would the interpretation be?!

	'''
	if args.fc2:
		x = Dense(args.fc2, activation='relu')(x)
		x = Dropout(args.dropout)(x)
	'''

	protein = Dense(numProteins, activation='softmax', name='prediction')(x)



	model = Model(inputs=[sequence], outputs=[protein])
	#model.compile(loss='mse', optimizer='adam')

	loss = 'categorical_crossentropy'
	#loss = 'mse'
	model.compile(loss=loss, optimizer=args.opt, metrics=['acc'])
	#model.compile(loss='binary_crossentropy', optimizer=args.opt, metrics=['acc'])
	try: plot_model(model, to_file='model.png', show_shapes=True)
	except:
		print("NOTE: Couldn't render model.png")

	print(model.summary())

	return model
