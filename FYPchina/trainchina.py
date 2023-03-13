#Building and training LSTM Network with dataset from preprocessing,
#to generate melodies.
#Time series music. Generating/Generative music


import time
import tensorflow as tf #measure performance on training
from matplotlib import pyplot as plt #pictures of performance in training the model
import tensorflow.keras as keras
from preprocesschina import generate_training_sequences, SEQUENCE_LENGTH


#38 because length of vocabulary in this case (erk data file)
OUTPUT_UNITS = 45
#ERROR function used for training
LOSS = "sparse_categorical_crossentropy"
LEARNING_RATE = 0.001
#number the neurons in the internal layesr of the network
#list because possible more than one internal layer
#in this case only one hidden layer with 256 hidden neurons
NUM_UNITS =[256]
#2 layers
#NUM_UNITS =[256, 256]
#40 - 100 is okay number of epochs
EPOCHS = 50
#number of samples that Network will see before running backpropagation
BATCH_SIZE = 64
#.h5 because keras save models like that (it stores all the information of the model)
SAVE_MODEL_PATH = "modelchina.h5"


#generating and compiling the model
def build_model(output_units,num_units, loss, learning_rate):
    #1.create the model arquitecture
    #output_units = vocabulary size
    input = keras.layers.Input(shape = (None, output_units))
    x = keras.layers.LSTM(num_units[0])(input)
    #Dropuout is a technique to avoid overfitting
    x = keras.layers.Dropout(0.2)(x)
    output = keras.layers.Dense(output_units, activation = "softmax")(x)
    #build the model after have input and output
    model = keras.Model(input, output)
    #2.compile model
    # https://machinelearningmastery.com/custom-metrics-deep-learning-keras-python/
    # https://keras.io/api/metrics/ 
    model.compile(loss = loss, optimizer = keras.optimizers.Adam(lr = learning_rate), metrics = ["accuracy", "mse", "mape",
    'mae', "msle", "categorical_accuracy", "binary_accuracy", "hinge", "sparse_top_k_categorical_accuracy", "top_k_categorical_accuracy",
#    "kullback_leibler_divergence", 
    tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.CosineSimilarity(axis=1),
    tf.keras.metrics.LogCoshError(), tf.keras.metrics.SquaredHinge(), tf.keras.metrics.CategoricalHinge() ])
    """
    mse = "mean_squared_error",
    mape = "mean_absolute_percentage_error",
    mae = "mean_absolute_error",
    msle = tf.keras.metrics.MeanSquaredLogarithmicError(),
    #########################################################
    accuracy = "sparse_categorical_accuracy",
    loss = "sparse_categorical_crossentropy",
    tf.keras.metrics.CosineSimilarity(axis=1) = "cosine_proximity",
    """
    #print all the layers of the model(visual feedback)
    model.summary()
    return model


#to train the Network(high level function). Arguments have default values.
def train(output_units = OUTPUT_UNITS, num_units = NUM_UNITS, loss = LOSS, learning_rate= LEARNING_RATE):
    #generate the training sequences
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)
    #build the network/graph (build the model)
    model = build_model(output_units, num_units, loss, learning_rate)
    #train the model
    model.fit(inputs, targets, epochs = EPOCHS, batch_size = BATCH_SIZE)
    #save the model(reuse, call back)
    model.save(SAVE_MODEL_PATH)
    history =  model.fit(inputs, targets, epochs = EPOCHS, batch_size = BATCH_SIZE)
    print(history.history.keys())
    # plot metrics
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
#    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('accuracy.png', dpi = 500)
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
#    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig('loss.png', dpi = 500)
    plt.show()
    # sumarise history for mse
    plt.plot(history.history['mse'])
    plt.title('model mse')
    plt.ylabel('mse')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig('mse.png', dpi = 500)
    plt.show()
    # sumarise history for mape
    plt.plot(history.history['mape'])
    plt.title('model mape')
    plt.ylabel('mape')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('mape.png', dpi = 500)
    plt.show()
    # sumarise history for mae
    plt.plot(history.history['mae'])
    plt.title('model mae')
    plt.ylabel('mae')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig('mae.png', dpi = 500)
    plt.show()
    # sumarise history for msle
    plt.plot(history.history['msle'])
    plt.title('model msle')
    plt.ylabel('msle')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig('msle.png', dpi = 500)
    plt.show()
    # sumarise history for categorical_accuracy
    plt.plot(history.history['categorical_accuracy'])
    plt.title('model categorical_accuracy')
    plt.ylabel('categorical_accuracy')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig('categorical_accuracy.png', dpi = 500)
    plt.show()
    # sumarise history for binary_accuracy
    plt.plot(history.history['binary_accuracy'])
    plt.title('model binary_accuracy')
    plt.ylabel('binary_accuracy')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig('binary_accuracy.png', dpi = 500)
    plt.show()
    # sumarise history for hinge
    plt.plot(history.history['hinge'])
    plt.title('model hinge')
    plt.ylabel('hinge')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig('hinge.png', dpi = 500)
    plt.show()
    # sumarise history for sparse_top_k_categorical_accuracy
    plt.plot(history.history['sparse_top_k_categorical_accuracy'])
    plt.title('model sparse_top_k_categorical_accuracy')
    plt.ylabel('sparse_top_k_categorical_accuracy')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig('sparse_top_k_categorical_accuracy.png', dpi = 500)
    plt.show()
    # sumarise history for top_k_categorical_accuracy
    plt.plot(history.history['top_k_categorical_accuracy'])
    plt.title('model top_k_categorical_accuracy')
    plt.ylabel('top_k_categorical_accuracy')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig('top_k_categorical_accuracy.png', dpi = 500)
    plt.show()
    '''
    # sumarise history for kullback_leibler_divergence
    plt.plot(history.history['kullback_leibler_divergence'])
    plt.title('model kullback_leibler_divergence')
    plt.ylabel('kullback_leibler_divergence')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig('kullback_leibler_divergence.png', dpi = 500)
    plt.show()
    '''
    # sumarise history for root_mean_squared_error
    plt.plot(history.history['root_mean_squared_error'])
    plt.title('model root_mean_squared_error')
    plt.ylabel('root_mean_squared_error')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig('root_mean_squared_error.png', dpi = 500)
    plt.show()
    # sumarise history for cosine_similarity
    plt.plot(history.history['cosine_similarity'])
    plt.title('model cosine_similarity')
    plt.ylabel('cosine_similarity')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig('cosine_similarity.png', dpi = 500)
    plt.show()
    # sumarise history for logcosh
    plt.plot(history.history['logcosh'])
    plt.title('model logcosh')
    plt.ylabel('logcosh')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig('logcosh.png', dpi = 500)
    plt.show()
    # sumarise history for squared_hinge
    plt.plot(history.history['squared_hinge'])
    plt.title('model squared_hinge')
    plt.ylabel('squared_hinge')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig('squared_hinge.png', dpi = 500)
    plt.show()
    # sumarise history for categorical_hinge
    plt.plot(history.history['categorical_hinge'])
    plt.title('model categorical_hinge')
    plt.ylabel('categorical_hinge')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig('categorical_hinge.png', dpi = 500)
    plt.show()
    # all lines going up in one image
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['mse'])
    plt.plot(history.history['msle'])
    plt.plot(history.history['hinge'])
    plt.plot(history.history['sparse_top_k_categorical_accuracy'])
#    plt.plot(history.history['kullback_leibler_divergence'])
    plt.plot(history.history['root_mean_squared_error'])
    plt.plot(history.history['squared_hinge'])
    plt.title('ALL measures going UP')
    plt.ylabel('numeric values for different performances')
    plt.xlabel('epoch')
    plt.legend(['accuracy', 'mse', 'msle', 'hinge', 'sparse_top_k_categorical_accuracy',
                'kullback_leibler_divergence', 'root_mean_squared_error', 'squared_hinge'], loc='upper left')
    plt.savefig('up.png', dpi = 600)
    plt.show()
    # all lines going down in one image
    plt.plot(history.history['loss'])
    plt.plot(history.history['cosine_similarity'])
    plt.title('ALL measures going DOWN')
    plt.ylabel('numeric values for different performances')
    plt.xlabel('epoch')
    plt.legend(['loss', 'cosine_similarity'], loc='upper left')
    plt.savefig('down.png', dpi = 600)
    plt.show()
    # all lines going flat
    plt.plot(history.history['categorical_hinge'])
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['logcosh'])
    plt.plot(history.history['mape'])
    plt.plot(history.history['mae'])
    plt.plot(history.history['binary_accuracy'])
    plt.plot(history.history['top_k_categorical_accuracy'])
    plt.title('ALL measures going FLAT (no trend)')
    plt.ylabel('numeric values for different performances')
    plt.xlabel('epoch')
    plt.legend(['categorical_hinge', 'categorical_accuracy', 'logcosh', 'mape', 'mae', 'binary_accuracy', 'top_k_categorical_accuracy'], loc='upper left')
    plt.savefig('flat.png', dpi = 600)
    plt.show()
    # all lines in one image
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['mse'])
    plt.plot(history.history['msle'])
    plt.plot(history.history['hinge'])
    plt.plot(history.history['sparse_top_k_categorical_accuracy'])
#    plt.plot(history.history['kullback_leibler_divergence'])
    plt.plot(history.history['root_mean_squared_error'])
    plt.plot(history.history['squared_hinge'])
    #
    plt.plot(history.history['categorical_hinge'])
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['logcosh'])
    plt.plot(history.history['mape'])
    plt.plot(history.history['mae'])
    plt.plot(history.history['binary_accuracy'])
    plt.plot(history.history['top_k_categorical_accuracy'])
    #
    plt.plot(history.history['loss'])
    plt.plot(history.history['cosine_similarity'])
    plt.title('ALL measures in one image')
    plt.ylabel('numeric values for different performances')
    plt.xlabel('epoch')
    plt.legend(['accuracy', 'mse', 'msle', 'hinge', 'sparse_top_k_categorical_accuracy','kullback_leibler_divergence', 'root_mean_squared_error', 'squared_hinge',
                'categorical_hinge', 'categorical_accuracy', 'logcosh', 'mape', 'mae', 'binary_accuracy', 'top_k_categorical_accuracy',
                'loss', 'cosine_similarity'], loc='upper left')
    plt.savefig('allinoneimage.png', dpi = 900)
    plt.show()


if __name__== "__main__":

    # get the start time
    st = time.time()
    # for CPU time
    # get the start time
    stp = time.process_time()
    train()
    # get the end time
    et = time.time()
    # get the execution time
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')
    # get execution time in minutes
    res = et - st
    final_res = res / 60
    print('Execution time:', final_res, 'minutes')
    # different format of showing time
    elapsed_time = time.time() - st
    print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    # for CPU time
    # get the end time
    etp = time.process_time()
    # get execution time
    res = etp - stp
    print('CPU Execution time:', res, 'seconds')
"""
1. select initial song in .krn and preprocessed (same song)
2. make seed with preprocessed style
3. see ML output in a preprocessed style
    3.1 now I have 2 strings: original and generated by ML, both in preprocessed style
4. Make comparison of both strings. EVALUATION.

5. Make it for same files used in train ML
6. Make with different files from other dataset

7. My own seed can be evaluated?
"""