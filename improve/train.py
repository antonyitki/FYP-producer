#Building and training LSTM Network with dataset from preprocessing,
#to generate melodies.
#Time series music. Generating/Generative music

import tensorflow.keras as keras
from preprocess import generate_training_sequences, SEQUENCE_LENGTH
#38 because length of vocabulary in this case (erk data file)
OUTPUT_UNITS = 18
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
SAVE_MODEL_PATH = "model.h5"


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
    model.compile(loss = loss, optimizer = keras.optimizers.Adam(lr = learning_rate), metrics = ["accuracy","mse","mape"])
    
    #print all the layers of the model(visual feedback)
    model.summary()

    return model


#to train the Network(high level function). Arguments have default values.
def train(output_units = OUTPUT_UNITS, num_units = NUM_UNITS, loss = LOSS,learning_rate= LEARNING_RATE):

    #generate the training sequences
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)
    #build the network/graph (build the model)
    model = build_model(output_units, num_units, loss, learning_rate)
    #train the model
    model.fit(inputs, targets, epochs = EPOCHS, batch_size = BATCH_SIZE)
    #save the model(reuse, call back)
    model.save(SAVE_MODEL_PATH)
    

if __name__== "__main__":
    train()