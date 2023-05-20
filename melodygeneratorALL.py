#Generating melodies with RNN-LSTM network series
#generating the melodies and sampling the output of the NN
#use train model to generate music (generative model class)


import json
import tensorflow.keras as keras
from preprocess import SEQUENCE_LENGTH, MAPPING_PATH
import numpy as np
import music21 as m21


class MelodyGenerator:
    def __init__(self, model_path = "code\\model.h5"):
        self.model_path = model_path
        self.model = keras.models.load_model(model_path)
        with open(MAPPING_PATH, "r") as fp:
            self._mappings = json.load(fp)
        self._start_symbols = ["/"] * SEQUENCE_LENGTH


    #main method of the class(build/generates melodyusing a DL model and stores in a time series representation)
    def generate_melody(self, seed, num_steps, max_sequence_length, temperature):
        # "64 _  63 _  _" (melody encoded in music representation based on time series)
        #create seed with start symbol (str to list)
        seed = seed.split()
        melody = seed
        seed = self._start_symbols + seed
        #map seed to int (convert)
        seed = [self._mappings[symbol] for symbol in seed]
        for _ in range(num_steps):
            #limit the seed to the max_sequence_length
            seed = seed[-max_sequence_length:]
            #one-hot encode the seed
            onehot_seed = keras.utils.to_categorical(seed, num_classes=len(self._mappings))
            #(1, max_sequence_length, number of symbols in the vocabulary)
            onehot_seed = onehot_seed[np.newaxis, ...]
            #make a prediction
            probabilities = self.model.predict(onehot_seed)[0]
            #[0.1, 0.2, 0.1, 0.6] -> 1
            output_int = self._sample_with_temperature(probabilities, temperature)   
            #update the seed
            seed.append(output_int)
            #map int to our encoding(mapping.json)
            output_symbol = [k for k, v in self._mappings.items() if v == output_int][0]
            #check wether we are at the end of a melody (/ symbol)
            if output_symbol == "/":
                break
            #update the melody
            melody.append(output_symbol)
        return melody


    def _sample_with_temperature(self, probabilities, temperature):
        #temperature -> infinity
        #temperature -> 0
        #(base case) temperature = 1
        predictions = np.log(probabilities) / temperature
        probabilities = np.exp(predictions) / np.sum(np.exp(predictions))
        #list. [0,1,2,3]
        choices = range(len(probabilities))
        index = np.random.choice(choices, p = probabilities)
        return index


    #convert melody (time series representation) to midi file
    def save_melody(self, melody, step_duration = 0.25, format = "midi", file_name = "mel.mid"):
        #create a m21 stream
        stream = m21.stream.Stream()
        #parse all the symbols in the melody and create note/rest objects
        #60 _ _ _ r _ 62 _
        start_symbol = None
        step_counter = 1
        for i, symbol in enumerate(melody):
            #handle case in wich we have a note/rest or the end of a melody list
            if symbol != "_" or i+1 == len(melody):
                #ensure we are dealing with note/rest beyond the first one
                if start_symbol is not None:
                    #0.25 * 4 = 1
                    quarter_length_duration = step_duration * step_counter
                    #1. handle rest
                    if start_symbol == "r":
                        m21_event = m21.note.Rest(quarterLength = quarter_length_duration)
                    #2. handle note
                    else:
                        m21_event = m21.note.Note(int(start_symbol), quarterLength = quarter_length_duration)
                    stream.append(m21_event)
                    #reset the step counter
                    step_counter = 1
                #update the start symbol
                start_symbol = symbol
            #handle case in wich we have a prolongation sign ("_")
            else:
                step_counter += 1
        #write the m21 stream to a midi file
        stream.write(format, file_name)


if __name__== "__main__":
    mg = MelodyGenerator()
    seed = "55 _ _ _ 60 _ _ _ 55 _"
    seed2 = "67 _ _ _ _ _ 65 _ 64 _ 62 _ 60 _ _ _"
    seed3 = "55 _ _ _ 65 _ 62 _ 60 _ _ _ _"
    seedERK = "60 _ _ 62 60 _ 60"
    seedCHINA = "72 _ _ _ 74 _ 76"
    seedMIX = "60 _ _ _ 60 _ 60 _ 64 _"
    melody = mg.generate_melody(seedMIX, 700, SEQUENCE_LENGTH, 0.19)
    #melody now is in a list format (melody sequence in time series music representation)
    print(melody)
    mg.save_melody(melody)
#non deterministec algorithm because each time output is different.