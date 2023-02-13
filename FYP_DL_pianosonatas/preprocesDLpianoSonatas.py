import time
import json
import os
import music21 as m21
import tensorflow.keras as keras
import numpy as np


KERN_DATASET_PATH = "RNN-LSTMpianoSonatas_DL\\pianosonata"
SAVE_DIR = "RNN-LSTMpianoSonatas_DL\\dataset"
#it will be saved in the current directory that is open (working on)
SINGLE_FILE_DATASET = "RNN-LSTMpianoSonatas_DL\\file_dataset"
MAPPING_PATH = "RNN-LSTMpianoSonatas_DL\\mapping.json"
#for training LSTM (64 items)
SEQUENCE_LENGTH = 64
ACCEPTABLE_DURATIONS = [
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2,
    3,
    4 #all note
]

def load_songs_in_kern(dataset_path):

    songs = []

    #ckeck all dataset files and load then(music21)
    for path, subdirs, files in os.walk(dataset_path):
        for file in files:
            if file[-3:] == "krn":
                #song = score/stream (music notation)
                song = m21.converter.parse(os.path.join(path, file))
                songs.append(song)
    return songs


#return bool
def has_acceptable_durations(song, aceptable_durations):
    #describe for help
    for note in song.flat.notesAndRests:
        if note.duration.quarterLenght not in ACCEPTABLE_DURATIONS:
            return False


def has_acceptable_durations(song, acceptable_durations):
    #from m21 and it flats all the objects in a single list
    #notesAndRest is a filter for objects that are not notes or rests
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in acceptable_durations:
            return False
    return True


def transpose(song):

    #get the key from the song directly
    #parts from the score (score = multiple parts). Get all the parts.
    parts = song.getElementsByClass(m21.stream.Part)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    #usually in the 4th index, is where data key object is stored
    key = measures_part0[0][4]

    #if key is not notated in the song then estimate it with m21
    if not isinstance(key, m21.key.Key):
        key = song.analyze("key")
    #see the original key for the song
    print(key)

    #get the interval for transposition (Bmaj->Cmaj)
    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode == "minor":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))

    #transpose song by calculated interval with m21
    transposed_song = song.transpose(interval)

    #NOT all keys(24 in total), because of huge dataset result. 
    return transposed_song


#song to encoded music time series representation
def encode_song(song, time_step = 0.25):
    #p = 60 and d = 1.0 -> [60, "_", "_", "_"]
    #list that holds all notes and rest in the time series
    encoded_song = []
    #single list flatten all objects
    for event in song.flat.notesAndRests:
        # focus to handle single notes and the rests
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi
        elif isinstance(event, m21.note.Rest):
            symbol = "r"
        #convert notes and rest into time series notation
        steps = int(event.duration.quarterLength / time_step)
        for step in range(steps):
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")
    #translate the encoded song to a string (cast)
    encoded_song = " ".join(map(str, encoded_song))

    return encoded_song



def preprocess(dataset_path):

    #load songs (music21)
    print("Loading songs...")
    songs = load_songs_in_kern(dataset_path)
    print(f"Loaded{len(songs)} songs.")

    #analize each song individually
    for i, song in enumerate(songs):
            
        #filter out songs without good duration
        if not has_acceptable_durations(song, ACCEPTABLE_DURATIONS):
            #ignore the song (not preprocessed) if not acceptable duration
            continue

        #transpose songs to Music Motation (Cmajor/Aminor)
        song = transpose(song)


        #encode songs with music time series representation (video #2 for details)
        encoded_song = encode_song(song)

        #save songs to .txt. each path is unique to each song
        save_path = os.path.join(SAVE_DIR, str(i))
        with open(save_path, "w") as fp:
            fp.write(encoded_song)


def load(file_path):
    with open(file_path, "r") as fp:
        song = fp.read()
    return song


def create_single_file_dataset(dataset_path, file_dataset_path, sequence_length):
    #symbols of a song
    new_song_delimiter = "/ " * sequence_length
    songs = ""
    #load encoded songs and add delimiters
    for path, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path, file)
            #song is a str of symbols
            song = load(file_path)
            songs = songs + song + " " + new_song_delimiter
    #to remove space from delimeter ("/ ")
    songs = songs[:-1]    

    #save all the string that contains all the dataset
    with open(file_dataset_path, "w") as fp:
        fp.write(songs)
    return songs


def create_mapping(songs, mappin_path):
    mappings = {}
    #identify the vocabulary (all the symbols in the dataset)
    songs = songs.split()
    #casting
    vocabulary = list(set(songs))

    # create mappings
    for i, symbol in enumerate(vocabulary):
        #creating new keys in the mapping dictionary
        mappings[symbol] = i

    #save the vocabulary to .json file 
    with open(mappin_path, "w") as fp:
        json.dump(mappings, fp, indent=4)


#last function of conversion program
def convert_songs_to_int(songs):
    int_songs = []
    #load mapping
    with open(MAPPING_PATH, "r") as fp:
        mappings = json.load(fp)
    #cast songs str to a list
    songs = songs.split()
    #map songs to int
    for symbol in songs:
        int_songs.append(mappings[symbol])
    return int_songs


#next note prediction(time series prediction)
#LAST STAGE OF PREPROCESSING!
def generate_training_sequences(sequence_length):
    #supervised task (inputs and targets)
    #[11, 12, 13, 14, ...] -> i(inputs): [11, 12], t(targets): 13; i: [12, 13] , t: 14; ...(until end data time series)

    #load songs and convert them to int (map to int)
    songs = load(SINGLE_FILE_DATASET)
    int_songs = convert_songs_to_int(songs)

    #generate the training sequences
    #100 samples in dataset, sequence_length = 64; 100-64 = 36 (generate 36 sequences of 64 times/items)
    inputs = []
    targets = []
    num_sequences = len(int_songs) - sequence_length
    for i in range(num_sequences):
        #inputs and targets
        inputs.append(int_songs[i:i+sequence_length])
        targets.append(int_songs[i+sequence_length])

    #one-hot encode the sequences
    #shape inputs = 2D-list(# of sequences, sequence length(64 in our case items), vocabulary size)
    #[[0,1,2], [1,1,2]] -> [[[1,0,0],[0,1,0],[0,0,1]], []]
    vocabulary_size = len(set(int_songs))
    #3D array (inputs)
    inputs = keras.utils.to_categorical(inputs, num_classes = vocabulary_size)
    targets = np.array(targets)

    return inputs, targets



def main():
    preprocess(KERN_DATASET_PATH)
    songs = create_single_file_dataset(SAVE_DIR, SINGLE_FILE_DATASET, SEQUENCE_LENGTH)
    create_mapping(songs, MAPPING_PATH)
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)
    

if __name__== "__main__":
    #songs = load_songs_in_kern(KERN_DATASET_PATH)
    #print(f"Loaded {(len(songs))} songs.")
    #song = songs[0]
    #m21.environment (documentation)
    #m21.configure.run()
    #current song to analyse
    #print(f"Has accetable durtion? {has_acceptable_durations(song, ACCEPTABLE_DURATIONS)}")
    #preprocess(KERN_DATASET_PATH)
    #see if transpose song works
    #transposed_song = transpose(song)
    
    
    #song.show()
    # to see changes in the song
    #transposed_song.show()
    
    #songs = create_single_file_dataset(SAVE_DIR, SINGLE_FILE_DATASET, SEQUENCE_LENGTH)




    # get the start time
    st = time.time()
    # for CPU time
    # get the start time
    stp = time.process_time()

    main()

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