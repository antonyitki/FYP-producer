# -*- coding: utf-8 -*-
"""
Spyder Editor

https://www.kdnuggets.com/2020/02/audio-data-analysis-deep-learning-python-part-1.html
https://publish.illinois.edu/augmentedlistening/tutorials/music-processing/tutorial-1-introduction-to-audio-processing-in-python/
https://www.topcoder.com/thrive/articles/audio-data-analysis-using-python
https://blog.neurotech.africa/audio-analysis-with-librosa/
https://blog.neurotech.africa/audio-analysis-with-librosa/
https://www.kaggle.com/code/hamditarek/audio-data-analysis-using-librosa/notebook
https://www.analyticsvidhya.com/blog/2021/06/visualizing-sounds-librosa/
"""

import librosa
audio_data = "C:\\Users\\Admin\\OneDrive\\Desktop\\Music\\file_example_WAV_5MG.wav"
x , sr = librosa.load(audio_data)
print(type(x), type(sr))
print(len(x))
print(x.shape)

import matplotlib.pyplot as plt
import librosa.display
plt.figure(figsize=(14, 5))
librosa.display.waveshow(x, sr=sr)

X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()


import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write
from IPython.display import Audio
from numpy.fft import fft, ifft
Fs, data = read("C:\\Users\\Admin\\OneDrive\\Desktop\\Music\\file_example_WAV_5MG.wav")
data = data[:, 0]
print("sampling Frequency is", Fs)

Audio(data, rate=Fs)

plt.figure()
plt.plot(data)
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.title("Waveform of the Test Audio")
plt.show()


import librosa
audio = "C:\\Users\\Admin\\OneDrive\\Desktop\\Music\\file_example_WAV_5MG.wav"
x, sr = librosa.load(audio)
X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize = (10, 5))
librosa.display.specshow(Xdb, sr = sr, x_axis = 'time', y_axis = 'hz')
plt.colorbar()


import numpy as np
import matplotlib.pyplot as plt
import librosa.display
data, sampling_rate = librosa.load("C:\\Users\\Admin\\OneDrive\\Desktop\\Music\\file_example_WAV_5MG.wav")
plt.figure(figsize=(12, 4))
librosa.display.waveshow(data, sr=sampling_rate)
plt.show()

n_fft = 2048
plt.figure(figsize=(12, 4))
ft = np.abs(librosa.stft(data[:n_fft], hop_length = n_fft+1))
plt.plot(ft);
plt.title('Spectrum');
plt.xlabel('Frequency Bin');
plt.ylabel('Amplitude');

n_fft = 2048
S = librosa.amplitude_to_db(abs(ft))
plt.figure(figsize=(12, 4))
librosa.display.specshow(S, sr=sampling_rate, hop_length=512, x_axis='time', y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.show()
plt.figure(figsize=(12, 4))
mfccs = librosa.feature.mfcc(data, sr=sampling_rate, n_mfcc=13) #computed MFCCs over frames.
librosa.display.specshow(mfccs, sr=sampling_rate, x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.show()


import matplotlib.pyplot as plt
import librosa.display
plt.figure(figsize=(14, 5))
librosa.display.waveshow(x, sr=sr)
X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()
#Mel-Frequency Cepstral Coefficients(MFCCs)
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
plt.colorbar()
fs=10
mfccs = librosa.feature.mfcc(x, sr=fs)
print(mfccs.shape)
(20, 97)
#Displaying  the MFCCs:
plt.figure(figsize=(15, 7))
librosa.display.specshow(mfccs, sr=sr, x_axis='time')
#Chroma feature
hop_length=12
chromagram = librosa.feature.chroma_stft(x, sr=sr, hop_length=hop_length)
plt.figure(figsize=(15, 5))
librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm')



import librosa
import numpy as np
y, sr = librosa.load("C:\\Users\\Admin\\OneDrive\\Desktop\\Music\\file_example_WAV_5MG.wav")
D = librosa.stft(y)
s = np.abs(librosa.stft(y)**2) # Get magnitude of stft
chroma = librosa.feature.chroma_stft(S=s, sr=sr)
print(chroma)
chroma = np.cumsum(chroma)
import matplotlib.pylab as plt
x = np.linspace(-chroma, chroma)
plt.plot(x, np.sin(x))
plt.xlabel('Angle [rad]')
plt.ylabel('sin(x)')
plt.axis('tight')
plt.show()
chroma_orig = librosa.feature.chroma_cqt(y=y, sr=sr)
# For display purposes, let's zoom in on a 15-second chunk from the middle of the song
idx = tuple([slice(None), slice(*list(librosa.time_to_frames([45, 60])))])
# And for comparison, we'll show the CQT matrix as well.
C = np.abs(librosa.cqt(y=y, sr=sr, bins_per_octave=12*3, n_bins=7*12*3))
fig, ax = plt.subplots(nrows=2, sharex=True)
img1 = librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max)[idx],
                                y_axis='cqt_note', x_axis='time', bins_per_octave=12*3,
                                ax=ax[0])
fig.colorbar(img1, ax=[ax[0]], format="%+2.f dB")
ax[0].label_outer()
img2 = librosa.display.specshow(chroma_orig[idx], y_axis='chroma', x_axis='time', ax=ax[1])
fig.colorbar(img2, ax=[ax[1]])
ax[1].set(ylabel='Default chroma')


x, sampling_rate = librosa.load("C:\\Users\\Admin\\OneDrive\\Desktop\\Music\\file_example_WAV_5MG.wav")
print('Sampling Rate: ', sampling_rate)
plt.figure(figsize=(14, 5))
plt.plot(x[:sampling_rate * 5])
plt.title('Plot for the first 5 seconds')
plt.xlabel('Frame number')
plt.ylabel('Magnitude')
plt.show()














