# https://github.com/thamsuppp/MusicGenDL
# https://github.com/topics/music-generation?l=python
# https://github.com/asigalov61/Meddleying-MAESTRO
# https://github.com/thamsuppp/MusicGenDL
# https://github.com/aamini/introtodeeplearning/
# 

"""
Convert time series representation to midi.
Analyze in music notation software (MuseScore).
TF, Keras
Midi -> 21 to 108
Video 8 and 9 watch again to consoidate learnings
"""

# Video 3/9 starts code. Preprocessing folk song dataset.
from msilib.schema import Environment
import music21
us = Environment.UserSettings()
for key in sorted(us.keys()):
    key