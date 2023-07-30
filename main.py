import whisper_timestamped as wt
from flask import Flask, jsonify, request, make_response
from werkzeug.utils import secure_filename
from flask_cors import CORS
import os
import json
import pyphen
from nltk.corpus import cmudict


CMU_DICT = cmudict.dict()
PYPHEN_DICT = pyphen.Pyphen(lang='en')
#local testing
app = Flask(__name__)
CORS(app)

QUICK_TEST = {'1': {'text': 'My', 'start': 0.2, 'end': 0.56, 'confidence': 0.862}, '2': {'text': 'name', 'start': 0.56, 'end': 0.84, 'confidence': 0.995}, '3': {'text': 'is', 'start': 0.84, 'end': 2.68, 'confidence': 0.994}, '4': {'text': 'Ray', 'start': 2.68, 'end': 4.06, 'confidence': 0.722}, '5': {'text': 'Demnitz,', 'start': 4.06, 'end': 4.74, 'confidence': 0.265}, '6': {'text': "I'm", 'start': 6.52, 'end': 8.64, 'confidence': 0.894}, '7': {'text': '20', 'start': 8.64, 'end': 10.14, 'confidence': 0.611}, '8': {'text': 'years', 'start': 10.14, 'end': 10.4, 'confidence': 0.954}, '9': {'text': 'old,', 'start': 10.4, 'end': 11.16, 'confidence': 0.71}, '10': {'text': "I'm", 'start': 11.52, 'end': 11.76, 'confidence': 0.809}, '11': {'text': 'from', 'start': 11.76, 'end': 12.76, 'confidence': 0.985}, '12': {'text': 'New', 'start': 12.76, 'end': 13.2, 'confidence': 0.42}, '13': {'text': 'York', 'start': 13.2, 'end': 13.38, 'confidence': 0.985}, '14': {'text': 'City,', 'start': 13.38, 'end': 13.72, 'confidence': 0.546}, '15': {'text': 'and', 'start': 13.78, 'end': 13.96, 'confidence': 0.97}, '16': {'text': 'I', 'start': 13.96, 'end': 14.1, 'confidence': 0.882}, '17': {'text': 'stutter.', 'start': 14.1, 'end': 14.3, 'confidence': 0.858}, '18': {'text': 'The', 'start': 15.62, 'end': 16.02, 'confidence': 0.803}, '19': {'text': 'hardest', 'start': 16.02, 'end': 16.56, 'confidence': 0.996}, '20': {'text': 'part', 'start': 16.56, 'end': 16.82, 'confidence': 0.988}, '21': {'text': 'of', 'start': 16.82, 'end': 16.98, 'confidence': 0.964}, '22': {'text': 'stuttering', 'start': 16.98, 'end': 17.32, 'confidence': 0.859}, '23': {'text': 'is', 'start': 17.32, 'end': 17.48, 'confidence': 0.977}, '24': {'text': 'not', 'start': 17.48, 'end': 17.7, 'confidence': 0.978}, '25': {'text': 'the', 'start': 17.7, 'end': 17.9, 'confidence': 0.974}, '26': {'text': 'physical', 'start': 17.9, 'end': 18.22, 'confidence': 0.993}, '27': {'text': 'stutter.', 'start': 18.22, 'end': 18.76, 'confidence': 0.791}, '28': {'text': "It's", 'start': 18.82, 'end': 19.28, 'confidence': 0.862}, '29': {'text': 'the', 'start': 19.28, 'end': 19.84, 'confidence': 0.995}, '30': {'text': 'mental', 'start': 19.84, 'end': 20.2, 'confidence': 0.982}, '31': {'text': 'and', 'start': 20.2, 'end': 20.36, 'confidence': 0.95}, '32': {'text': 'emotional', 'start': 20.36, 'end': 20.74, 'confidence': 0.999}, '33': {'text': 'baggage', 'start': 20.74, 'end': 21.24, 'confidence': 0.959}, '34': {'text': 'that', 'start': 21.24, 'end': 21.4, 'confidence': 0.808}, '35': {'text': 'comes', 'start': 21.4, 'end': 21.68, 'confidence': 0.954}, '36': {'text': 'along', 'start': 21.68, 'end': 22.04, 'confidence': 0.991}, '37': {'text': 'with', 'start': 22.04, 'end': 22.24, 'confidence': 0.981}, '38': {'text': 'it.', 'start': 22.24, 'end': 22.74, 'confidence': 0.954}, '39': {'text': "People's", 'start': 22.92, 'end': 23.54, 'confidence': 0.92}, '40': {'text': 'first', 'start': 23.54, 'end': 23.8, 'confidence': 0.99}, '41': {'text': 'impression', 'start': 23.8, 'end': 24.16, 'confidence': 0.996}, '42': {'text': 'of', 'start': 24.16, 'end': 24.3, 'confidence': 0.848}, '43': {'text': 'me', 'start': 24.3, 'end': 24.54, 'confidence': 0.998}, '44': {'text': 'feels', 'start': 24.54, 'end': 24.92, 'confidence': 0.952}, '45': {'text': 'like', 'start': 24.92, 'end': 25.26, 'confidence': 0.988}, '46': {'text': "it's", 'start': 25.26, 'end': 25.54, 'confidence': 0.978}, '47': {'text': 'not', 'start': 25.54, 'end': 25.86, 'confidence': 0.997}, '48': {'text': 'really', 'start': 25.86, 'end': 26.14, 'confidence': 0.994}, '49': {'text': 'neat.', 'start': 26.14, 'end': 26.4, 'confidence': 0.368}, '50': {'text': 'When', 'start': 26.46, 'end': 26.66, 'confidence': 0.215}, '51': {'text': 'I', 'start': 26.66, 'end': 26.8, 'confidence': 0.974}, '52': {'text': 'open', 'start': 26.8, 'end': 27.0, 'confidence': 0.635}, '53': {'text': 'my', 'start': 27.0, 'end': 27.16, 'confidence': 0.994}, '54': {'text': 'mouth', 'start': 27.16, 'end': 27.42, 'confidence': 0.996}, '55': {'text': 'and', 'start': 27.42, 'end': 27.6, 'confidence': 0.763}, '56': {'text': 'I', 'start': 27.6, 'end': 27.72, 'confidence': 0.676}, '57': {'text': 'look', 'start': 27.72, 'end': 27.88, 'confidence': 0.982}, '58': {'text': 'like', 'start': 27.88, 'end': 28.04, 'confidence': 0.98}, '59': {'text': "I'm", 'start': 28.04, 'end': 28.24, 'confidence': 0.981}, '60': {'text': 'in', 'start': 28.24, 'end': 28.34, 'confidence': 0.999}, '61': {'text': 'pain,', 'start': 28.34, 'end': 28.76, 'confidence': 0.468}, '62': {'text': 'I', 'start': 28.94, 'end': 29.14, 'confidence': 0.107}, '63': {'text': 'get', 'start': 29.14, 'end': 29.16, 'confidence': 0.073}, '64': {'text': 'a', 'start': 29.16, 'end': 29.3, 'confidence': 0.1}, '65': {'text': 'good', 'start': 29.3, 'end': 29.4, 'confidence': 0.85}, '66': {'text': 'choose', 'start': 29.4, 'end': 29.56, 'confidence': 0.332}, '67': {'text': 'of', 'start': 29.56, 'end': 29.66, 'confidence': 0.687}, '68': {'text': 'mayonnaise.', 'start': 29.66, 'end': 30.16, 'confidence': 0.265}, '69': {'text': 'I', 'start': 30.18, 'end': 30.58, 'confidence': 0.602}, '70': {'text': 'like', 'start': 30.58, 'end': 30.84, 'confidence': 0.948}, '71': {'text': 'communicating', 'start': 30.84, 'end': 31.22, 'confidence': 0.267}, '72': {'text': 'with', 'start': 31.22, 'end': 31.44, 'confidence': 0.984}, '73': {'text': 'people.', 'start': 31.44, 'end': 31.84, 'confidence': 0.795}, '74': {'text': 'I', 'start': 31.9, 'end': 32.04, 'confidence': 0.865}, '75': {'text': 'like', 'start': 32.04, 'end': 32.46, 'confidence': 0.945}, '76': {'text': 'expressing', 'start': 32.46, 'end': 32.9, 'confidence': 0.992}, '77': {'text': 'myself.', 'start': 32.9, 'end': 33.52, 'confidence': 0.431}, '78': {'text': 'And', 'start': 33.72, 'end': 34.48, 'confidence': 0.41}, '79': {'text': 'I', 'start': 34.48, 'end': 34.84, 'confidence': 0.942}, '80': {'text': 'feel', 'start': 34.84, 'end': 35.14, 'confidence': 0.991}, '81': {'text': 'like', 'start': 35.14, 'end': 35.34, 'confidence': 0.988}, '82': {'text': 'I', 'start': 35.34, 'end': 35.5, 'confidence': 0.863}, '83': {'text': 'like', 'start': 35.5, 'end': 35.76, 'confidence': 0.945}, '84': {'text': 'helping', 'start': 35.76, 'end': 36.08, 'confidence': 0.992}, '85': {'text': 'people', 'start': 36.08, 'end': 36.64, 'confidence': 0.998}, '86': {'text': 'in', 'start': 36.64, 'end': 37.54, 'confidence': 0.707}, '87': {'text': 'many', 'start': 37.54, 'end': 37.98, 'confidence': 0.978}, '88': {'text': 'ways,', 'start': 37.98, 'end': 38.32, 'confidence': 0.451}, '89': {'text': 'beak,', 'start': 38.88, 'end': 39.48, 'confidence': 0.341}, '90': {'text': 'beak,', 'start': 39.48, 'end': 39.88, 'confidence': 0.881}, '91': {'text': 'beak.', 'start': 39.9, 'end': 40.22, 'confidence': 0.679}, '92': {'text': 'My', 'start': 40.32, 'end': 40.6, 'confidence': 0.839}, '93': {'text': 'name', 'start': 40.6, 'end': 40.88, 'confidence': 0.995}, '94': {'text': 'is', 'start': 40.88, 'end': 42.44, 'confidence': 0.991}, '95': {'text': 'Ray', 'start': 42.44, 'end': 44.1, 'confidence': 0.71}, '96': {'text': 'Demnitz,', 'start': 44.1, 'end': 44.76, 'confidence': 0.291}, '97': {'text': "I'm", 'start': 46.94, 'end': 48.66, 'confidence': 0.883}, '98': {'text': '20', 'start': 48.66, 'end': 50.18, 'confidence': 0.589}, '99': {'text': 'years', 'start': 50.18, 'end': 50.44, 'confidence': 0.955}, '100': {'text': 'old,', 'start': 50.44, 'end': 51.02, 'confidence': 0.685}, '101': {'text': "I'm", 'start': 51.54, 'end': 51.84, 'confidence': 0.816}, '102': {'text': 'from', 'start': 51.84, 'end': 52.84, 'confidence': 0.98}, '103': {'text': 'New', 'start': 52.84, 'end': 53.22, 'confidence': 0.492}, '104': {'text': 'York', 'start': 53.22, 'end': 53.4, 'confidence': 0.992}, '105': {'text': 'City,', 'start': 53.4, 'end': 53.76, 'confidence': 0.528}, '106': {'text': 'and', 'start': 53.8, 'end': 53.96, 'confidence': 0.975}, '107': {'text': 'I', 'start': 53.96, 'end': 54.16, 'confidence': 0.845}, '108': {'text': 'stutter.', 'start': 54.16, 'end': 54.32, 'confidence': 0.84}, '109': {'text': 'The', 'start': 55.72, 'end': 56.08, 'confidence': 0.878}, '110': {'text': 'hardest', 'start': 56.08, 'end': 56.58, 'confidence': 0.998}, '111': {'text': 'part', 'start': 56.58, 'end': 56.84, 'confidence': 0.992}, '112': {'text': 'of', 'start': 56.84, 'end': 57.02, 'confidence': 0.976}, '113': {'text': 'stuttering', 'start': 57.02, 'end': 57.34, 'confidence': 0.887}, '114': {'text': 'is', 'start': 57.34, 'end': 57.52, 'confidence': 0.981}, '115': {'text': 'not', 'start': 57.52, 'end': 57.72, 'confidence': 0.995}, '116': {'text': 'the', 'start': 57.72, 'end': 57.88, 'confidence': 0.989}, '117': {'text': 'physical', 'start': 57.88, 'end': 58.26, 'confidence': 0.996}, '118': {'text': 'stutter.', 'start': 58.26, 'end': 58.8, 'confidence': 0.819}, '119': {'text': "It's", 'start': 58.86, 'end': 59.24, 'confidence': 0.659}, '120': {'text': 'not', 'start': 59.24, 'end': 59.48, 'confidence': 0.07}, '121': {'text': 'the', 'start': 59.48, 'end': 59.5, 'confidence': 0.02}, '122': {'text': 'physical', 'start': 59.5, 'end': 59.52, 'confidence': 0.0}, '123': {'text': 'stutter.', 'start': 59.52, 'end': 59.54, 'confidence': 0.006}}
QUICK_TEST_WITH_SYLLABLES = {'1': {'text': 'My', 'start': 0.2, 'end': 0.56, 'confidence': 0.862, 'syllable_count': 2}, '2': {'text': 'name', 'start': 0.56, 'end': 0.84, 'confidence': 0.995, 'syllable_count': 1}, '3': {'text': 'is', 'start': 0.84, 'end': 2.68, 'confidence': 0.994, 'syllable_count': 1}, '4': {'text': 'Ray', 'start': 2.68, 'end': 4.06, 'confidence': 0.722, 'syllable_count': 2}, '5': {'text': 'Demnitz,', 'start': 4.06, 'end': 4.74, 'confidence': 0.265, 'syllable_count': 3}, '6': {'text': "I'm", 'start': 6.52, 'end': 8.64, 'confidence': 0.894, 'syllable_count': 3}, '7': {'text': '20', 'start': 8.64, 'end': 10.14, 'confidence': 0.611, 'syllable_count': 2}, '8': {'text': 'years', 'start': 10.14, 'end': 10.4, 'confidence': 0.954, 'syllable_count': 2}, '9': {'text': 'old,', 'start': 10.4, 'end': 11.16, 'confidence': 0.71, 'syllable_count': 3}, '10': {'text': "I'm", 'start': 11.52, 'end': 11.76, 'confidence': 0.809, 'syllable_count': 1}, '11': {'text': 'from', 'start': 11.76, 'end': 12.76, 'confidence': 0.985, 'syllable_count': 3}, '12': {'text': 'New', 'start': 12.76, 'end': 13.2, 'confidence': 0.42, 'syllable_count': 3}, '13': {'text': 'York', 'start': 13.2, 'end': 13.38, 'confidence': 0.985, 'syllable_count': 2}, '14': {'text': 'City,', 'start': 13.38, 'end': 13.72, 'confidence': 0.546, 'syllable_count': 1}, '15': {'text': 'and', 'start': 13.78, 'end': 13.96, 'confidence': 0.97, 'syllable_count': 3}, '16': {'text': 'I', 'start': 13.96, 'end': 14.1, 'confidence': 0.882, 'syllable_count': 2}, '17': {'text': 'stutter.', 'start': 14.1, 'end': 14.3, 'confidence': 0.858, 'syllable_count': 1}, '18': {'text': 'The', 'start': 15.62, 'end': 16.02, 'confidence': 0.803, 'syllable_count': 1}, '19': {'text': 'hardest', 'start': 16.02, 'end': 16.56, 'confidence': 0.996, 'syllable_count': 1}, '20': {'text': 'part', 'start': 16.56, 'end': 16.82, 'confidence': 0.988, 'syllable_count': 2}, '21': {'text': 'of', 'start': 16.82, 'end': 16.98, 'confidence': 0.964, 'syllable_count': 3}, '22': {'text': 'stuttering', 'start': 16.98, 'end': 17.32, 'confidence': 0.859, 'syllable_count': 3}, '23': {'text': 'is', 'start': 17.32, 'end': 17.48, 'confidence': 0.977, 'syllable_count': 2}, '24': {'text': 'not', 'start': 17.48, 'end': 17.7, 'confidence': 0.978, 'syllable_count': 3}, '25': {'text': 'the', 'start': 17.7, 'end': 17.9, 'confidence': 0.974, 'syllable_count': 1}, '26': {'text': 'physical', 'start': 17.9, 'end': 18.22, 'confidence': 0.993, 'syllable_count': 3}, '27': {'text': 'stutter.', 'start': 18.22, 'end': 18.76, 'confidence': 0.791, 'syllable_count': 3}, '28': {'text': "It's", 'start': 18.82, 'end': 19.28, 'confidence': 0.862, 'syllable_count': 1}, '29': {'text': 'the', 'start': 19.28, 'end': 19.84, 'confidence': 0.995, 'syllable_count': 3}, '30': {'text': 'mental', 'start': 19.84, 'end': 20.2, 'confidence': 0.982, 'syllable_count': 3}, '31': {'text': 'and', 'start': 20.2, 'end': 20.36, 'confidence': 0.95, 'syllable_count': 2}, '32': {'text': 'emotional', 'start': 20.36, 'end': 20.74, 'confidence': 0.999, 'syllable_count': 2}, '33': {'text': 'baggage', 'start': 20.74, 'end': 21.24, 'confidence': 0.959, 'syllable_count': 1}, '34': {'text': 'that', 'start': 21.24, 'end': 21.4, 'confidence': 0.808, 'syllable_count': 3}, '35': {'text': 'comes', 'start': 21.4, 'end': 21.68, 'confidence': 0.954, 'syllable_count': 1}, '36': {'text': 'along', 'start': 21.68, 'end': 22.04, 'confidence': 0.991, 'syllable_count': 2}, '37': {'text': 'with', 'start': 22.04, 'end': 22.24, 'confidence': 0.981, 'syllable_count': 2}, '38': {'text': 'it.', 'start': 22.24, 'end': 22.74, 'confidence': 0.954, 'syllable_count': 1}, '39': {'text': "People's", 'start': 22.92, 'end': 23.54, 'confidence': 0.92, 'syllable_count': 3}, '40': {'text': 'first', 'start': 23.54, 'end': 23.8, 'confidence': 0.99, 'syllable_count': 1}, '41': {'text': 'impression', 'start': 23.8, 'end': 24.16, 'confidence': 0.996, 'syllable_count': 3}, '42': {'text': 'of', 'start': 24.16, 'end': 24.3, 'confidence': 0.848, 'syllable_count': 2}, '43': {'text': 'me', 'start': 24.3, 'end': 24.54, 'confidence': 0.998, 'syllable_count': 2}, '44': {'text': 'feels', 'start': 24.54, 'end': 24.92, 'confidence': 0.952, 'syllable_count': 2}, '45': {'text': 'like', 'start': 24.92, 'end': 25.26, 'confidence': 0.988, 'syllable_count': 2}, '46': {'text': "it's", 'start': 25.26, 'end': 25.54, 'confidence': 0.978, 'syllable_count': 3}, '47': {'text': 'not', 'start': 25.54, 'end': 25.86, 'confidence': 0.997, 'syllable_count': 3}, '48': {'text': 'really', 'start': 25.86, 'end': 26.14, 'confidence': 0.994, 'syllable_count': 3}, '49': {'text': 'neat.', 'start': 26.14, 'end': 26.4, 'confidence': 0.368, 'syllable_count': 2}, '50': {'text': 'When', 'start': 26.46, 'end': 26.66, 'confidence': 0.215, 'syllable_count': 2}, '51': {'text': 'I', 'start': 26.66, 'end': 26.8, 'confidence': 0.974, 'syllable_count': 1}, '52': {'text': 'open', 'start': 26.8, 'end': 27.0, 'confidence': 0.635, 'syllable_count': 2}, '53': {'text': 'my', 'start': 27.0, 'end': 27.16, 'confidence': 0.994, 'syllable_count': 1}, '54': {'text': 'mouth', 'start': 27.16, 'end': 27.42, 'confidence': 0.996, 'syllable_count': 3}, '55': {'text': 'and', 'start': 27.42, 'end': 27.6, 'confidence': 0.763, 'syllable_count': 3}, '56': {'text': 'I', 'start': 27.6, 'end': 27.72, 'confidence': 0.676, 'syllable_count': 3}, '57': {'text': 'look', 'start': 27.72, 'end': 27.88, 'confidence': 0.982, 'syllable_count': 3}, '58': {'text': 'like', 'start': 27.88, 'end': 28.04, 'confidence': 0.98, 'syllable_count': 1}, '59': {'text': "I'm", 'start': 28.04, 'end': 28.24, 'confidence': 0.981, 'syllable_count': 3}, '60': {'text': 'in', 'start': 28.24, 'end': 28.34, 'confidence': 0.999, 'syllable_count': 1}, '61': {'text': 'pain,', 'start': 28.34, 'end': 28.76, 'confidence': 0.468, 'syllable_count': 1}, '62': {'text': 'I', 'start': 28.94, 'end': 29.14, 'confidence': 0.107, 'syllable_count': 1}, '63': {'text': 'get', 'start': 29.14, 'end': 29.16, 'confidence': 0.073, 'syllable_count': 3}, '64': {'text': 'a', 'start': 29.16, 'end': 29.3, 'confidence': 0.1, 'syllable_count': 3}, '65': {'text': 'good', 'start': 29.3, 'end': 29.4, 'confidence': 0.85, 'syllable_count': 3}, '66': {'text': 'choose', 'start': 29.4, 'end': 29.56, 'confidence': 0.332, 'syllable_count': 1}, '67': {'text': 'of', 'start': 29.56, 'end': 29.66, 'confidence': 0.687, 'syllable_count': 3}, '68': {'text': 'mayonnaise.', 'start': 29.66, 'end': 30.16, 'confidence': 0.265, 'syllable_count': 1}, '69': {'text': 'I', 'start': 30.18, 'end': 30.58, 'confidence': 0.602, 'syllable_count': 1}, '70': {'text': 'like', 'start': 30.58, 'end': 30.84, 'confidence': 0.948, 'syllable_count': 2}, '71': {'text': 'communicating', 'start': 30.84, 'end': 31.22, 'confidence': 0.267, 'syllable_count': 1}, '72': {'text': 'with', 'start': 31.22, 'end': 31.44, 'confidence': 0.984, 'syllable_count': 1}, '73': {'text': 'people.', 'start': 31.44, 'end': 31.84, 'confidence': 0.795, 'syllable_count': 2}, '74': {'text': 'I', 'start': 31.9, 'end': 32.04, 'confidence': 0.865, 'syllable_count': 3}, '75': {'text': 'like', 'start': 32.04, 'end': 32.46, 'confidence': 0.945, 'syllable_count': 1}, '76': {'text': 'expressing', 'start': 32.46, 'end': 32.9, 'confidence': 0.992, 'syllable_count': 1}, '77': {'text': 'myself.', 'start': 32.9, 'end': 33.52, 'confidence': 0.431, 'syllable_count': 1}, '78': {'text': 'And', 'start': 33.72, 'end': 34.48, 'confidence': 0.41, 'syllable_count': 1}, '79': {'text': 'I', 'start': 34.48, 'end': 34.84, 'confidence': 0.942, 'syllable_count': 1}, '80': {'text': 'feel', 'start': 34.84, 'end': 35.14, 'confidence': 0.991, 'syllable_count': 1}, '81': {'text': 'like', 'start': 35.14, 'end': 35.34, 'confidence': 0.988, 'syllable_count': 3}, '82': {'text': 'I', 'start': 35.34, 'end': 35.5, 'confidence': 0.863, 'syllable_count': 2}, '83': {'text': 'like', 'start': 35.5, 'end': 35.76, 'confidence': 0.945, 'syllable_count': 1}, '84': {'text': 'helping', 'start': 35.76, 'end': 36.08, 'confidence': 0.992, 'syllable_count': 2}, '85': {'text': 'people', 'start': 36.08, 'end': 36.64, 'confidence': 0.998, 'syllable_count': 1}, '86': {'text': 'in', 'start': 36.64, 'end': 37.54, 'confidence': 0.707, 'syllable_count': 3}, '87': {'text': 'many', 'start': 37.54, 'end': 37.98, 'confidence': 0.978, 'syllable_count': 3}, '88': {'text': 'ways,', 'start': 37.98, 'end': 38.32, 'confidence': 0.451, 'syllable_count': 3}, '89': {'text': 'beak,', 'start': 38.88, 'end': 39.48, 'confidence': 0.341, 'syllable_count': 2}, '90': {'text': 'beak,', 'start': 39.48, 'end': 39.88, 'confidence': 0.881, 'syllable_count': 2}, '91': {'text': 'beak.', 'start': 39.9, 'end': 40.22, 'confidence': 0.679, 'syllable_count': 1}, '92': {'text': 'My', 'start': 40.32, 'end': 40.6, 'confidence': 0.839, 'syllable_count': 1}, '93': {'text': 'name', 'start': 40.6, 'end': 40.88, 'confidence': 0.995, 'syllable_count': 2}, '94': {'text': 'is', 'start': 40.88, 'end': 42.44, 'confidence': 0.991, 'syllable_count': 1}, '95': {'text': 'Ray', 'start': 42.44, 'end': 44.1, 'confidence': 0.71, 'syllable_count': 3}, '96': {'text': 'Demnitz,', 'start': 44.1, 'end': 44.76, 'confidence': 0.291, 'syllable_count': 1}, '97': {'text': "I'm", 'start': 46.94, 'end': 48.66, 'confidence': 0.883, 'syllable_count': 1}, '98': {'text': '20', 'start': 48.66, 'end': 50.18, 'confidence': 0.589, 'syllable_count': 1}, '99': {'text': 'years', 'start': 50.18, 'end': 50.44, 'confidence': 0.955, 'syllable_count': 2}, '100': {'text': 'old,', 'start': 50.44, 'end': 51.02, 'confidence': 0.685, 'syllable_count': 1}, '101': {'text': "I'm", 'start': 51.54, 'end': 51.84, 'confidence': 0.816, 'syllable_count': 1}, '102': {'text': 'from', 'start': 51.84, 'end': 52.84, 'confidence': 0.98, 'syllable_count': 2}, '103': {'text': 'New', 'start': 52.84, 'end': 53.22, 'confidence': 0.492, 'syllable_count': 2}, '104': {'text': 'York', 'start': 53.22, 'end': 53.4, 'confidence': 0.992, 'syllable_count': 1}, '105': {'text': 'City,', 'start': 53.4, 'end': 53.76, 'confidence': 0.528, 'syllable_count': 1}, '106': {'text': 'and', 'start': 53.8, 'end': 53.96, 'confidence': 0.975, 'syllable_count': 2}, '107': {'text': 'I', 'start': 53.96, 'end': 54.16, 'confidence': 0.845, 'syllable_count': 1}, '108': {'text': 'stutter.', 'start': 54.16, 'end': 54.32, 'confidence': 0.84, 'syllable_count': 3}, '109': {'text': 'The', 'start': 55.72, 'end': 56.08, 'confidence': 0.878, 'syllable_count': 2}, '110': {'text': 'hardest', 'start': 56.08, 'end': 56.58, 'confidence': 0.998, 'syllable_count': 3}, '111': {'text': 'part', 'start': 56.58, 'end': 56.84, 'confidence': 0.992, 'syllable_count': 1}, '112': {'text': 'of', 'start': 56.84, 'end': 57.02, 'confidence': 0.976, 'syllable_count': 2}, '113': {'text': 'stuttering', 'start': 57.02, 'end': 57.34, 'confidence': 0.887, 'syllable_count': 3}, '114': {'text': 'is', 'start': 57.34, 'end': 57.52, 'confidence': 0.981, 'syllable_count': 3}, '115': {'text': 'not', 'start': 57.52, 'end': 57.72, 'confidence': 0.995, 'syllable_count': 1}, '116': {'text': 'the', 'start': 57.72, 'end': 57.88, 'confidence': 0.989, 'syllable_count': 3}, '117': {'text': 'physical', 'start': 57.88, 'end': 58.26, 'confidence': 0.996, 'syllable_count': 2}, '118': {'text': 'stutter.', 'start': 58.26, 'end': 58.8, 'confidence': 0.819, 'syllable_count': 3}, '119': {'text': "It's", 'start': 58.86, 'end': 59.24, 'confidence': 0.659, 'syllable_count': 1}, '120': {'text': 'not', 'start': 59.24, 'end': 59.48, 'confidence': 0.07, 'syllable_count': 3}, '121': {'text': 'the', 'start': 59.48, 'end': 59.5, 'confidence': 0.02, 'syllable_count': 1}, '122': {'text': 'physical', 'start': 59.5, 'end': 59.52, 'confidence': 0.0, 'syllable_count': 1}, '123': {'text': 'stutter.', 'start': 59.52, 'end': 59.54, 'confidence': 0.006, 'syllable_count': 3}}
WORD_DIC = pyphen.Pyphen(lang='en')


def syllables_in_word(word):
    return len(WORD_DIC.inserted(word).split('-'))


def get_total_syllables(raw_transcription):
    words = raw_transcription.split(' ')
    return sum(syllables_in_word(word) for word in words)


def audio_transcribe(audiofile):
    audio = wt.load_audio(audiofile)
    model = wt.load_model("medium.en", device="cpu")

    return wt.transcribe(model, audio, language="en", temperature=0, beam_size=5, best_of=5)


@app.route('/', methods=['GET'])
def base():
    return 'hello world'


@app.route('/get_transcription', methods=['POST'])
def get_transcription():
    quick = False
    if not quick:
        if 'file' not in request.files:
            return "No file part", 400
        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400
        filename = secure_filename(file.filename)
        print("FILENAME: ", filename)
        file.save(os.path.join("./audio/", filename))

        wt_transcription = create_transcription_object('./audio/'+filename)

        final_trans_obj = json_obj_convert(wt_transcription)
        print(final_trans_obj)
        syllables_total = sum(word['syllable_count'] for word in final_trans_obj.values())
        print(syllables_total)

        json_obj = {"transcription_obj": final_trans_obj,
                    "syllables": syllables_total
                    }
    else:
        json_obj = {"transcription_obj": QUICK_TEST_WITH_SYLLABLES,
                "syllables": 83}

    return jsonify(json_obj)


@app.route('/manual_transcription', methods=['POST'])
def manual_transcription():
    manual_transcription_words = request.json.get("data").strip().split(" ")
    transcription_obj = {}
    for index, word in enumerate(manual_transcription_words):
        syllable_count = count_syllables(word)
        transcription_obj[index+1] = {
            'text': word,
            'start': None,
            'end': None,
            'syllable_count': syllable_count
        }

    print(transcription_obj)

    json_obj = {"transcription_obj": transcription_obj}
    return jsonify(json_obj)


def create_transcription_object(audioFileName):
    audio = wt.load_audio(audioFileName)
    model = wt.load_model("tiny.en", device="cpu")
    return wt.transcribe(model=model, audio=audio, language="en", temperature=0, beam_size=5, best_of=5)


def json_obj_convert(wt_transcription):
    word_id = 1
    json_obj_new = {}
    word_segments = wt_transcription["segments"]

    for index_out, seg in enumerate(word_segments):
        for index_in, word in enumerate(seg["words"]):
            json_obj_new[str(word_id)] = word
            word_id += 1

    json_obj_new = add_syllables(json_obj_new)

    return json_obj_new


def clean_word(word):
    return "".join(ch for ch in word if ch.isalnum()).lower()


def count_syllables_cmu(word):
    count = 0
    for ch in CMU_DICT[word][0]:
        if ch[-1].isdigit():
            count += 1
    return count


def count_syllables_pyphen(word):
    return len(PYPHEN_DICT.inserted(word).split('-')) + 1


def count_syllables(word):
    cleaned_word = clean_word(word)
    count = 0
    if cleaned_word in CMU_DICT:
        count += count_syllables_cmu(cleaned_word)
    else:
        count += count_syllables_pyphen(cleaned_word)

    return count


def add_syllables(transcription_obj):
    for key in transcription_obj.keys():
        transcription_obj[key]['syllable_count'] = int(count_syllables(transcription_obj[key]['text']))
    return transcription_obj


if __name__ == "__main__":
    app.run(debug=True, port=5000)



