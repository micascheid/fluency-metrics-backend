from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
from flask_cors import CORS
import os
from deepgram import Deepgram
import pyphen
import cmudict
# from nltk.corpus import cmudict
from dotenv import load_dotenv

load_dotenv()
CMU_DICT = cmudict.dict()
PYPHEN_DICT = pyphen.Pyphen(lang='en')
#local testing
app = Flask(__name__)
#PRODUCTION
CLIENT = "https://fluencymetrics-34537.web.app"
cors = CORS(app, resources={"/*": {"origins": f"{CLIENT}"}})
# CORS(app)
DEEPGRAM_API_KEY = os.environ.get('DEEPGRAM_API_KEY')
APP_DIR = os.path.abspath(os.path.dirname(__file__))
dg_client = Deepgram(DEEPGRAM_API_KEY)
QUICK_TEST_WITH_SYLLABLES = {'1': {'text': 'My', 'start': 0.2, 'end': 0.56, 'confidence': 0.862, 'syllable_count':2},
                             '2': {'text': 'name', 'start': 0.56, 'end': 0.84, 'confidence': 0.995, 'syllable_count': 1}, '3': {'text': 'is', 'start': 0.84, 'end': 2.68, 'confidence': 0.994, 'syllable_count': 1}, '4': {'text': 'Ray', 'start': 2.68, 'end': 4.06, 'confidence': 0.722, 'syllable_count': 2}, '5': {'text': 'Demnitz,', 'start': 4.06, 'end': 4.74, 'confidence': 0.265, 'syllable_count': 3}, '6': {'text': "I'm", 'start': 6.52, 'end': 8.64, 'confidence': 0.894, 'syllable_count': 3}, '7': {'text': '20', 'start': 8.64, 'end': 10.14, 'confidence': 0.611, 'syllable_count': 2}, '8': {'text': 'years', 'start': 10.14, 'end': 10.4, 'confidence': 0.954, 'syllable_count': 2}, '9': {'text': 'old,', 'start': 10.4, 'end': 11.16, 'confidence': 0.71, 'syllable_count': 3}, '10': {'text': "I'm", 'start': 11.52, 'end': 11.76, 'confidence': 0.809, 'syllable_count': 1}, '11': {'text': 'from', 'start': 11.76, 'end': 12.76, 'confidence': 0.985, 'syllable_count': 3}, '12': {'text': 'New', 'start': 12.76, 'end': 13.2, 'confidence': 0.42, 'syllable_count': 3}, '13': {'text': 'York', 'start': 13.2, 'end': 13.38, 'confidence': 0.985, 'syllable_count': 2}, '14': {'text': 'City,', 'start': 13.38, 'end': 13.72, 'confidence': 0.546, 'syllable_count': 1}, '15': {'text': 'and', 'start': 13.78, 'end': 13.96, 'confidence': 0.97, 'syllable_count': 3}, '16': {'text': 'I', 'start': 13.96, 'end': 14.1, 'confidence': 0.882, 'syllable_count': 2}, '17': {'text': 'stutter.', 'start': 14.1, 'end': 14.3, 'confidence': 0.858, 'syllable_count': 1}, '18': {'text': 'The', 'start': 15.62, 'end': 16.02, 'confidence': 0.803, 'syllable_count': 1}, '19': {'text': 'hardest', 'start': 16.02, 'end': 16.56, 'confidence': 0.996, 'syllable_count': 1}, '20': {'text': 'part', 'start': 16.56, 'end': 16.82, 'confidence': 0.988, 'syllable_count': 2}, '21': {'text': 'of', 'start': 16.82, 'end': 16.98, 'confidence': 0.964, 'syllable_count': 3}, '22': {'text': 'stuttering', 'start': 16.98, 'end': 17.32, 'confidence': 0.859, 'syllable_count': 3}, '23': {'text': 'is', 'start': 17.32, 'end': 17.48, 'confidence': 0.977, 'syllable_count': 2}, '24': {'text': 'not', 'start': 17.48, 'end': 17.7, 'confidence': 0.978, 'syllable_count': 3}, '25': {'text': 'the', 'start': 17.7, 'end': 17.9, 'confidence': 0.974, 'syllable_count': 1}, '26': {'text': 'physical', 'start': 17.9, 'end': 18.22, 'confidence': 0.993, 'syllable_count': 3}, '27': {'text': 'stutter.', 'start': 18.22, 'end': 18.76, 'confidence': 0.791, 'syllable_count': 3}, '28': {'text': "It's", 'start': 18.82, 'end': 19.28, 'confidence': 0.862, 'syllable_count': 1}, '29': {'text': 'the', 'start': 19.28, 'end': 19.84, 'confidence': 0.995, 'syllable_count': 3}, '30': {'text': 'mental', 'start': 19.84, 'end': 20.2, 'confidence': 0.982, 'syllable_count': 3}, '31': {'text': 'and', 'start': 20.2, 'end': 20.36, 'confidence': 0.95, 'syllable_count': 2}, '32': {'text': 'emotional', 'start': 20.36, 'end': 20.74, 'confidence': 0.999, 'syllable_count': 2}, '33': {'text': 'baggage', 'start': 20.74, 'end': 21.24, 'confidence': 0.959, 'syllable_count': 1}, '34': {'text': 'that', 'start': 21.24, 'end': 21.4, 'confidence': 0.808, 'syllable_count': 3}, '35': {'text': 'comes', 'start': 21.4, 'end': 21.68, 'confidence': 0.954, 'syllable_count': 1}, '36': {'text': 'along', 'start': 21.68, 'end': 22.04, 'confidence': 0.991, 'syllable_count': 2}, '37': {'text': 'with', 'start': 22.04, 'end': 22.24, 'confidence': 0.981, 'syllable_count': 2}, '38': {'text': 'it.', 'start': 22.24, 'end': 22.74, 'confidence': 0.954, 'syllable_count': 1}, '39': {'text': "People's", 'start': 22.92, 'end': 23.54, 'confidence': 0.92, 'syllable_count': 3}, '40': {'text': 'first', 'start': 23.54, 'end': 23.8, 'confidence': 0.99, 'syllable_count': 1}, '41': {'text': 'impression', 'start': 23.8, 'end': 24.16, 'confidence': 0.996, 'syllable_count': 3}, '42': {'text': 'of', 'start': 24.16, 'end': 24.3, 'confidence': 0.848, 'syllable_count': 2}, '43': {'text': 'me', 'start': 24.3, 'end': 24.54, 'confidence': 0.998, 'syllable_count': 2}, '44': {'text': 'feels', 'start': 24.54, 'end': 24.92, 'confidence': 0.952, 'syllable_count': 2}, '45': {'text': 'like', 'start': 24.92, 'end': 25.26, 'confidence': 0.988, 'syllable_count': 2}, '46': {'text': "it's", 'start': 25.26, 'end': 25.54, 'confidence': 0.978, 'syllable_count': 3}, '47': {'text': 'not', 'start': 25.54, 'end': 25.86, 'confidence': 0.997, 'syllable_count': 3}, '48': {'text': 'really', 'start': 25.86, 'end': 26.14, 'confidence': 0.994, 'syllable_count': 3}, '49': {'text': 'neat.', 'start': 26.14, 'end': 26.4, 'confidence': 0.368, 'syllable_count': 2}, '50': {'text': 'When', 'start': 26.46, 'end': 26.66, 'confidence': 0.215, 'syllable_count': 2}, '51': {'text': 'I', 'start': 26.66, 'end': 26.8, 'confidence': 0.974, 'syllable_count': 1}, '52': {'text': 'open', 'start': 26.8, 'end': 27.0, 'confidence': 0.635, 'syllable_count': 2}, '53': {'text': 'my', 'start': 27.0, 'end': 27.16, 'confidence': 0.994, 'syllable_count': 1}, '54': {'text': 'mouth', 'start': 27.16, 'end': 27.42, 'confidence': 0.996, 'syllable_count': 3}, '55': {'text': 'and', 'start': 27.42, 'end': 27.6, 'confidence': 0.763, 'syllable_count': 3}, '56': {'text': 'I', 'start': 27.6, 'end': 27.72, 'confidence': 0.676, 'syllable_count': 3}, '57': {'text': 'look', 'start': 27.72, 'end': 27.88, 'confidence': 0.982, 'syllable_count': 3}, '58': {'text': 'like', 'start': 27.88, 'end': 28.04, 'confidence': 0.98, 'syllable_count': 1}, '59': {'text': "I'm", 'start': 28.04, 'end': 28.24, 'confidence': 0.981, 'syllable_count': 3}, '60': {'text': 'in', 'start': 28.24, 'end': 28.34, 'confidence': 0.999, 'syllable_count': 1}, '61': {'text': 'pain.', 'start': 28.34, 'end': 28.76, 'confidence': 0.468, 'syllable_count': 1}}

WORD_DIC = pyphen.Pyphen(lang='en')


def syllables_in_word(word):
    return len(WORD_DIC.inserted(word).split('-'))


def get_total_syllables(raw_transcription):
    words = raw_transcription.split(' ')
    return sum(syllables_in_word(word) for word in words)


# def audio_transcribe(audiofile):
#     audio = wt.load_audio(audiofile)
#     model = wt.load_model("tiny.en", device="cpu")
#
#     return wt.transcribe(model, audio, language="en", temperature=0, beam_size=5, best_of=5)


@app.route('/', methods=['GET'])
def base():
    return 'fluencymetrics backend'


@app.route('/get_transcription2', methods=['POST'])
def get_transcription2():
    quick = False
    if not quick:
        if 'file' not in request.files:
            return "No file part", 400
        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400
        filename = secure_filename(file.filename)
        print("FILENAME: ", filename)
        file.save(os.path.join(f"{APP_DIR}/audio/", filename))

        dg_transcription = create_transcription_deepgram(f'{APP_DIR}/audio/'+filename)

        final_trans_obj = deepgram_response_convert(dg_transcription)
        final_trans_obj = {k: {**v, 'stuttered': False} for k, v in final_trans_obj.items()}
        syllables_total = sum(word['syllable_count'] for word in final_trans_obj.values())

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
            'syllable_count': syllable_count,
        }

    print(transcription_obj)

    json_obj = {"transcription_obj": transcription_obj}
    return jsonify(json_obj)


# def create_transcription_object(audioFileName):
#     audio = wt.load_audio(audioFileName)
#     model = wt.load_model("tiny.en", device="cpu")
#     return wt.transcribe(model=model, audio=audio, language="en", temperature=0, beam_size=5, best_of=5)


def create_transcription_deepgram(audioFileName):
    MIMETYPE = 'audio/wav'
    with open(audioFileName, 'rb') as audio:
        source = {'buffer': audio, 'mimetype': MIMETYPE}
        options = {"smart_format": True, "model": "general-enhanced", "language": "en-US"}

        response = dg_client.transcription.sync_prerecorded(source, options)
        result = response['results']['channels'][0]['alternatives'][0]['words']
        return result


def deepgram_response_convert(dg_response):
    word_id = 1
    json_obj_new = {}
    for index, word in enumerate(dg_response):
        json_obj_new[str(word_id)] = word
        word_id += 1

    json_obj_new = add_syllables(json_obj_new)

    return json_obj_new


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
        transcription_obj[key]['syllable_count'] = int(count_syllables(transcription_obj[key]['word']))
    return transcription_obj


if __name__ == "__main__":
    PORT = int(os.getenv("PORT")) if os.getenv("PORT") else 8080
    app.run("0.0.0.0", port=PORT, debug=True)
