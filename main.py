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
# CLIENT = "https://fluencymetrics-34537.web.app"
# cors = CORS(app, resources={"/*": {"origins": f"{CLIENT}"}})
CORS(app)
DEEPGRAM_API_KEY = os.environ.get('DEEPGRAM_API_KEY')
APP_DIR = os.path.abspath(os.path.dirname(__file__))
dg_client = Deepgram(DEEPGRAM_API_KEY)
QUICK_TEST_WITH_SYLLABLES = {'1': {'word': 'my', 'start': 0.24, 'end': 0.56, 'confidence': 0.64208984, 'punctuated_word': 'My', 'syllable_count': 1}, '2': {'word': 'name', 'start': 0.56, 'end': 0.79999995, 'confidence': 0.9995117, 'punctuated_word': 'name', 'syllable_count': 1}, '3': {'word': 'is', 'start': 0.79999995, 'end': 1.3, 'confidence': 0.99853516, 'punctuated_word': 'is', 'syllable_count': 1}, '4': {'word': 'ray', 'start': 2.48, 'end': 2.98, 'confidence': 0.390625, 'punctuated_word': 'Ray', 'syllable_count': 1}, '5': {'word': 'ray', 'start': 3.9199998, 'end': 4.08, 'confidence': 0.34570312, 'punctuated_word': 'Ray', 'syllable_count': 1}, '6': {'word': 'demitz', 'start': 4.08, 'end': 4.58, 'confidence': 0.50809735, 'punctuated_word': 'Demitz.', 'syllable_count': 3}, '7': {'word': "i'm", 'start': 4.72, 'end': 5.22, 'confidence': 0.95751953, 'punctuated_word': "I'm", 'syllable_count': 1}, '8': {'word': "i'm", 'start': 8.275, 'end': 8.775, 'confidence': 0.9382324, 'punctuated_word': "I'm", 'syllable_count': 1}, '9': {'word': 'twenty', 'start': 9.715, 'end': 10.035, 'confidence': 0.9951172, 'punctuated_word': 'twenty', 'syllable_count': 2}, '10': {'word': 'years', 'start': 10.035, 'end': 10.355, 'confidence': 0.9995117, 'punctuated_word': 'years', 'syllable_count': 1}, '11': {'word': 'old', 'start': 10.355, 'end': 10.855, 'confidence': 0.7961426, 'punctuated_word': 'old,', 'syllable_count': 1}, '12': {'word': "i'm", 'start': 11.235, 'end': 11.635, 'confidence': 0.9975586, 'punctuated_word': "I'm", 'syllable_count': 1}, '13': {'word': 'from', 'start': 11.635, 'end': 12.135, 'confidence': 0.99902344, 'punctuated_word': 'from', 'syllable_count': 1}, '14': {'word': 'new', 'start': 12.914999, 'end': 13.075, 'confidence': 0.98291016, 'punctuated_word': 'New', 'syllable_count': 1}, '15': {'word': 'york', 'start': 13.075, 'end': 13.315, 'confidence': 0.99902344, 'punctuated_word': 'York', 'syllable_count': 1}, '16': {'word': 'city', 'start': 13.315, 'end': 13.635, 'confidence': 0.99853516, 'punctuated_word': 'City', 'syllable_count': 2}, '17': {'word': 'and', 'start': 13.635, 'end': 13.875, 'confidence': 0.7607422, 'punctuated_word': 'and', 'syllable_count': 1}, '18': {'word': 'a', 'start': 13.875, 'end': 13.955, 'confidence': 0.6772461, 'punctuated_word': 'a', 'syllable_count': 1}, '19': {'word': 'stutter', 'start': 13.955, 'end': 14.42, 'confidence': 0.87565106, 'punctuated_word': 'stutter.', 'syllable_count': 2}, '20': {'word': 'the', 'start': 15.78, 'end': 15.940001, 'confidence': 0.5600586, 'punctuated_word': 'The', 'syllable_count': 1}, '21': {'word': 'hardest', 'start': 15.940001, 'end': 16.44, 'confidence': 0.9970703, 'punctuated_word': 'hardest', 'syllable_count': 2}, '22': {'word': 'part', 'start': 16.5, 'end': 16.74, 'confidence': 0.89697266, 'punctuated_word': 'part', 'syllable_count': 1}, '23': {'word': 'of', 'start': 16.74, 'end': 16.9, 'confidence': 0.9667969, 'punctuated_word': 'of', 'syllable_count': 1}, '24': {'word': 'stuttering', 'start': 16.9, 'end': 17.3, 'confidence': 0.99658203, 'punctuated_word': 'stuttering', 'syllable_count': 3}, '25': {'word': 'is', 'start': 17.3, 'end': 17.46, 'confidence': 0.99902344, 'punctuated_word': 'is', 'syllable_count': 1}, '26': {'word': 'not', 'start': 17.46, 'end': 17.619999, 'confidence': 0.99902344, 'punctuated_word': 'not', 'syllable_count': 1}, '27': {'word': 'the', 'start': 17.619999, 'end': 17.78, 'confidence': 0.9995117, 'punctuated_word': 'the', 'syllable_count': 1}, '28': {'word': 'physical', 'start': 17.78, 'end': 18.18, 'confidence': 0.9995117, 'punctuated_word': 'physical', 'syllable_count': 3}, '29': {'word': 'stutter', 'start': 18.18, 'end': 18.68, 'confidence': 0.97721356, 'punctuated_word': 'stutter.', 'syllable_count': 2}, '30': {'word': "it's", 'start': 18.74, 'end': 19.06, 'confidence': 0.99780273, 'punctuated_word': "It's", 'syllable_count': 1}, '31': {'word': 'the', 'start': 19.06, 'end': 19.56, 'confidence': 0.99609375, 'punctuated_word': 'the', 'syllable_count': 1}, '32': {'word': 'mental', 'start': 19.7, 'end': 20.1, 'confidence': 0.9995117, 'punctuated_word': 'mental', 'syllable_count': 2}, '33': {'word': 'and', 'start': 20.1, 'end': 20.26, 'confidence': 0.64208984, 'punctuated_word': 'and', 'syllable_count': 1}, '34': {'word': 'emotional', 'start': 20.26, 'end': 20.74, 'confidence': 0.9995117, 'punctuated_word': 'emotional', 'syllable_count': 4}, '35': {'word': 'baggage', 'start': 20.74, 'end': 21.22, 'confidence': 1.0, 'punctuated_word': 'baggage', 'syllable_count': 2}, '36': {'word': 'that', 'start': 21.22, 'end': 21.380001, 'confidence': 1.0, 'punctuated_word': 'that', 'syllable_count': 1}, '37': {'word': 'comes', 'start': 21.380001, 'end': 21.619999, 'confidence': 0.9995117, 'punctuated_word': 'comes', 'syllable_count': 1}, '38': {'word': 'along', 'start': 21.619999, 'end': 21.94, 'confidence': 0.99853516, 'punctuated_word': 'along', 'syllable_count': 2}, '39': {'word': 'with', 'start': 21.94, 'end': 22.1, 'confidence': 0.99902344, 'punctuated_word': 'with', 'syllable_count': 1}, '40': {'word': 'it', 'start': 22.1, 'end': 22.365, 'confidence': 0.94091797, 'punctuated_word': 'it.', 'syllable_count': 1}, '41': {'word': "people's", 'start': 22.845, 'end': 23.324999, 'confidence': 0.8688965, 'punctuated_word': "people's", 'syllable_count': 2}, '42': {'word': 'first', 'start': 23.324999, 'end': 23.645, 'confidence': 0.99316406, 'punctuated_word': 'first', 'syllable_count': 1}, '43': {'word': 'impression', 'start': 23.645, 'end': 24.045, 'confidence': 0.9995117, 'punctuated_word': 'impression', 'syllable_count': 3}, '44': {'word': 'of', 'start': 24.045, 'end': 24.205, 'confidence': 0.78027344, 'punctuated_word': 'of', 'syllable_count': 1}, '45': {'word': 'me', 'start': 24.205, 'end': 24.445, 'confidence': 0.9995117, 'punctuated_word': 'me', 'syllable_count': 1}, '46': {'word': 'feels', 'start': 24.445, 'end': 24.845, 'confidence': 0.9746094, 'punctuated_word': 'feels', 'syllable_count': 1}, '47': {'word': 'like', 'start': 24.845, 'end': 25.164999, 'confidence': 0.9921875, 'punctuated_word': 'like', 'syllable_count': 1}, '48': {'word': "it's", 'start': 25.164999, 'end': 25.404999, 'confidence': 0.9970703, 'punctuated_word': "it's", 'syllable_count': 1}, '49': {'word': 'not', 'start': 25.404999, 'end': 25.725, 'confidence': 0.99902344, 'punctuated_word': 'not', 'syllable_count': 1}, '50': {'word': 'really', 'start': 25.725, 'end': 26.045, 'confidence': 0.99853516, 'punctuated_word': 'really', 'syllable_count': 2}, '51': {'word': 'me', 'start': 26.045, 'end': 26.365, 'confidence': 0.89208984, 'punctuated_word': 'me.', 'syllable_count': 1}, '52': {'word': 'when', 'start': 26.365, 'end': 26.525, 'confidence': 0.8618164, 'punctuated_word': 'When', 'syllable_count': 1}, '53': {'word': 'i', 'start': 26.525, 'end': 26.685, 'confidence': 0.9995117, 'punctuated_word': 'I', 'syllable_count': 1}, '54': {'word': 'open', 'start': 26.685, 'end': 26.925, 'confidence': 0.43774414, 'punctuated_word': 'open', 'syllable_count': 2}, '55': {'word': 'my', 'start': 26.925, 'end': 27.085, 'confidence': 0.99609375, 'punctuated_word': 'my', 'syllable_count': 1}, '56': {'word': 'mouth', 'start': 27.085, 'end': 27.325, 'confidence': 1.0, 'punctuated_word': 'mouth', 'syllable_count': 1}, '57': {'word': 'and', 'start': 27.325, 'end': 27.485, 'confidence': 0.91748047, 'punctuated_word': 'and', 'syllable_count': 1}, '58': {'word': 'i', 'start': 27.485, 'end': 27.564999, 'confidence': 0.95996094, 'punctuated_word': 'I', 'syllable_count': 1}, '59': {'word': 'look', 'start': 27.564999, 'end': 27.805, 'confidence': 0.99560547, 'punctuated_word': 'look', 'syllable_count': 1}, '60': {'word': 'like', 'start': 27.805, 'end': 27.965, 'confidence': 0.9951172, 'punctuated_word': 'like', 'syllable_count': 1}, '61': {'word': "i'm", 'start': 27.965, 'end': 28.125, 'confidence': 0.9992676, 'punctuated_word': "I'm", 'syllable_count': 1}, '62': {'word': 'in', 'start': 28.125, 'end': 28.365, 'confidence': 0.9995117, 'punctuated_word': 'in', 'syllable_count': 1}, '63': {'word': 'pain', 'start': 28.365, 'end': 28.865, 'confidence': 0.86572266, 'punctuated_word': 'pain.', 'syllable_count': 1}}



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
    quick = True
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
        options = {"smart_format": True, "model": "nova", "language": "en-US"}

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
    PORT = int(os.getenv("PORT")) if os.getenv("PORT") else 5001
    app.run("0.0.0.0", port=PORT, debug=True)
