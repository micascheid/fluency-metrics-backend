import whisper_timestamped as wt
from flask import Flask, jsonify, request, make_response
from werkzeug.utils import secure_filename
from flask_cors import CORS
import os
from deepgram import Deepgram
import pyphen
from nltk.corpus import cmudict
from dotenv import load_dotenv

load_dotenv()
CMU_DICT = cmudict.dict()
PYPHEN_DICT = pyphen.Pyphen(lang='en')
#local testing
app = Flask(__name__)
CORS(app)
DEEPGRAM_API_KEY = os.getenv('DEEPGRAM_API_KEY')
dg_client = Deepgram(DEEPGRAM_API_KEY)


WORD_DIC = pyphen.Pyphen(lang='en')


def syllables_in_word(word):
    return len(WORD_DIC.inserted(word).split('-'))


def get_total_syllables(raw_transcription):
    words = raw_transcription.split(' ')
    return sum(syllables_in_word(word) for word in words)


def audio_transcribe(audiofile):
    audio = wt.load_audio(audiofile)
    model = wt.load_model("tiny.en", device="cpu")

    return wt.transcribe(model, audio, language="en", temperature=0, beam_size=5, best_of=5)


@app.route('/', methods=['GET'])
def base():
    return 'hello world'


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
        file.save(os.path.join("./audio/", filename))

        dg_transcription = create_transcription_deepgram('./audio/'+filename)

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


def create_transcription_object(audioFileName):
    audio = wt.load_audio(audioFileName)
    model = wt.load_model("tiny.en", device="cpu")
    return wt.transcribe(model=model, audio=audio, language="en", temperature=0, beam_size=5, best_of=5)


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
    app.run(debug=True, port=5000)