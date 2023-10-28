from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
from flask_cors import CORS
import os
from deepgram import Deepgram
import pyphen
import cmudict
import stripe
import json
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import firestore
from firebase_admin import credentials
from firebase_admin import exceptions as firestore_exceptions
from stripe.error import StripeError
from types import SimpleNamespace

load_dotenv()
CMU_DICT = cmudict.dict()
PYPHEN_DICT = pyphen.Pyphen(lang='en')
# local testing
app = Flask(__name__)
os.environ['FIRESTORE_EMULATOR_HOST'] = "localhost:8080"
CLIENT = "http://localhost:3000"
# PRODUCTION
# CLIENT = "https://app.fluencymetrics.com"
# cors = CORS(app, resources={"/*": {"origins": [CLIENT]}})
# Firebase credentials
cred = credentials.Certificate("./firebase_credentials.json")
firebase_admin.initialize_app(cred)
db = firestore.client()
CORS(app)
DEEPGRAM_API_KEY = os.environ.get('DEEPGRAM_API_KEY')
STRIPE_API_KEY = os.environ.get('STRIPE_API_KEY')
STRIPE_END_POINT_KEY = os.environ.get('STRIPE_END_POINT_KEY')
stripe.api_key = STRIPE_API_KEY
endpoint_secret = STRIPE_END_POINT_KEY
APP_DIR = os.path.abspath(os.path.dirname(__file__))
dg_client = Deepgram(DEEPGRAM_API_KEY)
SUBSCRIPTION_STATUS = SimpleNamespace(TRIAL="trial", ACTIVE="active", INACTIVE="inactive")

WORD_DIC = pyphen.Pyphen(lang='en')


def syllables_in_word(word):
    return len(WORD_DIC.inserted(word).split('-'))


def get_total_syllables(raw_transcription):
    words = raw_transcription.split(' ')
    return sum(syllables_in_word(word) for word in words)


def update_user_subscription(stripe_id, update_obj, event_type):
    users_ref = db.collection('users')
    # This query should only ever return one document array length == 1

    user_documents = users_ref.where(field_path='subscription.stripe_id', op_string='==', value=stripe_id).get()

    '''
    in the case that invoid.paid is triggered right after checkout.completed it will throw error 
    that no user_documents exist and so it breaks, but this is fine because checkout.completed handled
    the users subscription object. Expect no matching user found error if a user first signs up
    '''
    if not user_documents:
        print(f"⚠️ No matching user found for stripe_id while handling event {event_type}:", stripe_id)
    elif len(user_documents) > 1:
        raise ValueError(f"⚠️ Multiple users found for stripe_id while handling event {event_type}:", stripe_id)
    else:
        user_uid = user_documents[0].id
        user_ref = db.collection('users').document(user_uid)
        user_ref.update(update_obj)


def handle_checkout_session_completed(event):
    try:
        print('checkout session')
        session = event['data']['object']
        user_id = session.client_reference_id
        subscription_id = session.subscription

        subscription_obj = stripe.Subscription.retrieve(subscription_id)
        sub_type = subscription_type(subscription_obj)
        end = subscription_obj.current_period_end

        user_ref = db.collection('users').document(user_id)
        update_obj = {
            'subscription.subscription_end_time': end,
            'subscription.subscription_status': SUBSCRIPTION_STATUS.ACTIVE,
            'subscription.stripe_id': subscription_obj.customer,
            'subscription.subscription_type': sub_type
        }
        user_ref.update(update_obj)
    except StripeError as error:
        print(f"Stripe Error: {error}")
    except firestore_exceptions.FirebaseError as error:
        print(f"Firestore: {error}")
    except Exception as error:
        print(f"Other Exception: {error}")


def handle_invoice_paid(event):
    event_type = "invoice_paid"
    print(event_type)
    invoice = event['data']['object']
    subscription_id = invoice.subscription
    subscription_obj = stripe.Subscription.retrieve(subscription_id)
    sub_type = subscription_type(subscription_obj)

    current_period_end = subscription_obj.current_period_end
    stripe_id = subscription_obj.customer

    update_obj = {
        'subscription.subscription_end_time': current_period_end,
        'subscription.subscription_status': SUBSCRIPTION_STATUS.ACTIVE,
        'subscription.subscription_type': sub_type
    }

    try:
        update_user_subscription(stripe_id, update_obj, event_type)
        print("invoice paid successful")
    except ValueError as error:
        print(f"Issue handling {event_type}:", error)


def handle_invoice_payment_failed(event):
    event_type = "invoice_payment_failed"
    invoice = event['data']['object']
    subscription_id = invoice.subscription
    subscription_obj = stripe.Subscription.retrieve(subscription_id)
    stripe_id = subscription_obj.customer

    update_obj = {
        'subscription.subscription_status': SUBSCRIPTION_STATUS.INACTIVE
    }

    try:
        update_user_subscription(stripe_id, update_obj, event_type)
    except ValueError as error:
        print(f"Issue handling {event_type}:", error)

    print('invoice payment failed')


def handle_customer_subscription_updated(event):
    # handles upgrades and or downgrades
    event_type = "customer_subscription_updated"
    print(event_type)
    upgrade_obj = event['data']['object']

    stripe_id = upgrade_obj.customer

    subscription_end_time = upgrade_obj.current_period_end
    subscription_type_int = 1 if upgrade_obj['plan']['interval'] == 'month' else 2

    update_obj = {
        'subscription.subscription_end_time': subscription_end_time,
        'subscription.subscription_type': subscription_type_int
    }

    try:
        update_user_subscription(stripe_id, update_obj, event_type)
    except ValueError as error:
        print(f"Issue handling {event_type}:", error)


def handle_customer_subscription_deleted(event):
    # subscription deleted event represents cancellations.
    event_type = "customer_subscription_deleted"

    event_obj = event['data']['object']
    stripe_id = event_obj.customer

    # When user cancels they get to use till end of period cancel_end_of_period will always be true for now
    # but may want to handle this differently in the future if I prorate subscription back to them
    cancel_end_of_period = event_obj.cancel_at_period_end

    update_obj = {
        'subscription.subscription_status': SUBSCRIPTION_STATUS.INACTIVE
    }

    try:
        update_user_subscription(stripe_id, update_obj, event_type)
    except ValueError as error:
        print(f"Issue handling {event_type}:", error)


def handle_customer_subscription_paused(event):
    print("customer subscription paused")


EVENT_HANDLERS = {
    'checkout.session.completed': handle_checkout_session_completed,
    'invoice.paid': handle_invoice_paid,
    'invoice.payment_failed': handle_invoice_payment_failed,
    'customer.subscription.deleted': handle_customer_subscription_deleted,
    'customer.subscription.paused': handle_customer_subscription_paused,
    'customer.subscription.updated': handle_customer_subscription_updated,
}


@app.route('/', methods=['GET'])
def base():
    return 'SUCCESS'


@app.route('/get_auto_transcription', methods=['POST'])
def get_transcription2():

    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    filename = secure_filename(file.filename)
    print("FILENAME: ", filename)
    file.save(os.path.join(f"{APP_DIR}/audio/", filename))

    dg_transcription = create_transcription_deepgram(f'{APP_DIR}/audio/' + filename)

    final_trans_obj = deepgram_response_convert(dg_transcription)
    final_trans_obj = {k: {**v, 'stuttered': False} for k, v in final_trans_obj.items()}
    syllables_total = sum(word['syllable_count'] for word in final_trans_obj.values())

    json_obj = {"transcription_obj": final_trans_obj,
                "syllables": syllables_total
                }

    return jsonify(json_obj)


@app.route('/manual_transcription', methods=['POST'])
def manual_transcription():
    manual_transcription_words = request.json.get("data").strip().split(" ")
    transcription_obj = {}
    for index, word in enumerate(manual_transcription_words):
        syllable_count = count_syllables(word)
        transcription_obj[index + 1] = {
            'text': word,
            'start': None,
            'end': None,
            'syllable_count': syllable_count,
        }

    print(transcription_obj)

    json_obj = {"transcription_obj": transcription_obj}
    return jsonify(json_obj)


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


def subscription_type(subscription_obj):
    sub_kind = subscription_obj['plan']['interval']
    return 1 if sub_kind == 'month' else 2


@app.route('/customer_portal', methods=['POST'])
def customer_portal():
    payload = request.data
    request_obj = json.loads(payload)
    customer_id = request_obj['customerId']

    portal_session = stripe.billing_portal.Session.create(
        customer=f"{customer_id}",
        return_url=f"{CLIENT}/tool"
    )

    return jsonify({"portal_url": portal_session.url})


@app.route('/stripe_subscription_webhook', methods=['POST'])
def stripe_subscription_webhook():
    print("STRIPE SUBSCRIPTIONS WEBHOOK")
    event = None
    payload = request.data

    try:
        event = json.loads(payload)
    except json.decoder.JSONDecodeError as e:
        print('⚠️  Webhook error while parsing basic request.' + str(e))
        return jsonify(success=False)
    if endpoint_secret:
        # Only verify the event if there is an endpoint secret defined
        # Otherwise use the basic event deserialized with json
        sig_header = request.headers.get('stripe-signature')
        try:
            event = stripe.Webhook.construct_event(
                payload, sig_header, endpoint_secret
            )
        except StripeError as error:
            print('⚠️  Webhook signature verification failed.' + str(error))
            return jsonify(success=False)

    # Handle the events
    handler = EVENT_HANDLERS.get(event['type'])
    if handler:
        handler(event)
    else:
        # Unexpected event type
        print('Unhandled event type {}'.format(event['type']))

    return jsonify(success=True)


@app.route('/subscription_cancel', methods=['POST'])
def subscription_cancel():
    data_obj = json.loads(request.data)
    user_id = data_obj["userId"]
    update_obj = {
        'subscription.subscription_status': SUBSCRIPTION_STATUS.INACTIVE,
        'subscription.subscription_type': 0
    }
    user_ref = db.collection('users').document(user_id)
    user_ref.update(update_obj)

    return jsonify(success=True)


if __name__ == "__main__":
    PORT = int(os.getenv("PORT")) if os.getenv("PORT") else 5001
    app.run("0.0.0.0", port=PORT, debug=True)
