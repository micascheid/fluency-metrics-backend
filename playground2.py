import os
import sys

import firebase_admin
from firebase_admin import firestore
from firebase_admin import credentials

os.environ['FIRESTORE_EMULATOR_HOST'] = 'localhost:8080'
cred = credentials.Certificate("./firebase_credentials.json")
firebase_admin.initialize_app(cred)
db = firestore.client()


def delete_collection(coll_ref, batch_size):
    docs = coll_ref.list_documents(page_size=batch_size)
    deleted = 0

    for doc in docs:
        print(f"Deleting doc {doc.id} => {doc.get().to_dict()}")
        doc.delete()
        deleted = deleted + 1

    if deleted >= batch_size:
        return delete_collection(coll_ref, batch_size)

def deleteCollection(name):
    collection_ref = db.collection("users").document("6gvAjAwSLMEK7bsS0poVfeUGv5ME").collection(name)
    delete_collection(collection_ref, 1)
    print("Collection:", name, " deleted")

def add_organization_example():
    org_name = "test_org"
    org_data = {
        'name': "cool name",
        'subscription_status': "active",
        'members': [],
        'allowable_users': 5,
        'sign_up_deadline': ""
    }
    db.collection("organizations").document(org_name).set(org_data)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        flag = sys.argv[1]
        if flag == '-f':
            deleteCollection("workspaces")
            deleteCollection("workspaces_index")
            exit(0)
        elif flag == "-a":
            add_organization_example()
        else:
            print("sorry unfamiliar with that option")
            exit(0)


    else:
        check_delete = input("Are you sure you want to delete collections: workspaces and workspaces_index? (y/n)")
        if check_delete == 'y':
            deleteCollection("workspaces")
            deleteCollection("workspaces_index")
        else:
            print("exiting....")
            exit(0)
