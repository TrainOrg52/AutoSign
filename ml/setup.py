from time import sleep

from firebase import Firebase
from firebase_admin import credentials, firestore, initialize_app, storage
from PIL import Image, ImageOps
from model import *
import torchvision.transforms as transforms
from object_detector.inference import ObjectDetector


# @brief: Every 3 seconds, performs a get request from firestore and looks for any inspection walkthrough processing status fields that are 'pending'. It then iterates over each instance of the occurrence and then processes the inspection.
# @params: N/A
# @return: N/A
# @authors: Benjamin Sanati, Charlie Powell
def runServer():
    while True:

        print("Checking for unprocessed inspection walkthroughs...")

        # get all inspections where status is "pending"
        inspection_walkthroughs = inspection_walkthroughs_collection.where(u'processingStatus', u'==', u'pending').get()

        # checking if any pending inspections are present
        if len(inspection_walkthroughs) > 0:
            # pending inspection present -> need to process it

            # iterating over each pending inspection
            for inspection_walkthrough in inspection_walkthroughs:
                # creating model object
                inspection_walkthrough = InspectionWalkthrough.from_doc(inspection_walkthrough)
                inspection_vehicle = vehicle_collection.document(inspection_walkthrough.vehicleID).get()
                inspection_vehicle = Vehicle.from_doc(inspection_vehicle)
                print(f"Identified inspection walkthrough for train {inspection_walkthrough.vehicleID}")

                # processing inspection walkthrough
                processInspectionWalkthrough(inspection_walkthrough, inspection_vehicle)

            print("All trains processed!")

        sleep(3)


# @brief: Processes each inspection walkthrough and uploads the processed results to firestore and storage. Uses the object detector followed by (non-optimal) predicted instance/ground-truth instance matching logic.
# @params: 'inspection_walkthrough' is an instance of an inspection that is yet to be processed.
# @return: N/A
# @authors: Benjamin Sanati, Charlie Powell
def processInspectionWalkthrough(inspection_walkthrough, inspection_vehicle):
    # ############################################### #
    # STEP 1: UPDATE STATUS OF INSPECTION WALKTHROUGH #
    # ############################################### #

    print("-----------------------")
    print(f"Processing train {inspection_walkthrough.vehicleID}...")
    inspection_walkthrough.processingStatus = "processing"
    inspection_walkthrough.update(db)
    inspection_walkthrough.conformanceStatus = "conforming"
    inspection_vehicle.conformanceStatus = "conforming"

    new_checkpoints = inspection_walkthrough.checkpoints

    # ########################################### #
    # STEP 2: ITERATE OVER INSPECTION CHECKPOINTS #
    # ########################################### #

    storage_roots = []
    inspection_checkpoint_signs = []
    inspection_checkpoints = []
    for inspection_checkpoint_id in inspection_walkthrough.checkpoints:
        # STEP 2.1: GATHERING INSPECTION CHECKPOINT OBJECT #

        inspection_checkpoint = inspection_checkpoints_collection.document(inspection_checkpoint_id).get()
        inspection_checkpoint = InspectionCheckpoint.from_doc(inspection_checkpoint)
        inspection_checkpoints.append(inspection_checkpoint)
        inspection_checkpoint_signs.append(inspection_checkpoint.signs)

        # STEP 2.2: DOWNLOADING UNPROCESSED MEDIA FROM CLOUD STROAGE #

        # defining path to Cloud Storage
        storage_root = f"/{inspection_walkthrough.vehicleID}/inspectionWalkthroughs/{inspection_walkthrough.id}/{inspection_checkpoint.id}"
        storage_roots.append(storage_root)
        storage_path = f"{storage_root}/unprocessed.png"
        print(f"\tIdentified checkpoint {inspection_checkpoint.id}")

        # creating directory for media
        # if not os.path.exists(f"storage/images/{inspection_checkpoint.id}"):
        #    os.mkdir(f"storage/images/{inspection_checkpoint.id}")

        # defining path to local storage
        local_path = f"storage/images/{inspection_checkpoint.id}.png"

        # downloading image from firebase storage
        storage.child(storage_path).download(local_path)

        # STEP 2.3: PRE-PROCESSING MEDIA #

        image = Image.open(local_path)
        image = ImageOps.exif_transpose(image)
        image = resize(image)
        image.save(local_path)

    # STEP 2.4: PROCESSING MEDIA AND SAVING TO STORAGE #

    print("\tDetecting Signs...", flush=True)

    dst_root = fr'storage/images'
    local_root = fr'storage/processed_images'
    _, identified_signs = obj_det(dst_root, local_root, storage, storage_roots)

    # STEP 2.5: COMPARE LOCATED LABELS TO EXPECTED #

    print("\tChecking Conformance Status...")
    for predicted_signs, inspection_signs, inspection_checkpoint in zip(identified_signs, inspection_checkpoint_signs,
                                                                        inspection_checkpoints):
        new_checkpoint_conformance = "conforming"

        # updating signs
        new_signs = inspection_signs

        for (sign_id, sign_conformance) in inspection_signs.items():
            # checking if inspection sign identified
            if sign_id in predicted_signs:
                # sign identified -> updating status

                # setting new sign conformance
                new_sign_conformance = "conforming"

                # removing identified sign from list
                idx = predicted_signs.index(sign_id)
                predicted_signs.pop(idx)

            else:
                # sign missing -> updating status

                # setting new sign conformance
                new_sign_conformance = "non-conforming"

                # setting checkpoint conformance
                new_checkpoint_conformance = "non-conforming"
                inspection_walkthrough.conformanceStatus = "non-conforming"
                inspection_vehicle.conformanceStatus = "non-conforming"

            # updating signs
            new_signs[sign_id] = new_sign_conformance

        # STEP 2.6: UPDATE FIREBASE WITH CONFORMANCE AND PROCESSING STATUS FOR INSPECTION #

        # updating inspection checkpoint object
        inspection_checkpoint.signs = new_signs
        inspection_checkpoint.conformanceStatus = new_checkpoint_conformance
        inspection_checkpoint.update(db)
        inspection_vehicle.update(db)

        new_checkpoints[inspection_checkpoint.id] = new_checkpoint_conformance

    inspection_walkthrough.processingStatus = "processed"
    inspection_walkthrough.checkpoints = new_checkpoints
    inspection_walkthrough.update(db)
    print(f"\tConformance status uploaded.")

    print(f"Processing Complete!")
    print("-----------------------")


# @brief: Performs setup of firebase firestore, storage and the initialization of YOLOv7. Calls the server once the setup is completed
# @params: N/A
# @return: N/A
# @authors: Benjamin Sanati, Charlie Powell
if __name__ == "__main__":
    # ############## #
    # FIREBASE SETUP #
    # ############## #
    print("Firebase Setup...")

    # firebase_admin package (for Firestore)
    cred = credentials.Certificate(r"creds/train-vis-firebase-adminsdk-rg3vz-ee79e15b22.json")
    initialize_app(cred)
    db = firestore.client()

    # Firebase package (for Storage)
    config = {
        'apiKey': "AIzaSyDO-pKWQQwoC9GPUE4TmtZrQBICqWKpZBE",
        'authDomain': "train-vis.firebaseapp.com",
        'projectId': "train-vis",
        'storageBucket': "train-vis.appspot.com",
        'messagingSenderId': "321213051352",
        'appId': "1:321213051352:web:6a0a03d5ecc53b039a82ad",
        "databaseURL": ""
    }
    firebase = Firebase(config)
    storage = firebase.storage()

    # firestore collections
    walkthroughs_collection = db.collection(u'walkthroughs')
    checkpoints_collection = db.collection(u'checkpoints')
    inspection_walkthroughs_collection = db.collection(u'inspectionWalkthroughs')
    inspection_checkpoints_collection = db.collection(u'inspectionCheckpoints')
    vehicle_collection = db.collection(u'vehicles')

    # ################# #
    # PROCESSING SET UP #
    # ################# #

    print("Object Detector Setup...")

    # define image transform
    resize = transforms.Resize([1280, 1280])

    # initialize object detector
    obj_det = ObjectDetector(image_size=1280, conf_thresh=0.55, iou_thresh=0.65, num_classes=11, view_img=False)

    print("Setup Complete!")
    print("-----------------------")

    # ############## #
    # RUNNING SERVER #
    # ############## #

    # running server
    runServer()
