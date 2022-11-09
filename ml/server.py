from time import sleep

from firebase import Firebase
from firebase_admin import credentials, firestore, initialize_app, storage
from PIL import Image, ImageOps
from model import *
import torchvision.transforms as transforms
from object_detector.inference import ObjectDetector

"""
    @brief: Every 3 seconds, performs a get request from firestore and looks for any inspection walkthrough processing status fields that are 'pending'. It then iterates over each instance of the occurrence and then processes the inspection.
    @params: N/A
    @return: N/A
    @authors: Benjamin Sanati, Charlie Powell
"""
def runServer():
    while True:

        print("Checking for unprocessed inspection walkthroughs...")

        # get all inspections where status is "pending"
        vehicle_inspections = vehicle_inspections_collection.where(u'processingStatus', u'==', u'pending').get()

        # checking if any pending inspections are present
        if len(vehicle_inspections) > 0:
            # pending inspection present -> need to process it

            # iterating over each pending inspection
            for vehicle_inspection in vehicle_inspections:
                # creating model object
                vehicle_inspection = vehicleInspection.from_doc(vehicle_inspection)
                vehicle = vehicle_collection.document(vehicle_inspection.vehicleID).get()
                vehicle = Vehicle.from_doc(vehicle)
                print(f"Identified inspection walkthrough for train {vehicle_inspection.vehicleID}")

                # processing inspection walkthrough
                processInspectionWalkthrough(vehicle_inspection, vehicle)

            print("All trains processed!")

        sleep(3)


"""
    @brief: Processes each inspection walkthrough and uploads the processed results to firestore and storage. Uses the object detector followed by (non-optimal) predicted instance/ground-truth instance matching logic.
    @params: 'inspection_walkthrough' is an instance of an inspection that is yet to be processed.
    @return: N/A
    @authors: Benjamin Sanati, Charlie Powell
"""
def processInspectionWalkthrough(vehicle_inspection, vehicle):
    # ############################################### #
    # STEP 1: UPDATE STATUS OF INSPECTION WALKTHROUGH #
    # ############################################### #

    print("-----------------------")
    print(f"Processing train {vehicle_inspection.vehicleID}...")

    # update processing status of the vehicle inspection object
    vehicle_inspection.processingStatus = "processing"
    vehicle_inspection.update(db)

    # initialize conformance status of the vehicle inspection
    vehicle_inspection.conformanceStatus = "conforming"
    vehicle.conformanceStatus = "conforming"

    # ########################################### #
    # STEP 2: ITERATE OVER INSPECTION CHECKPOINTS #
    # ########################################### #

    checkpoint_inspections = checkpoint_inspections_collection.where(u'vehicleInspectionID', u'==',
                                                                     vehicle_inspection.id).get()

    storage_roots = []
    vehicle_checkpoints = []
    vehicle_checkpoint_signs = []
    for checkpoint_inspection in checkpoint_inspections:
        # STEP 2.1: GATHERING INSPECTION CHECKPOINT DOCUMENT #
        vehicle_checkpoint = CheckpointInspection.from_doc(checkpoint_inspection)

        # STEP 2.2: DOWNLOADING UNPROCESSED MEDIA FROM CLOUD STROAGE #

        # defining path to Cloud Storage
        storage_root = f"/{vehicle_inspection.vehicleID}/vehicleInspections/{vehicle_inspection.id}/{vehicle_checkpoint.id}"
        storage_path = f"{storage_root}/unprocessed.png"
        print(f"\tIdentified checkpoint {vehicle_checkpoint.id}")

        # defining path to local storage
        local_path = f"samples/images/{vehicle_checkpoint.id}.png"

        # downloading image from firebase storage
        storage.child(storage_path).download(local_path)

        # STEP 2.3: PRE-PROCESSING MEDIA #

        image = Image.open(local_path)
        image = ImageOps.exif_transpose(image)
        image = resize(image)
        image.save(local_path)

        # adding data to lists
        storage_roots.append(storage_root)
        vehicle_checkpoints.append(vehicle_checkpoint)
        vehicle_checkpoint_signs.append(vehicle_checkpoint.signs)

    """
    for vehicle_inspection_id in checkpoint_inspections:
        # STEP 2.1: GATHERING INSPECTION CHECKPOINT OBJECT AND CHECKPOINT OBJECT #

        vehicle_checkpoint = checkpoint_inspections_collection.document(vehicle_inspection_id).get()
        vehicle_checkpoint = CheckpointInspection.from_doc(vehicle_checkpoint)
        #vehicle_checkpoints.append(vehicle_checkpoint)
        #vehicle_checkpoint_signs.append(vehicle_checkpoint.signs)

        # STEP 2.2: DOWNLOADING UNPROCESSED MEDIA FROM CLOUD STROAGE #

        # defining path to Cloud Storage
        storage_root = f"/{vehicle_inspection.vehicleID}/inspectionWalkthroughs/{vehicle_inspection.id}/{vehicle_checkpoint.id}"
        storage_roots.append(storage_root)
        storage_path = f"{storage_root}/unprocessed.png"
        print(f"\tIdentified checkpoint {vehicle_checkpoint.id}")

        # defining path to local storage
        local_path = f"samples/images/{vehicle_checkpoint.id}.png"

        # downloading image from firebase storage
        storage.child(storage_path).download(local_path)

        # STEP 2.3: PRE-PROCESSING MEDIA #

        image = Image.open(local_path)
        image = ImageOps.exif_transpose(image)
        image = resize(image)
        image.save(local_path)
    """

    # STEP 2.4: PROCESSING MEDIA AND SAVING TO STORAGE #

    print("\tDetecting Signs...", flush=True)

    dst_root = fr'samples/images'
    local_root = fr'samples/processed_images'
    _, identified_signs = obj_det(dst_root, local_root, storage, storage_roots)

    # STEP 2.5: COMPARE LOCATED LABELS TO EXPECTED #

    print("\tChecking Conformance Status...")
    for predicted_signs, vehicle_sign, vehicle_checkpoint in zip(identified_signs, vehicle_checkpoint_signs,
                                                                 vehicle_checkpoints):
        new_checkpoint_conformance = "conforming"

        # updating signs
        new_signs = vehicle_sign

        checkpoint = checkpoints_collection.document(vehicle_checkpoint.checkpointID).get()
        checkpoint = Checkpoint.from_doc(checkpoint)

        checkpoint.conformanceStatus = "processing"
        checkpoint.lastVehicleInspectionResult = "conforming"

        for (sign_id, sign_conformance) in vehicle_sign.items():
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
                vehicle_inspection.conformanceStatus = "non-conforming"
                vehicle.conformanceStatus = "non-conforming"

                checkpoint.lastVehicleInspectionResult = "non-conforming"

            # updating checkpoints object
            checkpoint.conformanceStatus = checkpoint.lastVehicleInspectionResult
            checkpoint.lastVehicleInspectionID = vehicle_inspection.id
            checkpoint.update(db)

            # updating signs
            new_signs[sign_id] = new_sign_conformance

        # STEP 2.6: UPDATE FIREBASE WITH CONFORMANCE AND PROCESSING STATUS FOR INSPECTION #

        # updating inspection checkpoint object
        vehicle_checkpoint.signs = new_signs
        vehicle_checkpoint.conformanceStatus = new_checkpoint_conformance
        vehicle_checkpoint.update(db)

    vehicle_inspection.processingStatus = "processed"
    vehicle_inspection.update(db)

    # updating inspection vehicle object
    vehicle.update(db)

    print(f"\tConformance status uploaded.")

    print(f"Processing Complete!")
    print("-----------------------")


"""
    @brief: Performs setup of firebase firestore, storage and the initialization of YOLOv7. Calls the server once the setup is completed
    @params: N/A
    @return: N/A
    @authors: Benjamin Sanati, Charlie Powell
"""
if __name__ == "__main__":
    # ############## #
    # FIREBASE SETUP #
    # ############## #
    print("Firebase Setup...")

    # firebase_admin package (for Firestore)
    cred = credentials.Certificate(r".configs/train-vis-firebase-adminsdk-rg3vz-ee79e15b22.json")
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
    checkpoints_collection = db.collection(u'checkpoints')
    vehicle_inspections_collection = db.collection(u'vehicleInspections')
    checkpoint_inspections_collection = db.collection(u'checkpointInspections')
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
