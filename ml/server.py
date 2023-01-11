from model import *
from time import sleep
from PIL import Image, ImageOps
from tools.video_logic.logic import *

from firebase import Firebase
from firebase_admin import credentials, firestore, initialize_app, storage
import torchvision.transforms as transforms
from object_detector.inference import ObjectDetector
from damage_detector.inference import DamageDetector


"""
    @brief: 
        Every 3 seconds, performs a get request from firestore and looks for any inspection walkthrough processing status fields that are 'pending'. 
        It then iterates over each instance of the occurrence and then processes the inspection.
    
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
                processVehicleInspection(vehicle_inspection, vehicle)

            print("All trains processed!")

        sleep(3)


"""
    @brief: 
        Processes each inspection walkthrough and uploads the processed results to firestore and storage. Uses the object detector followed by 
        (non-optimal) predicted instance/ground-truth instance matching logic.
    
    @params: 'inspection_walkthrough' is an instance of an inspection that is yet to be processed.
    @return: N/A
    @authors: Benjamin Sanati, Charlie Powell
"""
def processVehicleInspection(vehicle_inspection, vehicle):
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

    vehicle_checkpoints, vehicle_checkpoint_signs = [], []
    identified_signs, conformance_statuses = [], []
    media_type = []
    for checkpoint_inspection in checkpoint_inspections:
        print(identified_signs)
        # STEP 2.1: GATHERING INSPECTION CHECKPOINT DOCUMENT #
        vehicle_checkpoint = CheckpointInspection.from_doc(checkpoint_inspection)

        # STEP 2.2: DOWNLOADING UNPROCESSED MEDIA FROM CLOUD STORAGE #

        media_type.append(vehicle_checkpoint.captureType)

        if vehicle_checkpoint.captureType == 'photo':
            # defining path to Cloud Storage
            storage_path = f"/{vehicle_inspection.vehicleID}/vehicleInspections/{vehicle_inspection.id}/{vehicle_checkpoint.id}.png"
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

            # image detection
            dst_root = 'samples/images'
            local_root = 'samples/processed_images'

            _, image_identified_signs = obj_det(dst_root, local_root)
            identified_signs += image_identified_signs

            # damage detection
            if len(image_identified_signs[0]) != 0:
                print(f"\tSign Damage Detection Processing...", flush=True)
                damage_root = 'samples/normalized_images'
                damage_classifications = BeIT_damage_detector(damage_root)
                conformance_statuses.append(damage_classifications)
            else:
                conformance_statuses.append([])
        elif vehicle_checkpoint.captureType == 'video':
            # defining path to Cloud Storage
            storage_path = f"/{vehicle_inspection.vehicleID}/vehicleInspections/{vehicle_inspection.id}/{vehicle_checkpoint.id}.mp4"
            print(f"\tIdentified checkpoint {vehicle_checkpoint.id}")

            # defining path to local storage
            local_path = f"samples/videos/{vehicle_checkpoint.id}.mp4"

            # downloading image from firebase storage
            storage.child(storage_path).download(local_path)

            # ################ #
            # VIDEO PROCESSING #
            # ################ #

            print("\tVideo Processing...")

            # load video
            video = cv2.VideoCapture(local_path)
            total_num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

            # frame image path in local
            frame_root = f"samples/video_images"

            # video preprocessing
            frame_num, count = 0, 5 # math.floor(total_num_frames / 10)
            #count = count if (count > 5) else 6
            count = count if (count < 11) else 10
            video_processing(video, frame_num, count, frame_root, total_num_frames)

            if os.path.exists(local_path):
                os.remove(local_path)

            # video detection
            dst_root = 'samples/video_images'
            local_root = 'samples/processed_videos'
            video_bbox_coords, video_signs = obj_det.video_forward(dst_root, local_root)

            # filter signs with video logic
            print("\tExtracting Signs from Video...")
            SignLogic = Sign_Presence(nms_diff=5, padding=10, debug=False)
            filtered_signs = SignLogic.sign_presence_logic(video_signs, video_bbox_coords, dst_root)
            identified_signs += [filtered_signs]
            print(f"\tFiltered Signs: {filtered_signs}", flush=True)

            # damage detection
            if len(filtered_signs) != 0:
                if len(filtered_signs[0]) != 0:
                    print(f"\t\tSign Damage Detection Processing...", flush=True)
                    damage_root = 'samples/normalized_images'
                    damage_classifications = BeIT_damage_detector(damage_root)
                    conformance_statuses.append(damage_classifications)
            else:
                conformance_statuses.append([])

        # adding data to lists
        vehicle_checkpoints.append(vehicle_checkpoint)
        vehicle_checkpoint_signs.append(vehicle_checkpoint.signs)

    print(f"\tIdentified Signs: {identified_signs}")
    print(f"\tConformance Status: {conformance_statuses}")

    # STEP 2.5: COMPARE LOCATED LABELS TO EXPECTED #

    print("\tChecking Conformance Status...")
    print('\t' + ('-' * 30))
    for idx_s, (predicted_signs, conformance_status, vehicle_sign, vehicle_checkpoint) in enumerate(zip(identified_signs,
                                                                                     conformance_statuses,
                                                                                     vehicle_checkpoint_signs,
                                                                                     vehicle_checkpoints)):

        print(f"\t{idx_s}:")

        # updating signs
        new_signs = vehicle_sign

        checkpoint = checkpoints_collection.document(vehicle_checkpoint.checkpointID).get()
        checkpoint = Checkpoint.from_doc(checkpoint)

        checkpoint.conformanceStatus = "processing"

        for index in range(len(conformance_status)):
            conformance_status[index] = conformance_status[index] if (conformance_status[index] == 'conforming') else 'damaged'

        if 'damaged' in conformance_status:
            checkpoint.lastVehicleInspectionResult = "non-conforming"
            new_checkpoint_conformance = "non-conforming"
            vehicle_inspection.conformanceStatus = "non-conforming"
            vehicle.conformanceStatus = "non-conforming"
        else:
            checkpoint.lastVehicleInspectionResult = "conforming"
            new_checkpoint_conformance = "conforming"

        for pos, (signage) in enumerate(zip(vehicle_sign)):
            # checking if inspection sign identified
            if (signage[0]['title'] in predicted_signs):
                # sign identified -> updating status
                sign_index = predicted_signs.index(signage[0]['title'])

                # setting new sign conformance
                new_sign_conformance = conformance_status[sign_index]
                checkpoint.signs[pos]['conformanceStatus'] = new_sign_conformance

                # removing identified sign from list
                print(f"\t\tSign: {predicted_signs[sign_index]}\t-\t{conformance_status[sign_index]}")
                predicted_signs.pop(sign_index)
                conformance_status.pop(sign_index)
            else:
                # setting new sign conformance
                new_sign_conformance = "missing"
                checkpoint.signs[pos]['conformanceStatus'] = new_sign_conformance

                # setting checkpoint conformance
                new_checkpoint_conformance = "non-conforming"
                vehicle_inspection.conformanceStatus = "non-conforming"
                vehicle.conformanceStatus = "non-conforming"

                checkpoint.lastVehicleInspectionResult = "non-conforming"

                print(f"\t\tSign: {signage[0]['title']}\t-\tmissing")

            # updating checkpoints object
            checkpoint.conformanceStatus = checkpoint.lastVehicleInspectionResult
            checkpoint.lastVehicleInspectionID = vehicle_inspection.id
            checkpoint.update(db)

            # updating signs
            new_signs[pos]['conformanceStatus'] = new_sign_conformance

            # STEP 2.6: UPDATE FIREBASE WITH CONFORMANCE AND PROCESSING STATUS FOR INSPECTION #

            # updating inspection checkpoint and checkpoint objects
            vehicle_checkpoint.signs = new_signs
            checkpoint.signs = new_signs
            vehicle_checkpoint.conformanceStatus = new_checkpoint_conformance
            vehicle_checkpoint.update(db)

    print('\t' + ('-' * 30))

    vehicle_inspection.processingStatus = "processed"
    vehicle_inspection.update(db)

    # updating inspection vehicle object
    vehicle.update(db)

    print(f"\tConformance status uploaded.")

    print(f"Processing Complete!")
    print("-----------------------")


"""
    @brief: 
        Performs setup of firebase firestore, storage and the initialization of YOLOv7. 
        Calls the server once the setup is completed.
    
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
    obj_det = ObjectDetector(image_size=1280, conf_thresh=0.6, iou_thresh=0.55, num_classes=36, view_img=False)

    print("Damage Detector Setup...")

    # initialize damage classifier
    BeIT_damage_detector = DamageDetector(model_type='simple')

    print("Setup Complete!")
    print("-----------------------")

    # ############## #
    # RUNNING SERVER #
    # ############## #

    # running server
    runServer()
