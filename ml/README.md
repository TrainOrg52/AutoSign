# Back-End Processor

## Functionality
- Performs setup configuration for interaction with firebase firestore and storage
- Performs setup configuration for the YOLOv7 object detector
- Gathers newly submitted inspection walkthroughs from firestore for processing
- Processes media from inspection walkthroughs with the YOLOv7 object detector 
- Computes conformance status for each sign and checkpoint within the train associated with an inspection walkthrough
- Updates firestore with conformance and processing status for each sign and checkpoint within the train associated with an inspection walkthrough

## Folder Hierarchy
```
+--ml
|   +-- .configs
|   |   ↪ Configuration files for firebase firestore and storage
|   +-- object_detector
|   |   +-- finetuned_models
|   |   |   ↪ Fine-tuned .pt YOLOv7 model for sign detection (see google drive)
|   |   +-- yolov7
|   |   |   ↪ YOLOv7 dependencies
|   |   +-- inference.py
|   |   |   ↪ Object detector inference script for inspection processing
|   +-- samples
|   |   +-- images
|   |   |   ↪ Firebase storage temporary download source folder
|   |   +-- processed_images
|   |   |   ↪ Processed image folder for Firebase storage upload
|   +-- tools
|   |   +-- media_handling
|   |   |   ↪ Useful media handling (img and video) package
|   |   +-- image_handling_demo.py
|   |   |   ↪ Image handling demo
|   |   +-- video_handling_demo.py
|   |   |   ↪ Video handling demo
|   +-- model.py
|   |   ↪ Firebase firestore object definitions
|   +-- server.py
|   |   ↪ Root script for server and processing
```

## Editing
Code editing should be limited to the following directories as all other directories are static:
- `ml/server.py`
- `ml/model.py`
- `ml/object_detector/inference.py`
- `ml/tools`

Please do not edit the `ml/object_detector/yolov7` directory.