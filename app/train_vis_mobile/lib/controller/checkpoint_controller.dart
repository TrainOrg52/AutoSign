import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_storage/firebase_storage.dart';
import 'package:train_vis_mobile/model/vehicle/checkpoint.dart';

/// Controller that manages the application's list of [Checkpoint] objects.
class CheckpointController {
  // MEMBER VARIABLES //
  // reference to Firestore collection
  final CollectionReference<Map<String, dynamic>> _checkpointsRef =
      FirebaseFirestore.instance.collection("checkpoints");

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  /// Constructs a new [CheckpointController].
  CheckpointController._();

  // //////// //
  // INSTANCE //
  // //////// //

  // single instance of controller
  static final CheckpointController instance = CheckpointController._();

  // //////////////////// //
  // RETRIEVING AS STREAM //
  // //////////////////// //

  /// Returns a [Stream] of the [Checkpoint] with the given [id].
  Stream<Checkpoint> getCheckpoint(String id) {
    // returning the required checkpoint as a stream
    return _checkpointsRef
        .doc(id)
        .snapshots()
        .map((snapshot) => Checkpoint.fromFirestore(snapshot));
  }

  // ///////////////////// //
  // RETREIVING AT INSTANT //
  // ///////////////////// //

  /// Returns the download URL for the image of the [Checkpoint].
  Future<String> getCheckpointImageDownloadURL(
    String vehicleID,
    String checkpointID,
  ) {
    // defining reference to Storage
    Reference reference = FirebaseStorage.instance
        .ref("$vehicleID/checkpoints/$checkpointID.png");

    // returning download URL
    return reference.getDownloadURL();
  }
}
