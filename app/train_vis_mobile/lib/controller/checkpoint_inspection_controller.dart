import 'dart:io';

import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_storage/firebase_storage.dart';
import 'package:flutter/foundation.dart';
import 'package:train_vis_mobile/model/inspection/checkpoint_inspection.dart';

/// Controller that manages the application's list of [CheckpointInspection]
/// objects.
class CheckpointInspectionController {
  // MEMBER VARIABLES //
  // checkpoint inspections reference
  final CollectionReference<Map<String, dynamic>> _checkpointInspectionsRef =
      FirebaseFirestore.instance.collection("checkpointInspections");

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  /// Constructs a new [CheckpointInspectionController].
  CheckpointInspectionController._();

  // //////// //
  // INSTANCE //
  // //////// //

  // single instance of controller
  static final CheckpointInspectionController instance =
      CheckpointInspectionController._();

  // //////////////////// //
  // RETRIEVING AS STREAM //
  // //////////////////// //

  /// Returns a [Stream] for the [CheckpointInspection] matching the given [id].
  Stream<CheckpointInspection> getCheckpointInspection(String id) {
    // returning the required vehicle as a stream
    return _checkpointInspectionsRef
        .doc(id)
        .snapshots()
        .map((snapshot) => CheckpointInspection.fromFirestore(snapshot));
  }

  // ////// //
  // ADDING //
  // ////// //

  /// TODO
  Future<void> addCheckpointInspection(
    CheckpointInspection checkpointInspection,
  ) async {
    // adding the vehicle inspection document to firestore
    await _checkpointInspectionsRef
        .add(checkpointInspection.toFirestore())
        .then((doc) => checkpointInspection.id = doc.id);

    // adding the file to cloud storage
    try {
      // getting the data for the capture
      Uint8List data =
          await File(checkpointInspection.capturePath).readAsBytes();

      // posting the capture to storage
      await FirebaseStorage.instance
          .ref(
              "${checkpointInspection.vehicleID}/vehicleInspections/${checkpointInspection.vehicleInspectionID}/${checkpointInspection.id}/unprocessed.png")
          .putData(data);
    } on FirebaseException catch (e) {
      // handling exception
      print("Upload failed : $e");

      await _checkpointInspectionsRef.doc(checkpointInspection.id).delete();
    }
  }
}
