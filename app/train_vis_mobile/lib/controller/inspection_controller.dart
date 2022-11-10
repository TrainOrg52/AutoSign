import 'dart:io';
import 'dart:typed_data';

import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_storage/firebase_storage.dart';
import 'package:train_vis_mobile/controller/vehicle_controller.dart';
import 'package:train_vis_mobile/model/inspection/checkpoint_inspection.dart';
import 'package:train_vis_mobile/model/inspection/vehicle_inspection.dart';
import 'package:train_vis_mobile/model/status/processing_status.dart';

/// Controller that manages the application's list of [VehicleInspection]
/// and [CheckpointInspection] objects.
///
/// Providess methods to access the data from Firebase Firestore and Storage as
/// [Stream]s or [Future]s.
class InspectionController {
  // MEMBER VARIABLES //

  // vehicle inspections reference
  final CollectionReference<Map<String, dynamic>> _vehicleInspectionsRef =
      FirebaseFirestore.instance.collection("vehicleInspections");

  // checkpoint inspections reference
  final CollectionReference<Map<String, dynamic>> _checkpointInspectionsRef =
      FirebaseFirestore.instance.collection("checkpointInspections");

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  /// Constructs a new [InspectionController].
  InspectionController._();

  // //////// //
  // INSTANCE //
  // //////// //

  // single instance of controller
  static final InspectionController instance = InspectionController._();

  // /////// //
  // GETTING //
  // /////// //

  /// Returns a [Stream] for the [VehicleInspection] matching the given [id].
  Stream<VehicleInspection> getVehicleInspection(String id) {
    // returning the required vehicle as a stream
    return _vehicleInspectionsRef
        .doc(id)
        .snapshots()
        .map((snapshot) => VehicleInspection.fromFirestore(snapshot));
  }

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

  /// Adds a [VehicleInspection] and its corresponding list of [CheckpointInspection]
  /// objects into the system.
  ///
  /// 1 - The [VehicleInspection] object is pushed to Firestore.
  ///
  /// 2 - The [conformanceStatus] of the [Vehicle] associated with the inspection
  /// is updated to reflect a new inspection has taken place.
  ///
  /// 3 - The list of [CheckpointInspection] objects is iterated over, and for
  /// each checkpoint is added to the system (see [_addCheckpointInspection]).
  ///
  /// 4 - The [processingStatus] of the [VehicleInspection] is updated to [pending]
  /// to reflec that the inspection is ready to be processed.
  Future<void> addVehicleInspection(
    VehicleInspection vehicleInspection,
    List<CheckpointInspection> checkpointInspections,
  ) async {
    // adding the vehicle inspection document to firestore
    await _vehicleInspectionsRef
        .add(vehicleInspection.toFirestore())
        .then((doc) => vehicleInspection.id = doc.id);

    // resetting the conformance status of the vehicle
    VehicleController.instance.resetVehicleConformanceStatus(
      vehicleInspection.vehicleID,
      vehicleInspection.id,
    );

    // iterating over the checkpoint inspections
    for (CheckpointInspection checkpointInspection in checkpointInspections) {
      // setting the vehicle inspection id of the checkpoint inspection
      checkpointInspection.vehicleInspectionID = vehicleInspection.id;

      // adding the CheckpointInspection + capture to firestore
      await InspectionController.instance
          ._addCheckpointInspection(checkpointInspection);

      // updating the information on the checkpoint object
      VehicleController.instance.resetCheckpointConformanceStatus(
        checkpointInspection.checkpointID,
        vehicleInspection.id,
      );
    }

    // updating the proessing status on the vehicle inspection
    await _vehicleInspectionsRef.doc(vehicleInspection.id).update({
      "processingStatus": ProcessingStatus.pending.toString(),
    });
  }

  /// Adds a given [CheckpointInspection] object to the system.
  ///
  /// 1 - The [CheckpointInspection] object is uploaded to Firestore, and it's [id]
  /// property is updated using the [id] of the document that was created.
  ///
  /// 2 - The image associated with the [CheckpointInspection] is uploaded to
  /// Storage under the ID of the Firestore document.
  Future<void> _addCheckpointInspection(
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

  // /////////////// //
  // GETTING OBJECTS //
  // /////////////// //

  /// Returns a [Stream] of the [List] of [VehicleInspection]s associated with
  /// the given [vehicleID].
  ///
  /// This [VehicleInspections] are sorted based on their [timestamp].
  Stream<List<VehicleInspection>> getVehicleInspectionsWhereVehicleIs(
    String vehicleID,
  ) {
    return _checkpointInspectionsRef
        .where("vehicleID", isEqualTo: vehicleID)
        .orderBy("timestamp")
        .snapshots()
        .map(
          (snapshot) => snapshot.docs
              .map((doc) => VehicleInspection.fromFirestore(doc))
              .toList(),
        );
  }

  /// Returns a [Stream] of the [List] of [CheckpointInspection]s associated
  /// with the given [vehicleInspectionID].
  ///
  /// This [CheckpointInspection]s are sorted based on their [index] property.
  Stream<List<CheckpointInspection>>
      getCheckpointInspectionsWhereVehicleInspectionIs(
    String vehicleInspectionID,
  ) {
    return _checkpointInspectionsRef
        .where("vehicleInspectionID", isEqualTo: vehicleInspectionID)
        .orderBy("index")
        .snapshots()
        .map(
          (snapshot) => snapshot.docs
              .map((doc) => CheckpointInspection.fromFirestore(doc))
              .toList(),
        );
  }

  // ////////////// //
  // GETTING IMAGES //
  // ////////////// //

  /// Returns the download URL for the unprocessed image of an
  /// [InspectionCheckpoint].
  Stream<String> getUnprocessedCheckpointInspectionImageDownloadURL(
    String vehicleID,
    String vehicleInspectionID,
    String checkpointInspectionID,
  ) {
    // defining reference to Storage
    Reference reference = FirebaseStorage.instance.ref(
        "$vehicleID/vehicleInspections/$vehicleInspectionID/checkpointInspectionID/unprocessed.png");

    // returning download URL
    return reference.getDownloadURL().asStream();
  }

  /// Returns the download URL for the processed image of an
  /// [InspectionCheckpoint].
  Stream<String> getProcessedCheckpointInspectionImageDownloadURL(
    String vehicleID,
    String vehicleInspectionID,
    String checkpointInspectionID,
  ) {
    // defining reference to Storage
    Reference reference = FirebaseStorage.instance.ref(
        "$vehicleID/vehicleInspections/$vehicleInspectionID/checkpointInspectionID/processed.png");

    // returning download URL
    return reference.getDownloadURL().asStream();
  }
}
