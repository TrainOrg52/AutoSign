import 'dart:io';
import 'dart:typed_data';

import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_storage/firebase_storage.dart';
import 'package:train_vis_mobile/controller/vehicle_controller.dart';
import 'package:train_vis_mobile/model/inspection/checkpoint_inspection.dart';
import 'package:train_vis_mobile/model/inspection/vehicle_inspection.dart';

/// Controller that manages the application's list of [VehicleInspection]
/// objects.
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

  /// TODO
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
  }

  /// TODO
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
}
