import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_storage/firebase_storage.dart';
import 'package:train_vis_mobile/model/status/conformance_status.dart';
import 'package:train_vis_mobile/model/vehicle/checkpoint.dart';
import 'package:train_vis_mobile/model/vehicle/vehicle.dart';

/// Controller that manages the application's list of [Vehicle] objects.
class VehicleController {
  // MEMBER VARIABLES //
  // reference to Firestore collection
  final CollectionReference<Map<String, dynamic>> _vehiclesRef =
      FirebaseFirestore.instance.collection("vehicles");
  final CollectionReference<Map<String, dynamic>> _checkpointsRef =
      FirebaseFirestore.instance.collection("checkpoints");

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  /// Constructs a new [VehicleController].
  VehicleController._();

  // //////// //
  // INSTANCE //
  // //////// //

  // single instance of controller
  static final VehicleController instance = VehicleController._();

  // /////// //
  // GETTING //
  // /////// //

  /// Returns a [Stream] for the [Vehicle] matching the given [vehicleID].
  Stream<Vehicle> getVehicle(String vehicleID) {
    // returning the required vehicle as a stream
    return _vehiclesRef
        .doc(vehicleID)
        .snapshots()
        .map((snapshot) => Vehicle.fromFirestore(snapshot));
  }

  /// Returns a [Stream] of the [Checkpoint] with the given [checkpointID].
  Stream<Checkpoint> getCheckpoint(String checkpointID) {
    // returning the required checkpoint as a stream
    return _checkpointsRef
        .doc(checkpointID)
        .snapshots()
        .map((snapshot) => Checkpoint.fromFirestore(snapshot));
  }

  /// TODO
  Stream<List<Checkpoint>> getCheckpointsWhereVehicleIs(String vehicleID) {
    return _checkpointsRef
        .where("vehicleID", isEqualTo: vehicleID)
        .orderBy("index")
        .snapshots()
        .map(
          (snapshot) => snapshot.docs
              .map((doc) => Checkpoint.fromFirestore(doc))
              .toList(),
        );
  }

  // ////////////// //
  // GETTING IMAGES //
  // ////////////// //

  /// Returnss the download URL for the avatar image of the [Vehicle] with the
  /// given [vehicleID].
  Future<String> getVehicleAvatarDownloadURL(String vehicleID) {
    // defining reference to Storage
    Reference reference = FirebaseStorage.instance.ref("$vehicleID/avatar.png");

    // returning download URL
    return reference.getDownloadURL();
  }

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

  // //////////////////////////// //
  // RESETTING CONFORMANCE STATUS //
  // //////////////////////////// //

  /// TODO
  Future<void> resetVehicleConformanceStatus(
    String vehicleID,
  ) async {
    // updating vehicle properties
    await _vehiclesRef.doc(vehicleID).update(
      {
        "conformanceStatus": ConformanceStatus.pending.toString(),
        "lastVehicleInspectionID": "",
      },
    );
  }

  /// TODO
  Future<void> resetCheckpointConformanceStatus(
    String vehicleID,
  ) async {
    // updating checkpoint properties
    await _checkpointsRef.doc(vehicleID).update(
      {
        "conformanceStatus": ConformanceStatus.pending.toString(),
        "lastVehicleInspectionID": "",
        "lastVehicleInspectionResult": "",
        "lastVehicleRemediationID": "",
      },
    );
  }
}
