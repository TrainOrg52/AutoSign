import 'package:auto_sign_mobile/model/enums/conformance_status.dart';
import 'package:auto_sign_mobile/model/vehicle/checkpoint.dart';
import 'package:auto_sign_mobile/model/vehicle/vehicle.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_storage/firebase_storage.dart';

/// Controller that manages the application's list of [Vehicle] and
/// [CheckpointInspection] objects.
///
/// Providess methods to access the data from Firebase Firestore and Storage as
/// [Stream]s or [Future]s.
class VehicleController {
  // MEMBER VARIABLES //

  // reference to vehicles Firestore collection
  final CollectionReference<Map<String, dynamic>> _vehiclesRef =
      FirebaseFirestore.instance.collection("vehicles");

  // reference to checkpoints Firestore collection
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

  // /////////////// //
  // GETTING OBJECTS //
  // /////////////// //

  /// Returns a [Stream] for the [Vehicle] matching the given [vehicleID].
  Stream<Vehicle> getVehicle(String vehicleID) {
    // returning the required vehicle as a stream
    return _vehiclesRef
        .doc(vehicleID)
        .snapshots()
        .map((snapshot) => Vehicle.fromFirestore(snapshot));
  }

  /// Returns the [Vehicle] matching the given [VehicleID].
  Future<Vehicle> getVehicleAtInstant(String vehicleID) async {
    return Vehicle.fromFirestore(await _vehiclesRef.doc(vehicleID).get());
  }

  /// Returns a [Stream] of the [Checkpoint] with the given [checkpointID].
  Stream<Checkpoint> getCheckpoint(String checkpointID) {
    // returning the required checkpoint as a stream
    return _checkpointsRef
        .doc(checkpointID)
        .snapshots()
        .map((snapshot) => Checkpoint.fromFirestore(snapshot));
  }

  /// Returns a [Stream] for the [List] of [Checkpoint]s associated with the
  /// given [vehcileID].
  ///
  /// The [Checkpoint]s are sorted based on their [index] property.
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
  Stream<String> getCheckpointImageDownloadURL(
    String vehicleID,
    String checkpointID,
  ) {
    // defining reference to Storage
    Reference reference = FirebaseStorage.instance
        .ref("$vehicleID/checkpoints/$checkpointID.png");

    // returning download URL
    return reference.getDownloadURL().asStream();
  }

  // //////////////////////////// //
  // RESETTING CONFORMANCE STATUS //
  // //////////////////////////// //

  /// Resets the conformance status of the given vehicle following an inspection.
  ///
  /// 1 - Sets the [conformanceStatus] of the vehicle to [pending].
  ///
  /// 2 - Sets the [lastVehicleInspectionID] to the given [lastVehicleInspectionID].
  Future<void> resetVehicleConformanceStatus(
    String vehicleID,
    String lastVehicleInspectionID,
  ) async {
    // updating vehicle properties
    await _vehiclesRef.doc(vehicleID).update(
      {
        "conformanceStatus": ConformanceStatus.pending.toString(),
        "lastVehicleInspectionID": lastVehicleInspectionID,
      },
    );
  }

  /// Resets the conformance status of the given [Checkpoint] following an
  /// inspection.
  ///
  /// 1 - Sets the [conformanceStatus] of the checkpoint to [pending].
  ///
  /// 2 - Sets the [lastVehicleInspectionID] to the given [lastVehicleInspectionID].
  ///
  /// 3 - Sets the [lastVehicleInspectionResult] to the [pending].
  ///
  /// 4 - Sets the [lastVehicleRemediationID] to null.
  Future<void> resetCheckpointConformanceStatus(
    String checkpointID,
    String lastVehicleInspectionID,
  ) async {
    // updating checkpoint properties
    await _checkpointsRef.doc(checkpointID).update(
      {
        "conformanceStatus": ConformanceStatus.pending.toString(),
        "lastVehicleInspectionID": lastVehicleInspectionID,
        "lastVehicleInspectionResult": ConformanceStatus.pending.toString(),
        "lastVehicleRemediationID": "",
      },
    );
  }
}
