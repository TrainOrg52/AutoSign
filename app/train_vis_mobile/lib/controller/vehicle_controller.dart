import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_storage/firebase_storage.dart';
import 'package:train_vis_mobile/model/vehicle/vehicle.dart';

/// Controller that manages the application's list of [Vehicle] objects.
class VehicleController {
  // MEMBER VARIABLES //
  // reference to Firestore collection
  final CollectionReference<Map<String, dynamic>> _vehiclesRef =
      FirebaseFirestore.instance.collection("vehicles");

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

  // //////////////////// //
  // RETRIEVING AS STREAM //
  // //////////////////// //

  /// Returns a [Stream] for the [Vehicle] matching the given [id].
  Stream<Vehicle> getVehicle(String id) {
    // returning the required vehicle as a stream
    return _vehiclesRef
        .doc(id)
        .snapshots()
        .map((snapshot) => Vehicle.fromFirestore(snapshot));
  }

  // ///////////////////// //
  // RETRIEVING AT INSTANT //
  // ///////////////////// //

  /// Returns a snapshot of [Vehicle] matching the given [id] at the current time.
  Future<Vehicle> getVehicleAtInstant(String id) async {
    // returning s snapshot of the module
    return await _vehiclesRef
        .doc(id)
        .get()
        .then((snapshot) => Vehicle.fromFirestore(snapshot));
  }

  /// Returnss the download URL for the avatar image of the [Vehicle] with the
  /// given [id].
  Future<String> getVehicleAvatarDownloadURL(String id) {
    // defining reference to Storage
    Reference reference = FirebaseStorage.instance.ref("$id/avatar.png");

    // returning download URL
    return reference.getDownloadURL();
  }
}
