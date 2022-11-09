import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:train_vis_mobile/controller/checkpoint_inspection_controller.dart';
import 'package:train_vis_mobile/model/inspection/checkpoint_inspection.dart';
import 'package:train_vis_mobile/model/inspection/vehicle_inspection.dart';

/// Controller that manages the application's list of [VehicleInspection]
/// objects.
class VehicleInspectionController {
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

  /// Constructs a new [VehicleInspectionController].
  VehicleInspectionController._();

  // //////// //
  // INSTANCE //
  // //////// //

  // single instance of controller
  static final VehicleInspectionController instance =
      VehicleInspectionController._();

  // //////////////////// //
  // RETRIEVING AS STREAM //
  // //////////////////// //

  /// Returns a [Stream] for the [VehicleInspection] matching the given [id].
  Stream<VehicleInspection> getVehicleInspection(String id) {
    // returning the required vehicle as a stream
    return _vehicleInspectionsRef
        .doc(id)
        .snapshots()
        .map((snapshot) => VehicleInspection.fromFirestore(snapshot));
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

    // updating the conformance status of the vehicle
    // TODO

    // iterating over the checkpoint inspections
    for (CheckpointInspection checkpointInspection in checkpointInspections) {
      // setting the vehicle inspection id of the checkpoint inspection
      checkpointInspection.vehicleInspectionID = vehicleInspection.id;

      // adding the CheckpointInspection + capture to firestore
      await CheckpointInspectionController.instance
          .addCheckpointInspection(checkpointInspection);

      // updating the information on the checkpoint object
      // TODO
    }
  }
}
