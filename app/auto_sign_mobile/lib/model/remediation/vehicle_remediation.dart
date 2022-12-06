import 'package:auto_sign_mobile/model/inspection/vehicle_inspection.dart';
import 'package:auto_sign_mobile/model/model_object.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

/// A remediation of a given train vehicle on a given day.
class VehicleRemediation extends ModelObject {
  // MEMBERS //
  String vehicleID; // ID of vehicle being remediated
  String location; // location of the inspection
  String vehicleInspectionID; // ID of vehicle inspection being remediated

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  VehicleRemediation({
    String id = "",
    int? timestamp,
    this.vehicleID = "",
    this.location = "",
    this.vehicleInspectionID = "",
  }) : super(id: id, timestamp: timestamp);

  // //////////// //
  // FROM VEHICLE //
  // //////////// //

  factory VehicleRemediation.fromVehicleInspection({
    required VehicleInspection vehicleInspection,
  }) {
    return VehicleRemediation(
      vehicleID: vehicleInspection.vehicleID,
      location: vehicleInspection.location,
      vehicleInspectionID: vehicleInspection.id,
    );
  }

  // ///////// //
  // FIRESTORE //
  // ///////// //

  /// Creates a [VehicleRemediation] object from the provided [DocumentSnapshot].
  factory VehicleRemediation.fromFirestore(
      DocumentSnapshot<Map<String, dynamic>> snapshot) {
    // getting snapshot data
    final data = snapshot.data();

    // converting the document to an object
    return VehicleRemediation(
      id: snapshot.id,
      timestamp: data?["timestamp"],
      vehicleID: data?["vehicleID"],
      location: data?["location"],
      vehicleInspectionID: data?["vehicleInspectionID"],
    );
  }

  /// Converts the [VehicleRemediation] into a [Map] that can be published to
  /// firestore.
  @override
  Map<String, dynamic> toFirestore() {
    // converting to a map
    return {
      "timestamp": timestamp,
      "vehicleID": vehicleID,
      "location": location,
      "vehicleInspectionID": vehicleInspectionID,
    };
  }
}
