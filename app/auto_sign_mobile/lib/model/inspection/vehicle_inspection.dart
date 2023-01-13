import 'package:auto_sign_mobile/model/enums/conformance_status.dart';
import 'package:auto_sign_mobile/model/enums/processing_status.dart';
import 'package:auto_sign_mobile/model/model_object.dart';
import 'package:auto_sign_mobile/model/vehicle/vehicle.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

/// An inspection of a given train vehicle.
class VehicleInspection extends ModelObject {
  // MEMBERS //
  String vehicleID; // ID of vehicle being inspected
  String location; // location of the inspection
  ProcessingStatus processingStatus; // processing status of inspection
  ConformanceStatus conformanceStatus; // conformance status of inspection

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  VehicleInspection({
    String id = "",
    int? timestamp,
    this.vehicleID = "",
    this.location = "Newport",
    ProcessingStatus? processingStatus,
    ConformanceStatus? conformanceStatus,
  })  : processingStatus = processingStatus ?? ProcessingStatus.uploading,
        conformanceStatus = conformanceStatus ?? ConformanceStatus.pending,
        super(id: id, timestamp: timestamp);

  // //////////// //
  // FROM VEHICLE //
  // //////////// //

  factory VehicleInspection.fromVehicle({
    required Vehicle vehicle,
  }) {
    return VehicleInspection(
      vehicleID: vehicle.id,
      location: vehicle.location,
    );
  }

  // ///////// //
  // FIRESTORE //
  // ///////// //

  /// Creates a [VehicleInspection] object from the provided [DocumentSnapshot].
  factory VehicleInspection.fromFirestore(
      DocumentSnapshot<Map<String, dynamic>> snapshot) {
    // getting snapshot data
    final data = snapshot.data();

    // converting the document to an object
    return VehicleInspection(
      id: snapshot.id,
      timestamp: data?["timestamp"],
      vehicleID: data?["vehicleID"],
      location: data?["location"],
      processingStatus: ProcessingStatus.fromString(data?["processingStatus"]),
      conformanceStatus:
          ConformanceStatus.fromString(data?["conformanceStatus"]),
    );
  }

  /// Converts the [VehicleInspection] into a [Map] that can be published to
  /// firestore.
  @override
  Map<String, dynamic> toFirestore() {
    // converting to a map
    return {
      "timestamp": timestamp,
      "vehicleID": vehicleID,
      "location": location,
      "processingStatus": processingStatus.toString(),
      "conformanceStatus": conformanceStatus.toString(),
    };
  }
}
