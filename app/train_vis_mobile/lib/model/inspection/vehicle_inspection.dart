import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:train_vis_mobile/model/model_object.dart';
import 'package:train_vis_mobile/model/status/conformance_status.dart';
import 'package:train_vis_mobile/model/status/processing_status.dart';
import 'package:train_vis_mobile/model/vehicle/vehicle.dart';

/// An inspection of a given train vehicle.
class VehicleInspection extends ModelObject {
  // MEMBERS //
  String vehicleID; // ID of vehicle being inspected
  ProcessingStatus processingStatus; // processing status of inspection
  ConformanceStatus conformanceStatus; // conformance status of inspection

  Map<String, ConformanceStatus>
      checkpoints; // map of inspection checkpoint IDs to conformance status

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  VehicleInspection({
    String id = "",
    int? timestamp,
    this.vehicleID = "",
    ProcessingStatus? processingStatus,
    ConformanceStatus? conformanceStatus,
    Map<String, ConformanceStatus>? checkpoints,
  })  : processingStatus = processingStatus ?? ProcessingStatus.pending,
        conformanceStatus = conformanceStatus ?? ConformanceStatus.pending,
        checkpoints = checkpoints ?? {},
        super(id: id, timestamp: timestamp);

  // //////////// //
  // FROM VEHICLE //
  // //////////// //

  factory VehicleInspection.fromVehicle({
    required Vehicle vehicle,
  }) {
    return VehicleInspection(
      vehicleID: vehicle.id,
      processingStatus: ProcessingStatus.pending,
      conformanceStatus: ConformanceStatus.pending,
      checkpoints: {
        for (String checkpointID in vehicle.checkpoints)
          checkpointID: ConformanceStatus.pending
      },
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

    // cocnverting document data to an [InspectionWalkthrough]
    return VehicleInspection(
      id: snapshot.id,
      timestamp: data?["timestamp"],
      vehicleID: data?["vehicleID"],
      processingStatus: ProcessingStatus.fromString(data?["processingStatus"]),
      conformanceStatus:
          ConformanceStatus.fromString(data?["conformanceStatus"]),
      checkpoints: data?["checkpoints"],
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
      "processingStatus": processingStatus.toString(),
      "conformanceStatus": conformanceStatus.toString(),
      "checkpoints": checkpoints,
    };
  }
}
