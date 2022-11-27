import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:train_vis_mobile/model/model_object.dart';
import 'package:train_vis_mobile/model/status/conformance_status.dart';

/// An inspection of a given train vehicle.
class VehicleInspection extends ModelObject {
  // MEMBERS //
  ConformanceStatus conformanceStatus; // conformance status of the walkthrough
  Map<String, ConformanceStatus>
      checkpoints; // map of inspection checkpoint IDs to conformance status

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  VehicleInspection({
    String id = "",
    int? timestamp,
    ConformanceStatus? conformanceStatus,
    Map<String, ConformanceStatus>? checkpoints,
  })  : conformanceStatus = conformanceStatus ?? ConformanceStatus.pending,
        checkpoints = checkpoints ?? {},
        super(id: id, timestamp: timestamp);

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
      "conformanceStatus": conformanceStatus.toString(),
      "checkpoints": checkpoints,
    };
  }
}
