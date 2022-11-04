import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:train_vis_mobile/model/model_object.dart';
import 'package:train_vis_mobile/model/status/conformance_status.dart';
import 'package:train_vis_mobile/model/status/processing_status.dart';

/// An inspection of a given train vehicle.
class InspectionWalkthrough extends ModelObject {
  // MEMBERS //
  ProcessingStatus
      processingStatus; // processing status of the inspection walkthrough
  ConformanceStatus conformanceStatus; // conformance status of the walkthrough
  Map<String, ConformanceStatus>
      checkpoints; // map of inspection checkpoint IDs to conformance status

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  InspectionWalkthrough({
    String id = "",
    int? timestamp,
    ProcessingStatus? processingStatus,
    ConformanceStatus? conformanceStatus,
    Map<String, ConformanceStatus>? checkpoints,
  })  : processingStatus = processingStatus ?? ProcessingStatus.pending,
        conformanceStatus = conformanceStatus ?? ConformanceStatus.pending,
        checkpoints = checkpoints ?? {},
        super(id: id, timestamp: timestamp);

  // ///////// //
  // FIRESTORE //
  // ///////// //

  /// Creates a [InspectionWalkthrough] object from the provided [DocumentSnapshot].
  factory InspectionWalkthrough.fromFirestore(
      DocumentSnapshot<Map<String, dynamic>> snapshot) {
    // getting snapshot data
    final data = snapshot.data();

    // cocnverting document data to an [InspectionWalkthrough]
    return InspectionWalkthrough(
      id: snapshot.id,
      timestamp: data?["timestamp"],
      processingStatus: ProcessingStatus.fromString(data?["processingStatus"]),
      conformanceStatus:
          ConformanceStatus.fromString(data?["conformanceStatus"]),
      checkpoints: data?["checkpoints"],
    );
  }

  /// Converts the [InspectionWalkthrough] into a [Map] that can be published to
  /// firestore.
  @override
  Map<String, dynamic> toFirestore() {
    // converting to a map
    return {
      "timestamp": timestamp,
      "processingStatus": processingStatus.toString(),
      "conformanceStatus": conformanceStatus.toString(),
      "checkpoints": checkpoints,
    };
  }
}
