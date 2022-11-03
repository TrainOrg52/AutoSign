import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:train_vis_mobile/model/ModelObject.dart';
import 'package:train_vis_mobile/model/status/conformance_status.dart';

/// A checkpoint within the inspection walkthrough of a given train vehicle.
class InspectionCheckpoint extends ModelObject {
  // MEMBERS //
  String inspectionWalkthroughID; // id of the inspection
  String checkpointID; // id of the corresponding checkpoint
  String title; // title of the checkpoint
  Map<String, ConformanceStatus> signs; // map of signs to conformance statuss

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  InspectionCheckpoint({
    String id = "",
    int? timestamp,
    this.inspectionWalkthroughID = "",
    this.checkpointID = "",
    this.title = "",
    Map<String, ConformanceStatus>? signs,
  })  : signs = signs ?? {},
        super(id: id, timestamp: timestamp);

  // ///////// //
  // FIRESTORE //
  // ///////// //

  /// Creates a [InspectionCheckpoint] object from the provided [DocumentSnapshot].
  factory InspectionCheckpoint.fromFirestore(
      DocumentSnapshot<Map<String, dynamic>> snapshot) {
    // getting snapshot data
    final data = snapshot.data();

    // cocnverting document data to an [Checkpoint]
    return InspectionCheckpoint(
      id: snapshot.id,
      timestamp: data?["timestamp"],
      inspectionWalkthroughID: data?["inspectionWalkthroughID"],
      checkpointID: data?["checkpointID"],
      title: data?["title"],
      signs: data?["signs"],
    );
  }

  /// Converts the [Walkthrough] into a [Map] that can be published to firestore.
  @override
  Map<String, dynamic> toFirestore() {
    // converting the walkthrough into a map
    return {
      "timestamp": timestamp,
      "inspectionWalkthroughID": inspectionWalkthroughID,
      "checkpointID": checkpointID,
      "title": title,
      "signs": signs,
    };
  }
}
