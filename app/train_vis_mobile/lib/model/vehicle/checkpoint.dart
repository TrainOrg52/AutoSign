import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:train_vis_mobile/model/model_object.dart';
import 'package:train_vis_mobile/model/status/conformance_status.dart';

/// A checkpoint within the gold standard walkthrough of a given train vehicle.
class Checkpoint extends ModelObject {
  // MEMBERS //
  String vehicleID; // id of this checkpoints associated vehicle
  String title; // title of the checkpoint
  String prompt; // prompt shown when capturing the checkpoint
  List<String> signs; // list of signs expected within the checkpoint
  ConformanceStatus conformanceStatus; // current conformance status of CP
  String
      mostRecentInspectionWalkthroughID; // most recent inspection walkthrough
  ConformanceStatus
      mostRecentInspectionWalkthroughResult; // ID of most recent inspection walkthrough
  String?
      mostRecentRemediationWalkthroughID; // ID of most recent remediation (if there is one)

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  Checkpoint({
    String id = "",
    int? timestamp,
    this.vehicleID = "",
    this.title = "",
    this.prompt = "",
    List<String>? signs,
    ConformanceStatus? conformanceStatus,
    this.mostRecentInspectionWalkthroughID = "",
    ConformanceStatus? mostRecentInspectionWalkthroughResult,
    this.mostRecentRemediationWalkthroughID,
  })  : signs = signs ?? [],
        conformanceStatus = conformanceStatus ?? ConformanceStatus.pending,
        mostRecentInspectionWalkthroughResult =
            mostRecentInspectionWalkthroughResult ?? ConformanceStatus.pending,
        super(id: id, timestamp: timestamp);

  // ///////// //
  // FIRESTORE //
  // ///////// //

  /// Creates a [Checkpoint] object from the provided [DocumentSnapshot].
  factory Checkpoint.fromFirestore(
      DocumentSnapshot<Map<String, dynamic>> snapshot) {
    // getting snapshot data
    final data = snapshot.data();

    // cocnverting document data to an [Checkpoint]
    return Checkpoint(
      id: snapshot.id,
      timestamp: data?["timestamp"],
      vehicleID: data?["vehicleID"],
      title: data?["title"],
      prompt: data?["prompt"],
      signs: List.from(data?["signs"]),
      conformanceStatus:
          ConformanceStatus.fromString(data?["conformanceStatus"]),
      mostRecentInspectionWalkthroughID:
          data?["mostRecentInspectionWalkthroughID"],
      mostRecentInspectionWalkthroughResult: ConformanceStatus.fromString(
          data?["mostRecentInspectionWalkthroughResult"]),
      mostRecentRemediationWalkthroughID:
          data?["mostRecentRemediationWalkthroughID"] == "null"
              ? null
              : data?["mostRecentRemediationWalkthroughID"],
    );
  }

  /// Converts the [Walkthrough] into a [Map] that can be published to firestore.
  @override
  Map<String, dynamic> toFirestore() {
    // converting to a map
    return {
      "timestamp": timestamp,
      "vehicleID": vehicleID,
      "title": title,
      "prompt": prompt,
      "signs": signs,
      "conformanceStatus": conformanceStatus.toString(),
      "mostRecentInspectionWalkthroughID": mostRecentInspectionWalkthroughID,
      "mostRecentInspectionWalkthroughResult":
          mostRecentInspectionWalkthroughResult,
      "mostRecentRemediationWalkthroughID": mostRecentRemediationWalkthroughID,
    };
  }
}
