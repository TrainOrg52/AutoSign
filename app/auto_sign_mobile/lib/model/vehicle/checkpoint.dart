import 'package:auto_sign_mobile/model/model_object.dart';
import 'package:auto_sign_mobile/model/status/conformance_status.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

/// A checkpoint within the gold standard walkthrough of a given train vehicle.
class Checkpoint extends ModelObject {
  // MEMBERS //
  String vehicleID; // id of the checkpoints vehicle
  String title; // title of the checkpoint
  String prompt; // prompt shown when capturing the checkpoint
  int index; // index for the checkpoint within the vehicle
  List<String> signs; // list of signs expected within the checkpoint
  ConformanceStatus conformanceStatus; // current status of the checkpoint
  String lastVehicleInspectionID; // last inspection
  ConformanceStatus lastVehicleInspectionResult; // result of last inspection
  String? lastVehicleRemediationID; // last remediation (if exists)

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  Checkpoint({
    String id = "",
    int? timestamp,
    this.vehicleID = "",
    this.title = "",
    this.prompt = "",
    this.index = 0,
    List<String>? signs,
    ConformanceStatus? conformanceStatus,
    this.lastVehicleInspectionID = "",
    ConformanceStatus? lastVehicleInspectionResult,
    this.lastVehicleRemediationID,
  })  : signs = signs ?? [],
        conformanceStatus = conformanceStatus ?? ConformanceStatus.pending,
        lastVehicleInspectionResult =
            lastVehicleInspectionResult ?? ConformanceStatus.pending,
        super(id: id, timestamp: timestamp);

  // ///////// //
  // FIRESTORE //
  // ///////// //

  /// Creates a [Checkpoint] object from the provided [DocumentSnapshot].
  factory Checkpoint.fromFirestore(
      DocumentSnapshot<Map<String, dynamic>> snapshot) {
    // getting snapshot data
    final data = snapshot.data();

    // cocnverting document data to an object
    return Checkpoint(
      id: snapshot.id,
      timestamp: data?["timestamp"],
      vehicleID: data?["vehicleID"],
      title: data?["title"],
      prompt: data?["prompt"],
      index: data?["index"],
      signs: List.from(data?["signs"]),
      conformanceStatus:
          ConformanceStatus.fromString(data?["conformanceStatus"]),
      lastVehicleInspectionID: data?["lastVehicleInspectionID"],
      lastVehicleInspectionResult:
          ConformanceStatus.fromString(data?["lastVehicleInspectionResult"]),
      lastVehicleRemediationID: data?["lastVehicleRemediationID"] == "null"
          ? null
          : data?["lastVehicleRemediationID"],
    );
  }

  /// Converts the [Walkthrough] into a [Map] that can be published to firestore.
  @override
  Map<String, dynamic> toFirestore() {
    // converting object to a map
    return {
      "timestamp": timestamp,
      "vehicleID": vehicleID,
      "title": title,
      "prompt": prompt,
      "index": index,
      "signs": signs,
      "conformanceStatus": conformanceStatus.toString(),
      "lastVehicleInspectionID": lastVehicleInspectionID,
      "lastVehicleInspectionResult": lastVehicleInspectionResult,
      "lastVehicleRemediationID": lastVehicleRemediationID,
    };
  }
}
