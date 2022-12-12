import 'package:auto_sign_mobile/model/enums/capture_type.dart';
import 'package:auto_sign_mobile/model/enums/conformance_status.dart';
import 'package:auto_sign_mobile/model/enums/remediation_status.dart';
import 'package:auto_sign_mobile/model/model_object.dart';
import 'package:auto_sign_mobile/model/vehicle/sign.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

/// A checkpoint within the gold standard walkthrough of a given train vehicle.
class Checkpoint extends ModelObject {
  // MEMBERS //
  String vehicleID; // id of the checkpoints vehicle
  String title; // title of the checkpoint
  String prompt; // prompt shown when capturing the checkpoint
  int index; // index for the checkpoint within the vehicle
  List<Sign> signs; // list of signs in the checkpoint
  CaptureType captureType; // capture type for the checkpoint
  ConformanceStatus conformanceStatus; // current status of the checkpoint
  String lastVehicleInspectionID; // last inspection
  String lastCheckpointInspectionID; // ID of last checkpoint inspection
  ConformanceStatus lastVehicleInspectionResult; // result of last inspection
  RemediationStatus remediationStatus; // remediation status of the checkpoint
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
    List<Sign>? signs,
    CaptureType? captureType,
    ConformanceStatus? conformanceStatus,
    this.lastVehicleInspectionID = "",
    this.lastCheckpointInspectionID = "",
    ConformanceStatus? lastVehicleInspectionResult,
    RemediationStatus? remediationStatus,
    this.lastVehicleRemediationID,
  })  : captureType = captureType ?? CaptureType.photo,
        signs = signs ?? [],
        conformanceStatus = conformanceStatus ?? ConformanceStatus.pending,
        lastVehicleInspectionResult =
            lastVehicleInspectionResult ?? ConformanceStatus.pending,
        remediationStatus = remediationStatus ?? RemediationStatus.none,
        super(id: id, timestamp: timestamp);

  // ///////// //
  // FIRESTORE //
  // ///////// //

  /// Creates a [Checkpoint] object from the provided [DocumentSnapshot].
  factory Checkpoint.fromFirestore(
      DocumentSnapshot<Map<String, dynamic>> snapshot) {
    // getting snapshot data
    final data = snapshot.data();

    // gathering sign data
    List<Sign> signs = [];
    data?["signs"].forEach((sign) {
      signs.add(Sign.fromFirestoreData(sign));
    });

    print(data?["lastCheckpointInspectionID"]);

    // cocnverting document data to an object
    return Checkpoint(
      id: snapshot.id,
      timestamp: data?["timestamp"],
      vehicleID: data?["vehicleID"],
      title: data?["title"],
      prompt: data?["prompt"],
      index: data?["index"],
      captureType: CaptureType.fromString(data?["captureType"]),
      signs: signs,
      conformanceStatus:
          ConformanceStatus.fromString(data?["conformanceStatus"]),
      lastVehicleInspectionID: data?["lastVehicleInspectionID"],
      lastCheckpointInspectionID: data?["lastCheckpointInspectionID"],
      lastVehicleInspectionResult:
          ConformanceStatus.fromString(data?["lastVehicleInspectionResult"]),
      remediationStatus:
          RemediationStatus.fromString(data?["remediationStatus"]),
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
      "captureType": captureType.toString(),
      "signs": [
        for (Sign sign in signs) sign.toFirestore(),
      ],
      "conformanceStatus": conformanceStatus.toString(),
      "lastVehicleInspectionID": lastVehicleInspectionID,
      "lastCheckpointInspectionID": lastCheckpointInspectionID,
      "lastVehicleInspectionResult": lastVehicleInspectionResult.toString(),
      "remediationStatus": remediationStatus.toString(),
      "lastVehicleRemediationID": lastVehicleRemediationID,
    };
  }
}
