import 'package:auto_sign_mobile/model/enums/conformance_status.dart';
import 'package:auto_sign_mobile/model/enums/remediation_action.dart';
import 'package:auto_sign_mobile/model/inspection/checkpoint_inspection.dart';
import 'package:auto_sign_mobile/model/model_object.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

/// The remediation of an individual sign within a train.
class SignRemediation extends ModelObject {
  // MEMBERS //
  String vehicleID; // id of the vehicle being inspected
  String checkpointID; // id of the corresponding checkpoint
  String checkpointInspectionID; // ID of the corresponding inspection
  String vehicleRemediationID; // ID of the vehicle remediation
  String checkpointTitle; // title of the checkpoint
  String signTitle; // the title of the sign being remediated
  ConformanceStatus
      preRemediationConformanceStatus; // conformance status prior to remediation
  RemediationAction remediationAction; // action taken to remediate the sign

  // helper (NOT TO BE SENT TO FIRESTORE)
  String capturePath; // path to the image of the remediation

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  SignRemediation({
    String id = "",
    int? timestamp,
    this.vehicleID = "",
    this.checkpointID = "",
    this.checkpointInspectionID = "",
    this.vehicleRemediationID = "",
    this.checkpointTitle = "",
    this.signTitle = "",
    ConformanceStatus? preRemediationConformanceStatus,
    RemediationAction? remediationAction,
    this.capturePath = "",
  })  : preRemediationConformanceStatus =
            preRemediationConformanceStatus ?? ConformanceStatus.pending,
        remediationAction = remediationAction ?? RemediationAction.replaced,
        super(id: id, timestamp: timestamp);

  // /////////////// //
  // FROM CHECKPOINT //
  // /////////////// //

  factory SignRemediation.fromCheckpointInspection({
    String vehicleRemediationID = "",
    required CheckpointInspection checkpointInspection,
    required String signTitle,
    required RemediationAction remediationAction,
    required String capturePath,
  }) {
    return SignRemediation(
      vehicleID: checkpointInspection.vehicleID,
      checkpointID: checkpointInspection.checkpointID,
      checkpointInspectionID: checkpointInspection.id,
      vehicleRemediationID: vehicleRemediationID,
      checkpointTitle: checkpointInspection.title,
      signTitle: signTitle,
      preRemediationConformanceStatus: checkpointInspection.conformanceStatus,
      remediationAction: remediationAction,
      capturePath: capturePath,
    );
  }

  // ///////// //
  // FIRESTORE //
  // ///////// //

  /// Creates a [SignRemediation] object from the provided [DocumentSnapshot].
  factory SignRemediation.fromFirestore(
      DocumentSnapshot<Map<String, dynamic>> snapshot) {
    // getting snapshot data
    final data = snapshot.data();

    // gathering sign data
    Map<String, ConformanceStatus> signs = {};
    data?["signs"].forEach((sign, conformanceStatus) {
      signs[sign] = ConformanceStatus.fromString(conformanceStatus)!;
    });

    // cocnverting document data to an object
    return SignRemediation(
      id: snapshot.id,
      timestamp: data?["timestamp"],
      vehicleID: data?["vehicleID"],
      checkpointID: data?["checkpointID"],
      checkpointInspectionID: data?["checkpointInspectionID"],
      vehicleRemediationID: data?["vehicleInspectionID"],
      checkpointTitle: data?["title"],
      signTitle: data?["signTitle"],
      preRemediationConformanceStatus: ConformanceStatus.fromString(
          data?["preRemediationConformanceStatus"]),
      remediationAction:
          RemediationAction.fromString(data?["remediationAction"]),
    );
  }

  /// Converts the [Walkthrough] into a [Map] that can be published to firestore.
  @override
  Map<String, dynamic> toFirestore() {
    // converting the object to a map
    return {
      "timestamp": timestamp,
      "vehicleID": vehicleID,
      "checkpointID": checkpointID,
      "checkpointInspectionID": checkpointInspectionID,
      "vehicleRemediationID": vehicleRemediationID,
      "checkpointTitle": checkpointTitle,
      "signTitle": signTitle,
      "preRemediationConformanceStatus":
          preRemediationConformanceStatus.toString(),
      "remediationAction": remediationAction,
    };
  }
}
