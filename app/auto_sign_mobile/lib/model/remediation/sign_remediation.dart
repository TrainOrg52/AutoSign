import 'package:auto_sign_mobile/model/enums/capture_type.dart';
import 'package:auto_sign_mobile/model/enums/conformance_status.dart';
import 'package:auto_sign_mobile/model/enums/remediation_action.dart';
import 'package:auto_sign_mobile/model/model_object.dart';
import 'package:auto_sign_mobile/model/vehicle/sign.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

/// The remediation of an individual sign within a train.
class SignRemediation extends ModelObject {
  // MEMBERS //
  String signID; // id of sign being remediated
  String title; // the title of the sign being remediated
  String checkpointID; // id of the corresponding checkpoint
  String checkpointTitle; // title of the checkpoint
  CaptureType checkpointCaptureType; // capture type of the checkpoint
  String checkpointInspectionID; // ID of checkpoint inspection being remediated
  String vehicleRemediationID; // ID of the vehicle remediation this is for
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
    this.signID = "",
    this.title = "",
    this.checkpointID = "",
    this.checkpointTitle = "",
    CaptureType? checkpointCaptureType,
    this.checkpointInspectionID = "",
    this.vehicleRemediationID = "",
    ConformanceStatus? preRemediationConformanceStatus,
    RemediationAction? remediationAction,
    this.capturePath = "",
  })  : checkpointCaptureType = checkpointCaptureType ?? CaptureType.photo,
        preRemediationConformanceStatus =
            preRemediationConformanceStatus ?? ConformanceStatus.pending,
        remediationAction = remediationAction ?? RemediationAction.replaced,
        super(id: id, timestamp: timestamp);

  // /////////////// //
  // FROM CHECKPOINT //
  // /////////////// //

  factory SignRemediation.fromSign({
    required Sign sign,
    required String checkpointID,
    required String checkpointTitle,
    required CaptureType checkpointCaptureType,
    required String checkpointInspectionID,
    required RemediationAction remediationAction,
    required String capturePath,
  }) {
    return SignRemediation(
      signID: sign.id,
      title: sign.title,
      checkpointID: checkpointID,
      checkpointTitle: checkpointTitle,
      checkpointCaptureType: checkpointCaptureType,
      checkpointInspectionID: checkpointInspectionID,
      preRemediationConformanceStatus: sign.conformanceStatus,
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

    // creating the object using the data
    return SignRemediation(
      id: snapshot.id,
      timestamp: data?["timestamp"],
      signID: data?["signID"],
      title: data?["title"],
      checkpointID: data?["checkpointID"],
      checkpointTitle: data?["checkpointTitle"],
      checkpointCaptureType:
          CaptureType.fromString(data?["checkpointCaptureType"]),
      checkpointInspectionID: data?["checkpointInspectionID"],
      vehicleRemediationID: data?["vehicleRemediationID"],
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
      "signID": signID,
      "title": title,
      "checkpointID": checkpointID,
      "checkpointTitle": checkpointTitle,
      "checkpointCaptureType": checkpointCaptureType.toString(),
      "checkpointInspectionID": checkpointInspectionID,
      "vehicleRemediationID": vehicleRemediationID,
      "preRemediationConformanceStatus":
          preRemediationConformanceStatus.toString(),
      "remediationAction": remediationAction.toString(),
    };
  }
}
