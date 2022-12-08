import 'package:auto_sign_mobile/model/enums/capture_type.dart';
import 'package:auto_sign_mobile/model/enums/conformance_status.dart';
import 'package:auto_sign_mobile/model/model_object.dart';
import 'package:auto_sign_mobile/model/vehicle/checkpoint.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

/// A checkpoint within the inspection walkthrough of a given train vehicle.
class CheckpointInspection extends ModelObject {
  // MEMBERS //
  String vehicleID; // id of the vehicle being inspected
  String vehicleInspectionID; // id of the inspection
  String checkpointID; // id of the corresponding checkpoint
  String title; // title of the checkpoint
  int index; // index o f checkpoint in vehicle
  ConformanceStatus conformanceStatus; // conformance status of checkpoint
  List<Map<String, ConformanceStatus>>
      signs; // map of signs to conformance status
  CaptureType captureType; // type of media captured in inspection

  // helper (NOT TO BE SENT TO FIRESTORE)
  String capturePath;

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  CheckpointInspection({
    String id = "",
    int? timestamp,
    this.vehicleID = "",
    this.vehicleInspectionID = "",
    this.checkpointID = "",
    this.title = "",
    this.index = 0,
    CaptureType? captureType,
    ConformanceStatus? conformanceStatus,
    List<Map<String, ConformanceStatus>>? signs,
    this.capturePath = "",
  })  : captureType = captureType ?? CaptureType.photo,
        conformanceStatus = conformanceStatus ?? ConformanceStatus.pending,
        signs = signs ?? [],
        super(id: id, timestamp: timestamp);

  // /////////////// //
  // FROM CHECKPOINT //
  // /////////////// //

  factory CheckpointInspection.fromCheckpoint({
    String vehicleInspectionID = "",
    required Checkpoint checkpoint,
    required String capturePath,
  }) {
    // List<Map<String, ConformanceStatus>> signs = [];
    // for (String signID in checkpoint.signs) {
    //   signs.add({signID: ConformanceStatus.pending});
    // }

    // print(signs);

    return CheckpointInspection(
      vehicleID: checkpoint.vehicleID,
      vehicleInspectionID: vehicleInspectionID,
      checkpointID: checkpoint.id,
      title: checkpoint.title,
      index: checkpoint.index,
      captureType: checkpoint.captureType,
      conformanceStatus: ConformanceStatus.pending,
      signs: [
        for (Map<String, ConformanceStatus> sign in checkpoint.signs)
          {sign.entries.first.key: ConformanceStatus.pending}
      ],
      capturePath: capturePath,
    );
  }

  // ///////// //
  // FIRESTORE //
  // ///////// //

  /// Creates a [CheckpointInspection] object from the provided [DocumentSnapshot].
  factory CheckpointInspection.fromFirestore(
      DocumentSnapshot<Map<String, dynamic>> snapshot) {
    // getting snapshot data
    final data = snapshot.data();

    // gathering sign data
    List<Map<String, ConformanceStatus>> signs = [];
    data?["signs"].forEach((sign) {
      signs.add({
        sign.entries.first.key:
            ConformanceStatus.fromString(sign.entries.first.value)!
      });
    });

    // cocnverting document data to an object
    return CheckpointInspection(
      id: snapshot.id,
      timestamp: data?["timestamp"],
      vehicleID: data?["vehicleID"],
      vehicleInspectionID: data?["vehicleInspectionID"],
      checkpointID: data?["checkpointID"],
      title: data?["title"],
      index: data?["index"],
      captureType: CaptureType.fromString(data?["captureType"]),
      conformanceStatus:
          ConformanceStatus.fromString(data?["conformanceStatus"]),
      signs: signs,
    );
  }

  /// Converts the [Walkthrough] into a [Map] that can be published to firestore.
  @override
  Map<String, dynamic> toFirestore() {
    // creating firebase object for signs
    List<Map<String, String>> signsToPost = [];
    for (var sign in signs) {
      signsToPost
          .add({sign.entries.first.key: sign.entries.first.value.toString()});
    }

    // converting the object to a map
    return {
      "timestamp": timestamp,
      "vehicleID": vehicleID,
      "vehicleInspectionID": vehicleInspectionID,
      "checkpointID": checkpointID,
      "title": title,
      "index": index,
      "captureType": captureType.toString(),
      "conformanceStatus": conformanceStatus.toString(),
      "signs": signsToPost,
    };
  }
}
