import 'dart:io';
import 'dart:typed_data';

import 'package:auto_sign_mobile/controller/vehicle_controller.dart';
import 'package:auto_sign_mobile/main.dart';
import 'package:auto_sign_mobile/model/remediation/sign_remediation.dart';
import 'package:auto_sign_mobile/model/remediation/vehicle_remediation.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_storage/firebase_storage.dart';

/// TODO
class RemediationController {
  // MEMBER VARIABLES //

  // vehicle remediations reference
  final CollectionReference<Map<String, dynamic>> _vehicleRemediationsRef =
      FirebaseFirestore.instance.collection("vehicleRemediations");

  // sign remediations reference
  final CollectionReference<Map<String, dynamic>> _signRemediationsRef =
      FirebaseFirestore.instance.collection("signRemediations");

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  /// Constructs a new [RemediationController].
  RemediationController._();

  // //////// //
  // INSTANCE //
  // //////// //

  // single instance of controller
  static final RemediationController instance = RemediationController._();

  // /////// //
  // GETTING //
  // /////// //

  /// Returns a [Stream] for the [VehicleRemediation] matching the given [id].
  Stream<VehicleRemediation> getVehicleRemediation(String id) {
    // returning the required vehicle as a stream
    return _vehicleRemediationsRef
        .doc(id)
        .snapshots()
        .map((snapshot) => VehicleRemediation.fromFirestore(snapshot));
  }

  /// Returns a [Stream] for the [SignRemediation] matching the given [id].
  Stream<SignRemediation> getSignRemediation(String id) {
    // returning the required vehicle as a stream
    return _signRemediationsRef
        .doc(id)
        .snapshots()
        .map((snapshot) => SignRemediation.fromFirestore(snapshot));
  }

  // ////// //
  // ADDING //
  // ////// //

  /// TODO
  Future<void> addSignRemediation(
    SignRemediation signRemediation,
    String vehicleID,
    String vehicleInspectionID,
  ) async {
    // getting the vehicle remediation for the sign remediation
    VehicleRemediation vehicleRemediation =
        await _getVehicleRemediationForSignRemediation(
            signRemediation, vehicleID, vehicleInspectionID);

    // updating the id of the sign remediation
    signRemediation.vehicleRemediationID = vehicleRemediation.id;

    // adding the sign remediation document to firestore
    await _signRemediationsRef
        .add(signRemediation.toFirestore())
        .then((doc) => signRemediation.id = doc.id);

    // adding the file to cloud storage
    try {
      // variable to hold capture path
      String capturePath = signRemediation.capturePath;

      // getting the data for the capture
      Uint8List data = await File(capturePath).readAsBytes();

      // posting the capture to storage
      await FirebaseStorage.instance
          .ref(
              "${vehicleRemediation.vehicleID}/vehicleRemediations/${vehicleRemediation.id}/${signRemediation.id}.png")
          .putData(data);

      // updating checkpoint
      await VehicleController.instance.remediateCheckpointSign(signRemediation);
    } on FirebaseException catch (e) {
      // handling exception
      print("Upload failed : $e");

      await _signRemediationsRef.doc(signRemediation.id).delete();
    }
  }

  /// TODO
  Future<VehicleRemediation> _getVehicleRemediationForSignRemediation(
    SignRemediation signRemediation,
    String vehicleID,
    String vehicleInspectionID,
  ) async {
    // variable to store vehicle remediation
    VehicleRemediation vehicleRemediation;

    // getting most recent vehicle remediation
    VehicleRemediation? mostRecentVehicleRemediation =
        await _vehicleRemediationsRef
            .orderBy("timestamp", descending: true)
            .get()
            .then((snapshot) {
      if (snapshot.docs.isEmpty) {
        return null;
      } else {
        return VehicleRemediation.fromFirestore(snapshot.docs.first);
      }
    });

    // checking if the vehicle remediation is null
    if (mostRecentVehicleRemediation == null) {
      // most recent vehicle remediation is null -> need to create a new one

      // creating new vehicle remediation
      vehicleRemediation = VehicleRemediation(
        vehicleID: vehicleID,
        vehicleInspectionID: vehicleInspectionID,
      );

      // posting the vehicle remediation to firestore
      await _vehicleRemediationsRef
          .add(vehicleRemediation.toFirestore())
          .then((doc) => vehicleRemediation.id = doc.id);
    } else {
      // most recent vehicle remediation is not null -> need to check if it is today

      // checking timestamp of remediation against current date
      if (mostRecentVehicleRemediation.timestamp.isToday()) {
        // vehicle remediation was today -> this is the vehicle remediation

        // updating vehicle remediation ID of sign remediation
        vehicleRemediation = mostRecentVehicleRemediation;
      } else {
        // vehicle remediation was not today -> need to create new one for today

        // creating new vehicle remediation
        vehicleRemediation = VehicleRemediation(
          vehicleID: vehicleID,
          vehicleInspectionID: vehicleInspectionID,
        );
      }
    }

    // returning the vehicle remediation
    return vehicleRemediation;
  }

  // /////////////// //
  // GETTING OBJECTS //
  // /////////////// //

  /// Returns a [Stream] of the [List] of [VehicleRemediation]s associated with
  /// the given [vehicleID].
  ///
  /// This [VehicleRemediation] are sorted based on their [timestamp].
  Stream<List<VehicleRemediation>> getVehicleRemediationsWhereVehicleIs(
    String vehicleID,
  ) {
    return _vehicleRemediationsRef
        .where("vehicleID", isEqualTo: vehicleID)
        .orderBy("timestamp", descending: true)
        .snapshots()
        .map(
          (snapshot) => snapshot.docs
              .map((doc) => VehicleRemediation.fromFirestore(doc))
              .toList(),
        );
  }

  /// Returns a [Stream] of the [List] of [SignRemediation]s associated
  /// with the given [vehicleRemediationID].
  ///
  /// This [SignRemediation]s are sorted based on their [index] property.
  Stream<List<SignRemediation>> getSignRemediationsWhereVehicleRemediationIs(
    String vehicleRemediationID,
  ) {
    return _signRemediationsRef
        .where("vehicleRemediationID", isEqualTo: vehicleRemediationID)
        .snapshots()
        .map(
          (snapshot) => snapshot.docs
              .map((doc) => SignRemediation.fromFirestore(doc))
              .toList(),
        );
  }

  // ////////////// //
  // GETTING IMAGES //
  // ////////////// //

  /// Returns the download URL for the unprocessed image of an
  /// [InspectionCheckpoint].
  Stream<String> getSignRemediationDownloadURL(
    String vehicleID,
    String vehicleRemediationID,
    String signRemediationID,
  ) {
    // defining reference to Storage
    Reference reference = FirebaseStorage.instance.ref(
        "$vehicleID/vehicleRemediations/$vehicleRemediationID/$signRemediationID.png");

    // returning download URL
    return reference.getDownloadURL().asStream();
  }
}
