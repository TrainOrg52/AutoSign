import 'package:auto_sign_mobile/model/model_object.dart';
import 'package:auto_sign_mobile/model/status/conformance_status.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

/// A single train vehicle.
class Vehicle extends ModelObject {
  // MEMBERS //
  String title; // title of the vehicle
  String location; // current location of the vehicle
  ConformanceStatus conformanceStatus; // current conformance status of vehicle
  String lastVehicleInspectionID; // ID of last inspection done on vehicle

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  Vehicle({
    String id = "",
    int? timestamp,
    this.title = "",
    this.location = "",
    ConformanceStatus? conformanceStatus,
    List<String>? checkpoints,
    this.lastVehicleInspectionID = "",
  })  : conformanceStatus = conformanceStatus ?? ConformanceStatus.pending,
        super(id: id, timestamp: timestamp);

  // ///////// //
  // FIRESTORE //
  // ///////// //

  /// Creates a [Vehicle] object from the provided [DocumentSnapshot].
  factory Vehicle.fromFirestore(
      DocumentSnapshot<Map<String, dynamic>> snapshot) {
    // getting snapshot data
    final data = snapshot.data();

    // cocnverting document data to an object
    return Vehicle(
      id: snapshot.id,
      timestamp: data?["timestamp"],
      title: data?["title"],
      location: data?["location"],
      conformanceStatus:
          ConformanceStatus.fromString(data?["conformanceStatus"]),
      lastVehicleInspectionID: data?["lastVehicleInspectionID"],
    );
  }

  /// Converts the [Vehicle] into a [Map] that can be published to firestore.
  @override
  Map<String, dynamic> toFirestore() {
    // converting the object into a map
    return {
      "timestamp": timestamp,
      "title": title,
      "location": location,
      "conformanceStatus": conformanceStatus.toString(),
      "lastVehicleInspectionID": lastVehicleInspectionID,
    };
  }
}
