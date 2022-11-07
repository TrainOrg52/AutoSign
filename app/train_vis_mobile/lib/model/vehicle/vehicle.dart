import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:train_vis_mobile/model/model_object.dart';
import 'package:train_vis_mobile/model/status/conformance_status.dart';

/// A single train vehicle.
class Vehicle extends ModelObject {
  // MEMBERS //
  String title; // title for the vehicle
  String location; // current location of vehicle
  ConformanceStatus conformanceStatus; // current conformance status of vehicle
  List<String> checkpoints; // checkpoints in the vehicle

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
  })  : conformanceStatus = conformanceStatus ?? ConformanceStatus.pending,
        checkpoints = checkpoints ?? [],
        super(id: id, timestamp: timestamp);

  // ///////// //
  // FIRESTORE //
  // ///////// //

  /// Creates a [Vehicle] object from the provided [DocumentSnapshot].
  factory Vehicle.fromFirestore(
      DocumentSnapshot<Map<String, dynamic>> snapshot) {
    // getting snapshot data
    final data = snapshot.data();

    // cocnverting document data to an [Vehicle]
    return Vehicle(
      id: snapshot.id,
      timestamp: data?["timestamp"],
      title: data?["title"],
      location: data?["location"],
      conformanceStatus:
          ConformanceStatus.fromString(data?["conformanceStatus"]),
      checkpoints: List.from(data?["checkpoints"]),
    );
  }

  /// Converts the [Vehicle] into a [Map] that can be published to firestore.
  @override
  Map<String, dynamic> toFirestore() {
    // converting the module into a map
    return {
      "timestamp": timestamp,
      "title": title,
      "location": location,
      "conformanceStatus": conformanceStatus.toString(),
      "checkpoints": checkpoints,
    };
  }
}
