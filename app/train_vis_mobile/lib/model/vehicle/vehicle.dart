import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:train_vis_mobile/model/model_object.dart';

/// A single train vehicle.
class Vehicle extends ModelObject {
  // MEMBERS //
  String title; // title for the vehicle
  String location; // current location of vehicle

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  Vehicle({
    String id = "",
    int? timestamp,
    this.title = "",
    this.location = "",
  }) : super(id: id, timestamp: timestamp);

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
    };
  }
}
