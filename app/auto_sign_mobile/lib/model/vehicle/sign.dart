import 'package:auto_sign_mobile/model/enums/conformance_status.dart';

/// TODO
class Sign {
  // MEMBERS //
  String id; // id of the sign
  String title; // title of the vehicle
  ConformanceStatus conformanceStatus; // current conformance status of vehicle

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  Sign({
    this.id = "",
    this.title = "",
    ConformanceStatus? conformanceStatus,
  }) : conformanceStatus = conformanceStatus ?? ConformanceStatus.pending;

  // ///////// //
  // FIRESTORE //
  // ///////// //

  /// Creates a [Sign] object from the provided [data] from a Firestore Document.
  factory Sign.fromFirestoreData(dynamic data) {
    // cocnverting document data to an object
    return Sign(
      id: data?["id"],
      title: data?["title"],
      conformanceStatus:
          ConformanceStatus.fromString(data?["conformanceStatus"]),
    );
  }

  /// Converts the [Sign] into a [Map] that can be published to firestore.
  Map<String, dynamic> toFirestore() {
    // converting the object into a map
    return {
      "id": id,
      "title": title,
      "conformanceStatus": conformanceStatus.toString(),
    };
  }
}
