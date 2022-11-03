import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:train_vis_mobile/model/ModelObject.dart';

/// The gold standard walkthrough of a given train vehicle.
class Walkthrough extends ModelObject {
  // MEMBERS //
  List<String> checkpoints; // list of checkpoints within the walkthrough

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  Walkthrough({
    String id = "",
    int? timestamp,
    List<String>? checkpoints,
  })  : checkpoints = checkpoints ?? [],
        super(id: id, timestamp: timestamp);

  // ///////// //
  // FIRESTORE //
  // ///////// //

  /// Creates a [Walkthrough] object from the provided [DocumentSnapshot].
  factory Walkthrough.fromFirestore(
      DocumentSnapshot<Map<String, dynamic>> snapshot) {
    // getting snapshot data
    final data = snapshot.data();

    // cocnverting document data to an [Walkthrough]
    return Walkthrough(
      id: snapshot.id,
      timestamp: data?["timestamp"],
      checkpoints: data?["checkpoints"],
    );
  }

  /// Converts the [Walkthrough] into a [Map] that can be published to firestore.
  @override
  Map<String, dynamic> toFirestore() {
    // converting the walkthrough into a map
    return {
      "timestamp": timestamp,
      "checkpoints": checkpoints,
    };
  }
}
