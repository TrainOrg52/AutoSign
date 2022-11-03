import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:train_vis_mobile/model/ModelObject.dart';

/// A checkpoint within the gold standard walkthrough of a given train vehicle.
class Checkpoint extends ModelObject {
  // MEMBERS //
  String walkthroughID; // id of this checkpoints associated walkthrough
  String title; // title of the checkpoint
  String prompt; // prompt shown when capturing the checkpoint
  List<String> signs; // list of signs expected within the checkpoint

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  Checkpoint({
    String id = "",
    int? timestamp,
    this.walkthroughID = "",
    this.title = "",
    this.prompt = "",
    List<String>? signs,
  })  : signs = signs ?? [],
        super(id: id, timestamp: timestamp);

  // ///////// //
  // FIRESTORE //
  // ///////// //

  /// Creates a [Checkpoint] object from the provided [DocumentSnapshot].
  factory Checkpoint.fromFirestore(
      DocumentSnapshot<Map<String, dynamic>> snapshot) {
    // getting snapshot data
    final data = snapshot.data();

    // cocnverting document data to an [Checkpoint]
    return Checkpoint(
      id: snapshot.id,
      timestamp: data?["timestamp"],
      walkthroughID: data?["walkthroughID"],
      title: data?["title"],
      prompt: data?["prompt"],
      signs: data?["signs"],
    );
  }

  /// Converts the [Walkthrough] into a [Map] that can be published to firestore.
  @override
  Map<String, dynamic> toFirestore() {
    // converting to a map
    return {
      "timestamp": timestamp,
      "walkthroughID": walkthroughID,
      "title": title,
      "prompt": prompt,
      "signs": signs,
    };
  }
}
