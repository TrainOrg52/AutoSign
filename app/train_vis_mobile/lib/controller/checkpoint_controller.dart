import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:train_vis_mobile/model/vehicle/checkpoint.dart';

/// Controller that manages the application's list of [Checkpoint] objects.
class CheckpointController {
  // MEMBER VARIABLES //
  // reference to Firestore collection
  final CollectionReference<Map<String, dynamic>> _checkpointsRef =
      FirebaseFirestore.instance.collection("checkpoints");

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  /// Constructs a new [CheckpointController].
  CheckpointController._();

  // //////// //
  // INSTANCE //
  // //////// //

  // single instance of controller
  static final CheckpointController instance = CheckpointController._();

  // //////////////////// //
  // RETRIEVING AS STREAM //
  // //////////////////// //

  /// Returns a [Stream] of the [Checkpoint] with the given [id].
  Stream<Checkpoint> getCheckpoint(String id) {
    // returning the required checkpoint as a stream
    return _checkpointsRef
        .doc(id)
        .snapshots()
        .map((snapshot) => Checkpoint.fromFirestore(snapshot));
  }

  // /// Returns the list of [Checkpoint]s associated with the given [Walkthrough].
  // Stream<List<Checkpoint> getCheckpointsWhereWalkthroughIs(String walkthroughID){
  //   // getting walkthrough object
  //   Walkthrough
  // }
}
