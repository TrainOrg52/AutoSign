import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:train_vis_mobile/model/vehicle/walkthrough.dart';

/// Controller that manages the application's list of [Walkthroughs] objects.
class WalkthroughController {
  // MEMBER VARIABLES //
  // reference to Firestore collection
  final CollectionReference<Map<String, dynamic>> _walkthroughsRef =
      FirebaseFirestore.instance.collection("walkthroughs");

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  /// Constructs a new [WalkthroughController].
  WalkthroughController._();

  // //////// //
  // INSTANCE //
  // //////// //

  // single instance of controller
  static final WalkthroughController instance = WalkthroughController._();

  // //////////////////// //
  // RETRIEVING AS STREAM //
  // //////////////////// //

  /// Returns a [Stream] of the [Walkthrough] for the vehicle with the given ID.
  Stream<Walkthrough> getWalkthrough(String id) {
    // returning the required walkthrough as a stream
    return _walkthroughsRef
        .doc(id)
        .snapshots()
        .map((snapshot) => Walkthrough.fromFirestore(snapshot));
  }
}
