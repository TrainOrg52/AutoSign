/// Represents an application-level object that is modelling a Firestore
/// document.
abstract class ModelObject {
  // MEMBERS //
  String id; // firestore document ID
  int timestamp; // timestamp of original object creation

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  ModelObject({
    this.id = "",
    int? timestamp,
  }) : timestamp =
            timestamp ?? (DateTime.now().millisecondsSinceEpoch / 1000).round();

  // ///////// //
  // FIRESTORE //
  // ///////// //

  /// Converts the [ModelObject] into a [Map] that can be posted to firestore.
  Map<String, dynamic> toFirestore();
}
