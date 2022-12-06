import 'package:auto_sign_mobile/view/theme/data/my_colors.dart';
import 'package:flutter/material.dart';

/// Defines the processing status of a vehicle inspection.
class RemediationStatus {
  // MEMBER VARIABLES //
  final String title; // title of the status
  final Color color; // color associated with the status
  final Color accentColor; // acccent color associated with the status

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  /// Constructs a new [RemediationStatus] with the provided title and color.
  ///
  /// Private so that only the pre-defined [RemediationStatus] instances
  /// can be used.
  const RemediationStatus._({
    required this.title,
    required this.color,
    required this.accentColor,
  });

  // ///////////////// //
  // STRING CONVERTION //
  // ///////////////// //

  /// Creates a [RemediationStatus] based on the given string.
  static RemediationStatus? fromString(String status) {
    // returning based on string value
    if (status == none.title) {
      return none;
    } else if (status == partial.title) {
      return partial;
    } else if (status == complete.title) {
      return complete;
    } else if (status == error.title) {
      return error;
    }

    // no matching value -> return null
    else {
      return null;
    }
  }

  /// Converts the [RemediationStatus] object to a string representation.
  @override
  String toString() {
    // returning title
    return title;
  }

  // ///////// //
  // INSTANCES //
  // ///////// //

  // all instances
  static const List<RemediationStatus> values = [
    none,
    partial,
    complete,
    error,
  ];

  // none
  static const RemediationStatus none = RemediationStatus._(
    title: "none",
    color: MyColors.lineColor,
    accentColor: MyColors.greyAccent,
  );

  // partial
  static const RemediationStatus partial = RemediationStatus._(
    title: "partial",
    color: MyColors.amber,
    accentColor: MyColors.amberAccent,
  );

  // complete
  static const RemediationStatus complete = RemediationStatus._(
    title: "complete",
    color: MyColors.green,
    accentColor: MyColors.greenAccent,
  );

  // error
  static const RemediationStatus error = RemediationStatus._(
    title: "error",
    color: MyColors.negative,
    accentColor: MyColors.negativeAccent,
  );
}
