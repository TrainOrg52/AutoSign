import 'package:flutter/material.dart';
import 'package:train_vis_mobile/view/theme/my_colors.dart';

/// Defines the conformance status of an inspection walkthrough, checkpoint or
/// sign.
class ConformanceStatus {
  // MEMBER VARIABLES //
  final String title; // the title for the conformance status
  final Color color; // the color associated with the conformances status

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  /// Constructs a new [ConformanceStatus] with the provided title and color.
  ///
  /// Private so that only the pre-defined [ConformanceStatus] instances
  /// can be used.
  const ConformanceStatus._({
    required this.title,
    required this.color,
  });

  // ///////////////// //
  // STRING CONVERTION //
  // ///////////////// //

  /// Creates a [ConformanceStatus] based on the given string.
  static ConformanceStatus? fromString(String status) {
    // returning priority based on string
    if (status == pending.title) {
      return pending;
    } else if (status == conforming.title) {
      return conforming;
    } else if (status == nonConforming.title) {
      return nonConforming;
    } else if (status == error.title) {
      return error;
    }

    // no matching status -> return null
    else {
      return null;
    }
  }

  /// Converts the [ConformanceStatus] object to a string representation.
  @override
  String toString() {
    // returning the title of the status
    return title;
  }

  // ///////// //
  // INSTANCES //
  // ///////// //

  // all instances
  static const List<ConformanceStatus> statuses = [
    pending,
    conforming,
    nonConforming,
    error,
  ];

  // unknown
  static const ConformanceStatus pending = ConformanceStatus._(
    title: "pending",
    color: MyColors.lineColor,
  );

  // conforming
  static const ConformanceStatus conforming = ConformanceStatus._(
    title: "conforming",
    color: MyColors.green,
  );

  // non-conforming
  static const ConformanceStatus nonConforming = ConformanceStatus._(
    title: "non-conforming",
    color: MyColors.negative,
  );

  // error
  static const ConformanceStatus error = ConformanceStatus._(
    title: "error",
    color: MyColors.negative,
  );
}
