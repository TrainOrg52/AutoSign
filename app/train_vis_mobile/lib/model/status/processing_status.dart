import 'package:flutter/material.dart';
import 'package:train_vis_mobile/view/theme/my_colors.dart';

/// Defines the processing status of an inspection.
class ProcessingStatus {
  // MEMBER VARIABLES //
  final String title; // the title for the processing status
  final Color color; // the color associated with the processing status

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  /// Constructs a new [ProcessingStatus] with the provided title and color.
  ///
  /// Private so that only the pre-defined [ProcessingStatus] instances
  /// can be used.
  const ProcessingStatus._({
    required this.title,
    required this.color,
  });

  // ///////////////// //
  // STRING CONVERTION //
  // ///////////////// //

  /// Creates a [ProcessingStatus] based on the given string.
  static ProcessingStatus? fromString(String status) {
    // returning priority based on string
    if (status == pending.title) {
      return pending;
    } else if (status == processing.title) {
      return processing;
    } else if (status == processed.title) {
      return processed;
    } else if (status == error.title) {
      return error;
    }

    // no matching status -> return null
    else {
      return null;
    }
  }

  /// Converts the [ProcessingStatus] object to a string representation.
  @override
  String toString() {
    // returning the title of the status
    return title;
  }

  // ////////////////// //
  // PRIORITY INSTANCES //
  // ////////////////// //

  // all instances
  static const List<ProcessingStatus> statuses = [
    pending,
    processing,
    processed,
    error,
  ];

  // pending
  static const ProcessingStatus pending = ProcessingStatus._(
    title: "pending",
    color: MyColors.lineColor,
  );

  // processing
  static const ProcessingStatus processing = ProcessingStatus._(
    title: "processing",
    color: MyColors.amber,
  );

  // processed
  static const ProcessingStatus processed = ProcessingStatus._(
    title: "processed",
    color: MyColors.green,
  );

  // error
  static const ProcessingStatus error = ProcessingStatus._(
    title: "error",
    color: MyColors.negative,
  );
}
