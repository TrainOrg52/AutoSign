import 'package:auto_sign_mobile/view/theme/data/my_colors.dart';
import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';

/// Defines the processing status of a vehicle inspection.
class ProcessingStatus {
  // MEMBER VARIABLES //
  final String title; // title of the status
  final Color color; // color associated with the status
  final Color accentColor; // acccent color associated with the status
  final IconData iconData; // icon to be displayed for this status

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
    required this.accentColor,
    required this.iconData,
  });

  // ///////////////// //
  // STRING CONVERTION //
  // ///////////////// //

  /// Creates a [ProcessingStatus] based on the given string.
  static ProcessingStatus? fromString(String status) {
    // returning based on string value
    if (status == uploading.title) {
      return uploading;
    } else if (status == pending.title) {
      return pending;
    } else if (status == processing.title) {
      return processing;
    } else if (status == processed.title) {
      return processed;
    } else if (status == processed.title) {
      return processed;
    } else if (status == error.title) {
      return error;
    }

    // no matching value -> return null
    else {
      return null;
    }
  }

  /// Converts the [ProcessingStatus] object to a string representation.
  @override
  String toString() {
    // returning title
    return title;
  }

  // ///////// //
  // INSTANCES //
  // ///////// //

  // all instances
  static const List<ProcessingStatus> values = [
    uploading,
    pending,
    processing,
    processed,
    error,
  ];

  // uploading
  static const ProcessingStatus uploading = ProcessingStatus._(
    title: "uploading",
    color: MyColors.amber,
    accentColor: MyColors.amberAccent,
    iconData: FontAwesomeIcons.arrowUp,
  );

  // pending
  static const ProcessingStatus pending = ProcessingStatus._(
    title: "pending",
    color: MyColors.amber,
    accentColor: MyColors.amberAccent,
    iconData: FontAwesomeIcons.solidClock,
  );

  // processing
  static const ProcessingStatus processing = ProcessingStatus._(
    title: "processing",
    color: MyColors.amber,
    accentColor: MyColors.amberAccent,
    iconData: FontAwesomeIcons.solidClock,
  );

  // processed
  static const ProcessingStatus processed = ProcessingStatus._(
    title: "processed",
    color: MyColors.green,
    accentColor: MyColors.greenAcent,
    iconData: FontAwesomeIcons.solidCircleCheck,
  );

  // error
  static const ProcessingStatus error = ProcessingStatus._(
    title: "error",
    color: MyColors.negative,
    accentColor: MyColors.negativeAccent,
    iconData: FontAwesomeIcons.circleExclamation,
  );
}
