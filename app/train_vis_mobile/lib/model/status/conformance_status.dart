import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import 'package:train_vis_mobile/view/theme/data/my_colors.dart';

/// The conformance status of a vehicle, checkpoint, or sign.
class ConformanceStatus {
  // MEMBER VARIABLES //
  final String title; // title for the status
  final String description; // description of the status
  final Color color; // color associated with the status
  final Color accentColor; // acccent color associated with the status
  final IconData iconData; // icon to be displayed for this status

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  /// Constructs a new [ConformanceStatus] with the provided title and color.
  ///
  /// Private so that only the pre-defined [ConformanceStatus] instances
  /// can be used.
  const ConformanceStatus._({
    required this.title,
    required this.description,
    required this.color,
    required this.accentColor,
    required this.iconData,
  });

  // ///////////////// //
  // STRING CONVERTION //
  // ///////////////// //

  /// Creates a [ConformanceStatus] based on the given string.
  static ConformanceStatus? fromString(String status) {
    // returning value based on string
    if (status == pending.title) {
      return pending;
    } else if (status == conforming.title) {
      return conforming;
    } else if (status == nonConforming.title) {
      return nonConforming;
    } else if (status == error.title) {
      return error;
    }

    // no matching value -> return null
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
  static const List<ConformanceStatus> values = [
    pending,
    conforming,
    nonConforming,
    error,
  ];

  // pending
  static const ConformanceStatus pending = ConformanceStatus._(
    title: "pending",
    description: "Inspection Pending",
    color: MyColors.amber,
    accentColor: MyColors.amberAccent,
    iconData: FontAwesomeIcons.solidClock,
  );

  // conforming
  static const ConformanceStatus conforming = ConformanceStatus._(
    title: "conforming",
    description: "Conforming",
    color: MyColors.green,
    accentColor: MyColors.greenAcent,
    iconData: FontAwesomeIcons.solidCircleCheck,
  );

  // non-conforming
  static const ConformanceStatus nonConforming = ConformanceStatus._(
    title: "non-conforming",
    description: "Non-Conforming",
    color: MyColors.negative,
    accentColor: MyColors.negativeAccent,
    iconData: FontAwesomeIcons.circleExclamation,
  );

  // error
  static const ConformanceStatus error = ConformanceStatus._(
    title: "error",
    description: "Error",
    color: MyColors.negative,
    accentColor: MyColors.negativeAccent,
    iconData: FontAwesomeIcons.circleExclamation,
  );
}
