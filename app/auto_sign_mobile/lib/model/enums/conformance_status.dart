import 'package:auto_sign_mobile/view/theme/data/my_colors.dart';
import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';

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
    } else if (status == missing.title) {
      return missing;
    } else if (status == damaged.title) {
      return damaged;
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

  // ////////////// //
  // HELPER METHODS //
  // ////////////// //

  /// Determines if the [ConformanceStatus] is conforming.
  ///
  /// Returns true if it is conforming, false otherwise.
  bool isConforming() {
    // returning based on conformance status
    return (this == ConformanceStatus.conforming);
  }

  /// Determines if the [ConformanceStatus] is non conforming (non-conforming,
  /// missing or damaged).
  ///
  /// Returns true if it is non-conforming, false otherwise.
  bool isNonConforming() {
    // returning based on conformance status
    return (this == ConformanceStatus.nonConforming ||
        this == ConformanceStatus.missing ||
        this == ConformanceStatus.damaged);
  }

  // ///////// //
  // INSTANCES //
  // ///////// //

  // all instances
  static const List<ConformanceStatus> values = [
    pending,
    conforming,
    nonConforming,
    missing,
    damaged,
    error,
  ];

  // user-selectable instances
  static const List<ConformanceStatus> userSelectableValues = [
    conforming,
    nonConforming,
    missing,
    damaged,
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
    accentColor: MyColors.greenAccent,
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

  // missing
  static const ConformanceStatus missing = ConformanceStatus._(
    title: "missing",
    description: "Missing",
    color: MyColors.negative,
    accentColor: MyColors.negativeAccent,
    iconData: FontAwesomeIcons.circleExclamation,
  );

  // damaged
  static const ConformanceStatus damaged = ConformanceStatus._(
    title: "damaged",
    description: "Damaged",
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
