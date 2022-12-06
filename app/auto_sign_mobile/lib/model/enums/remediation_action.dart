import 'package:auto_sign_mobile/view/theme/data/my_colors.dart';
import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';

/// The conformance status of a vehicle, checkpoint, or sign.
class RemediationAction {
  // MEMBER VARIABLES //
  final String title; // title for the status
  final String description; // description of the status
  final Color color; // color associated with the status
  final Color accentColor; // acccent color associated with the status
  final IconData iconData; // icon to be displayed for this status

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  /// Constructs a new [RemediationAction] with the provided title and color.
  ///
  /// Private so that only the pre-defined [RemediationAction] instances
  /// can be used.
  const RemediationAction._({
    required this.title,
    required this.description,
    required this.color,
    required this.accentColor,
    required this.iconData,
  });

  // ///////////////// //
  // STRING CONVERTION //
  // ///////////////// //

  /// Creates a [RemediationAction] based on the given string.
  static RemediationAction? fromString(String status) {
    // returning value based on string
    if (status == cleaned.title) {
      return cleaned;
    } else if (status == replaced.title) {
      return replaced;
    } else if (status == error.title) {
      return error;
    }

    // no matching value -> return null
    else {
      return null;
    }
  }

  /// Converts the [RemediationAction] object to a string representation.
  @override
  String toString() {
    // returning the title of the status
    return title;
  }

  // ///////// //
  // INSTANCES //
  // ///////// //

  // all instances
  static const List<RemediationAction> values = [
    cleaned,
    replaced,
    error,
  ];

  // cleaned
  static const RemediationAction cleaned = RemediationAction._(
    title: "cleaned",
    description: "Cleaned",
    color: MyColors.green,
    accentColor: MyColors.greenAccent,
    iconData: FontAwesomeIcons.broom,
  );

  // replaced
  static const RemediationAction replaced = RemediationAction._(
    title: "replaced",
    description: "Replaced",
    color: MyColors.green,
    accentColor: MyColors.greenAccent,
    iconData: FontAwesomeIcons.arrowRotateRight,
  );

  // error
  static const RemediationAction error = RemediationAction._(
    title: "error",
    description: "Error",
    color: MyColors.negative,
    accentColor: MyColors.negativeAccent,
    iconData: FontAwesomeIcons.circleExclamation,
  );
}
