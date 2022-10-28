import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:train_vis_mobile/view/theme/my_colors.dart';

/// Defines the [TextStyle]s to be used in the project.
class MyTextStyles {
  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  // Private class constructor to prevent instantiation.
  MyTextStyles._();

  // /////// //
  // HEADERS //
  // /////// //

  static const headerText1 = TextStyle(
    fontSize: 20,
    fontWeight: FontWeight.w600,
    color: MyColors.textPrimary,
  );
  static const headerText2 = TextStyle(
    fontSize: 17,
    fontWeight: FontWeight.w600,
    color: MyColors.textPrimary,
  );
  static const headerText3 = TextStyle(
    fontSize: 15,
    fontWeight: FontWeight.w600,
    color: MyColors.textPrimary,
  );

  // ///////// //
  // BODY TEXT //
  // ///////// //

  static const TextStyle bodyText1 = TextStyle(
    fontSize: 15,
    fontWeight: FontWeight.w400,
    color: MyColors.textPrimary,
  );
  static const TextStyle bodyText2 = TextStyle(
    fontSize: 12,
    fontWeight: FontWeight.w400,
    color: MyColors.textPrimary,
  );
  static const TextStyle bodyText3 = TextStyle(
    fontSize: 10,
    fontWeight: FontWeight.w400,
    color: MyColors.textPrimary,
  );

  // /////////// //
  // BUTTON TEXT //
  // /////////// //

  // button text style
  static const TextStyle buttonTextStyle = TextStyle(
    fontWeight: FontWeight.w500,
  );

  // /////////// //
  // DATE FORMAT //
  // /////////// //

  // year-month-day format
  static DateFormat dateFormat = DateFormat("dd-MM-yyyy");
}
