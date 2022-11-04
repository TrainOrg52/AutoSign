import 'package:flutter/material.dart';

/// Definitions for application sizing.
class MySizes {
  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  // Private class constructor to prevent instantiation.
  MySizes._();

  // ///////// //
  // RAW SIZING //
  // ///////// //

  static const double spacing = 10.0;
  static const double paddingValue = 10.0;
  static const EdgeInsetsGeometry padding = EdgeInsets.all(paddingValue);
  static const double lineWidth = 2.0;
  static const double borderWidth = 2.0;
  static const double dividerThickness = 1.0;
  static const double borderRadius = 6.0;

  // ///////////// //
  // WIDGET SIZING //
  // ///////////// //

  // dialogs
  static const double regularDialogWidth = 300;
  static const double largeDialogWidth = 400;
  static const double configurationEndDrawerWidth = 500;
  // buttons
  static const double buttonHeight = 36;
  // icons
  static const double smallIconSize = 16;
  static const double mediumIconSize = 20;
  static const double largeIconSize = 24;
}
