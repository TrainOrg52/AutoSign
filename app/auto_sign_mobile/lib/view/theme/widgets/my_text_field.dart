import 'package:auto_sign_mobile/view/theme/data/my_colors.dart';
import 'package:auto_sign_mobile/view/theme/data/my_sizes.dart';
import 'package:flutter/material.dart';

/// Text fields to be used within the application.
class MyTextField extends StatelessWidget {
  // MEMBER VARIABLES //
  final TextEditingController controller;
  final String? hintText;

  // THEME-ING //
  // Sizes
  final EdgeInsetsGeometry padding;
  final double borderWidth;
  final double borderRadius;
  // colors
  final Color borderColor;

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  /// Constructs a new [MyTextField] using the provided information.
  ///
  /// Private so only the pre-defined icon buttons can be used.
  const MyTextField._({
    Key? key,
    // member variables
    required this.controller,
    required this.hintText,
    // MySizes
    EdgeInsetsGeometry? padding,
    double? borderWidth,
    double? borderRadius,
    // colors
    Color? borderColor,
  })  : padding = padding ?? EdgeInsets.zero,
        borderWidth = borderWidth ?? MySizes.borderWidth,
        borderRadius = borderRadius ?? MySizes.borderRadius,
        borderColor = borderColor ?? MyColors.lineColor,
        super(key: key);

  // //////////// //
  // BUILD METHOD //
  // //////////// //

  @override
  Widget build(BuildContext context) {
    return TextField(
      controller: controller,
      decoration: InputDecoration(
        hintText: hintText,
        isDense: true,
        contentPadding: MySizes.padding,
        border: const OutlineInputBorder(
          borderSide:
              BorderSide(color: MyColors.lineColor, width: MySizes.lineWidth),
        ),
        focusedBorder: const OutlineInputBorder(
          borderSide:
              BorderSide(color: MyColors.lineColor, width: MySizes.lineWidth),
        ),
      ),
    );
  }

  // ////// //
  // NORMAL //
  // ////// //

  /// Primary icon button.
  static MyTextField normal({
    // member variables
    required TextEditingController controller,
    String? hintText,
    // MySizes
    EdgeInsetsGeometry? padding,
    double? borderWidth,
    double? borderRadius,
    double? iconSize,
  }) {
    return MyTextField._(
      // member variables
      controller: controller,
      hintText: hintText,
      // MySizes
      padding: padding,
      borderWidth: borderWidth,
      borderRadius: borderRadius,
      // colors
      borderColor: MyColors.lineColor,
    );
  }

  // /////// //
  // PRIMARY //
  // /////// //

  /// Primary icon button.
  static MyTextField plain({
    // member variables
    required TextEditingController controller,
    String? hintText,
    // MySizes
    double? borderWidth,
    double? borderRadius,
    double? iconSize,
  }) {
    return MyTextField._(
      // member variables
      controller: controller,
      hintText: hintText,
      // MySizes
      borderWidth: 0,
      borderRadius: 0,
      // colors
      borderColor: Colors.transparent,
    );
  }
}
