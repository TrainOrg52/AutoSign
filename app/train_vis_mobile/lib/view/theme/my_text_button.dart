import 'package:flutter/material.dart';
import 'package:train_vis_mobile/view/theme/my_colors.dart';
import 'package:train_vis_mobile/view/theme/my_sizes.dart';
import 'package:train_vis_mobile/view/theme/my_text_styles.dart';

/// Definitions for application text buttons.
class MyTextButton extends StatelessWidget {
  // MEMBER VARIABLES //
  final String text;
  final Function() onPressed;

  // THEME-ING //
  // MySizes
  final double height;
  final double? width;
  final EdgeInsetsGeometry? padding;
  final double borderWidth;
  // text styles
  final TextStyle textStyle;
  // colors
  final Color primaryColor;
  final Color backgroundColor;
  final Color borderColor;

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  /// Constructs a new [MyTextButton] using the provided information.
  ///
  /// Private so only the pre-defined text buttons can be used.
  const MyTextButton._({
    Key? key, // member variables
    required this.text,
    required this.onPressed,
    // MySizes
    double? height,
    this.width,
    this.padding,
    double? borderWidth,
    // text style
    TextStyle? textStyle,
    // colors
    required this.primaryColor,
    required this.backgroundColor,
    required this.borderColor,
  })  : height = height ?? MySizes.buttonHeight,
        borderWidth = borderWidth ?? MySizes.borderWidth,
        textStyle = textStyle ?? MyTextStyles.buttonTextStyle,
        super(key: key);

  // //////////// //
  // BUILD METHOD //
  // //////////// //

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: width,
      height: height,
      child: OutlinedButton(
        style: OutlinedButton.styleFrom(
          foregroundColor: primaryColor,
          backgroundColor: backgroundColor,
          padding: padding,
          side: BorderSide(
            width: borderWidth,
            color: borderColor,
          ),
        ),
        onPressed: onPressed,
        child: Text(
          text,
          style: textStyle,
        ),
      ),
    );
  }

  // /////// //
  // PRIMARY //
  // /////// //

  /// Primary text button.
  static MyTextButton primary({
    // member variables
    required String text,
    required Function() onPressed,
    // MySizes
    double? height,
    double? width,
    EdgeInsetsGeometry? padding,
    double? borderWidth,
    // text style
    TextStyle? textStyle,
  }) {
    return MyTextButton._(
      // member variables
      text: text,
      onPressed: onPressed,
      // MySizes
      height: height,
      width: width,
      padding: padding,
      borderWidth: borderWidth,
      // colors
      primaryColor: MyColors.antiPrimary,
      backgroundColor: MyColors.primary,
      borderColor: MyColors.primary,
    );
  }

  // ///////// //
  // SECONDARY //
  // ///////// //

  /// Secondary text button.
  static MyTextButton secondary({
    // member variables
    required String text,
    required Function() onPressed,
    // MySizes
    double? height,
    double? width,
    EdgeInsetsGeometry? padding,
    double? borderWidth,
    // text style
    TextStyle? textStyle,
  }) {
    return MyTextButton._(
      // member variables
      text: text,
      onPressed: onPressed,
      // MySizes
      height: height,
      width: width,
      padding: padding,
      borderWidth: borderWidth,
      // colors
      primaryColor: MyColors.textPrimary,
      backgroundColor: Colors.transparent,
      borderColor: MyColors.borderColor,
    );
  }

  // //////// //
  // NEGATIVE //
  // //////// //

  /// Negative tet button.
  static MyTextButton negative({
    // member variables
    required String text,
    required Function() onPressed,
    // MySizes
    double? height,
    double? width,
    EdgeInsetsGeometry? padding,
    double? borderWidth,
    // text style
    TextStyle? textStyle,
  }) {
    return MyTextButton._(
      // member variables
      text: text,
      onPressed: onPressed,
      // MySizes
      height: height,
      width: width,
      padding: padding,
      borderWidth: borderWidth,
      // colors
      primaryColor: MyColors.antiNegative,
      backgroundColor: MyColors.negative,
      borderColor: MyColors.negative,
    );
  }
}
