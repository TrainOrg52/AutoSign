import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import 'package:train_vis_mobile/view/theme/my_colors.dart';
import 'package:train_vis_mobile/view/theme/my_sizes.dart';

/// The set of icon buttons to be used within the application.
class MyIconButton extends StatelessWidget {
  // MEMBER VARIABLES //
  final IconData iconData;
  final Function() onPressed;

  // THEME-ING //
  // MySizes
  final double height;
  final double width;
  final EdgeInsetsGeometry padding;
  final double borderWidth;
  final double borderRadius;
  final double iconSize;
  // colors
  final Color? iconColor;
  final Color backgroundColor;
  final Color borderColor;

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  /// Constructs a new [MyIconButton] using the provided information.
  ///
  /// Private so only the pre-defined icon buttons can be used.
  const MyIconButton._({
    Key? key,
    // member variables
    required this.iconData,
    required this.onPressed,
    // MySizes
    double? height,
    double? width,
    EdgeInsetsGeometry? padding,
    double? borderWidth,
    double? borderRadius,
    double? iconSize,
    // colors
    this.iconColor,
    required this.backgroundColor,
    required this.borderColor,
  })  : height = height ?? MySizes.buttonHeight,
        width = width ?? MySizes.buttonHeight,
        padding = padding ?? EdgeInsets.zero,
        borderWidth = borderWidth ?? MySizes.borderWidth,
        borderRadius = borderRadius ?? MySizes.borderRadius,
        iconSize = iconSize ?? MySizes.mediumIconSize,
        super(key: key);

  // //////////// //
  // BUILD METHOD //
  // //////////// //

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: height,
      width: width,
      child: Material(
        shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(borderRadius),
            side: BorderSide(
              color: borderColor,
              width: borderWidth,
            )),
        color: backgroundColor,
        child: InkWell(
          borderRadius: BorderRadius.circular(borderRadius),
          onTap: onPressed,
          child: Container(
            padding: padding,
            child: Icon(
              iconData,
              size: iconSize,
              color: iconColor,
            ),
          ),
        ),
      ),
    );
  }

  // /////// //
  // PRIMARY //
  // /////// //

  /// Primary icon button.
  static MyIconButton primary({
    // member variables
    required IconData iconData,
    required Function() onPressed,
    // MySizes
    double? height,
    double? width,
    EdgeInsetsGeometry? padding,
    double? borderWidth,
    double? borderRadius,
    double? iconSize,
  }) {
    return MyIconButton._(
      // member variables
      iconData: iconData,
      onPressed: onPressed,
      // MySizes
      height: height,
      width: width,
      padding: padding,
      borderWidth: borderWidth,
      borderRadius: borderRadius,
      iconSize: iconSize,
      // colors
      iconColor: MyColors.antiPrimary,
      backgroundColor: MyColors.primary,
      borderColor: Colors.transparent,
    );
  }

  // ///////// //
  // SECONDARY //
  // ///////// //

  /// Secondary icon button.
  static MyIconButton secondary({
    // member variables
    required IconData iconData,
    required Function() onPressed,
    // MySizes
    double? height,
    double? width,
    EdgeInsetsGeometry? padding,
    double? borderWidth,
    double? borderRadius,
    double? iconSize,
    // colors
    Color? iconColor,
  }) {
    return MyIconButton._(
      // member variables
      iconData: iconData,
      onPressed: onPressed,
      // MySizes
      height: height,
      width: width,
      padding: padding,
      borderWidth: borderWidth,
      borderRadius: borderRadius,
      iconSize: iconSize,
      // colors
      iconColor: iconColor,
      backgroundColor: Colors.transparent,
      borderColor: Colors.transparent,
    );
  }

  // //////// //
  // NEGATIVE //
  // //////// //

  /// Negative Icon button.
  static MyIconButton negative({
    // member variables
    required IconData iconData,
    required Function() onPressed,
    // MySizes
    double? height,
    double? width,
    EdgeInsetsGeometry? padding,
    double? borderWidth,
    double? borderRadius,
    double? iconSize,
  }) {
    return MyIconButton._(
      // member variables
      iconData: iconData,
      onPressed: onPressed,
      // MySizes
      height: height,
      width: width,
      padding: padding,
      borderWidth: borderWidth,
      borderRadius: borderRadius,
      iconSize: iconSize,
      // colors
      iconColor: MyColors.antiNegative,
      backgroundColor: MyColors.negative,
      borderColor: Colors.transparent,
    );
  }

  // /// //
  // BACK //
  // /// //

  /// Back icon button.
  static MyIconButton back({
    // member variables
    required Function() onPressed,
    // MySizes
    double? height, // increased height
    double? width, // increased width
    EdgeInsetsGeometry? padding,
    double? borderWidth,
    double? borderRadius,
    double iconSize = MySizes.largeIconSize,
  }) {
    return MyIconButton.secondary(
      // member variables
      iconData: FontAwesomeIcons.arrowLeftLong,
      onPressed: onPressed,
      // MySizes
      height: height,
      width: width,
      padding: padding,
      borderWidth: borderWidth,
      borderRadius: borderRadius,
      iconSize: iconSize,
    );
  }
}
