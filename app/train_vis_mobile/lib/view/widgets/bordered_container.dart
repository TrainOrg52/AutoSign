import 'package:flutter/material.dart';
import 'package:train_vis_mobile/view/theme/data/my_colors.dart';
import 'package:train_vis_mobile/view/theme/data/my_sizes.dart';

/// A convenience [Container] object that comes with a border, padding and
/// background as defined by the app's theme.
class BorderedContainer extends StatelessWidget {
  // MEMBER VARIABLES //
  final Widget child;

  // THEME-ING //
  // MySizes
  final double? height;
  final double width;
  final double spacing;
  final EdgeInsetsGeometry padding;
  final double borderRadius;
  final double borderWidth;
  // colors
  final Color backgroundColor;
  final Color borderColor;

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  /// Constructs a new [BorderedContainer] using the provided information.
  const BorderedContainer({
    Key? key,
    // member variables
    required this.child,
    //MySizes
    this.height,
    this.width = double.infinity,
    this.spacing = MySizes.spacing,
    this.padding = MySizes.padding,
    this.borderRadius = MySizes.borderRadius,
    this.borderWidth = MySizes.borderWidth,
    // colors
    this.borderColor = MyColors.borderColor,
    this.backgroundColor = MyColors.backgroundSecondary,
  }) : super(key: key);

  // //////////// //
  // BUILD METHOD //
  // //////////// //

  @override
  Widget build(BuildContext context) {
    return Container(
      // CONFIGURATION //
      height: height,
      width: width,
      padding: padding,
      decoration: BoxDecoration(
        color: backgroundColor,
        border: Border.all(
          color: borderColor,
          width: borderWidth,
        ),
        borderRadius: BorderRadius.circular(borderRadius),
      ),

      // CONTENT //
      child: child,
    );
  }
}
