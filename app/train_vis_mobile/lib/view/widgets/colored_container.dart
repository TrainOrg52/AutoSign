import 'package:flutter/material.dart';
import 'package:train_vis_mobile/view/theme/my_sizes.dart';

/// A wrapper class for a [Container] that has a background color and rounded
/// corners.
class ColoredContainer extends StatelessWidget {
  // MEMBER VARIABLES //
  final Widget? child;
  final Color color;

  // THEME-ING //
  // sizing
  final EdgeInsetsGeometry padding;
  final double borderRadius;

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  /// Constructs a new [ColoredContainer] with the provided information.
  const ColoredContainer({
    Key? key,
    // member variables
    this.child,
    required this.color,
    // sizing
    this.padding = const EdgeInsets.all(MySizes.paddingValue / 2),
    this.borderRadius = MySizes.borderRadius,
  }) : super(key: key);

  // //////////// //
  // BUILD METHOD //
  // //////////// //

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: color,
        borderRadius: BorderRadius.circular(borderRadius),
      ),
      child: Padding(
        padding: padding,
        child: child,
      ),
    );
  }
}
