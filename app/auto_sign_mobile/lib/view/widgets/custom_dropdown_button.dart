import 'package:auto_sign_mobile/view/theme/data/my_colors.dart';
import 'package:auto_sign_mobile/view/theme/data/my_sizes.dart';
import 'package:dropdown_button2/dropdown_button2.dart';
import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';

/// The set of dropdown menu buttons to be used within the application
class CustomDropdownButton<T> extends StatelessWidget {
  // MEMBER VARIABLES //
  final T value;
  final Function(T?) onChanged;
  final List<DropdownMenuItem<T>> items;

  // THEME-ING //
  // sizing
  final double buttonHeight;
  final double itemHeight;
  final EdgeInsetsGeometry buttonPadding;
  final EdgeInsetsGeometry itemPadding;
  final EdgeInsetsGeometry dropDownPadding;
  // colors
  final Color focusColor;
  // widgets
  final Icon icon;
  // styling
  final InputDecoration inputDecoration;

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  /// Constructs a [CustomDropdownButton] with the provided information.
  const CustomDropdownButton({
    Key? key,
    // member variables
    required this.value,
    required this.onChanged,
    required this.items,
    // sizing
    this.buttonHeight = MySizes.buttonHeight,
    this.itemHeight = MySizes.buttonHeight,
    this.buttonPadding = EdgeInsets.zero,
    this.itemPadding = EdgeInsets.zero,
    this.dropDownPadding = MySizes.padding,
    // colors
    this.focusColor = Colors.transparent,
    // widgets
    this.icon = const Icon(
      FontAwesomeIcons.angleDown,
      color: MyColors.textPrimary,
    ),
    // styling
    this.inputDecoration = const InputDecoration(
      border: InputBorder.none,
      isDense: true,
    ),
  }) : super(key: key);

  // //////////// //
  // BUILD METHOD //
  // //////////// //

  @override
  Widget build(BuildContext context) {
    return DropdownButtonFormField2<T>(
      // member variables
      value: value,
      items: items,
      onChanged: onChanged,
      // sizing
      buttonHeight: buttonHeight,
      buttonPadding: buttonPadding,
      itemPadding: itemPadding,
      dropdownPadding: dropDownPadding,
      itemHeight: itemHeight,
      // colors
      focusColor: focusColor,
      // widgets
      icon: icon,
      // decoration
      decoration: inputDecoration,
    );
  }
}
