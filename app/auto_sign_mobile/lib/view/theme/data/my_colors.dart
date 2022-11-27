import 'package:flutter/material.dart';

/// Definitions for application colors.
class MyColors {
  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  // Private class constructor to prevent instantiation.
  MyColors._();

  // //////////// //
  // THEME COLORS //
  // //////////// //

  // main colors
  static const Color primary = Color.fromARGB(255, 127, 90, 240);
  static const Color negative = red;
  // accent colors
  static const Color primaryAccent = Color.fromARGB(255, 173, 150, 242);
  static const Color negativeAccent = redAccent;
  // anti colors
  static const Color antiPrimary = Colors.white;
  static const Color antiNegative = Colors.white;
  // background color
  static const Color backgroundPrimary = Color.fromARGB(255, 242, 241, 245);
  static const Color backgroundSecondary = Colors.white;
  // text colors
  static const Color textPrimary = Color.fromARGB(255, 32, 34, 37);
  static const Color textSecondary = grey500;
  // line colors
  static const Color lineColor = grey500;
  static const Color borderColor = grey500;

  // ////////// //
  // RAW COLORS //
  // ////////// //

  // MAIN COLORS //
  static const Color green = Color.fromARGB(255, 104, 214, 155);
  static const Color greenAcent = Color.fromARGB(255, 193, 232, 211);
  static const Color amber = Color.fromARGB(255, 226, 162, 126);
  static const Color amberAccent = Color.fromARGB(255, 234, 202, 182);
  static const Color red = Color.fromARGB(255, 217, 119, 117);
  static const Color redAccent = Color.fromARGB(255, 235, 183, 183);
  static const Color greyAccent = grey100;

  // GREY COLORS (higher is darker) //
  static const Color grey100 = Color.fromARGB(255, 250, 250, 250);
  static const Color grey200 = Color.fromARGB(255, 245, 245, 245);
  static const Color grey300 = Color.fromARGB(255, 238, 238, 238);
  static const Color grey400 = Color.fromARGB(255, 224, 224, 224);
  static const Color grey500 = Color.fromARGB(255, 189, 189, 189);
  static const Color grey600 = Color.fromARGB(255, 153, 153, 153);
  static const Color grey700 = Color.fromARGB(255, 117, 117, 117);
  static const Color grey800 = Color.fromARGB(255, 97, 97, 97);
  static const Color grey900 = Color.fromARGB(255, 66, 66, 66);
  static const Color grey1000 = Color.fromARGB(255, 33, 33, 33);
}
