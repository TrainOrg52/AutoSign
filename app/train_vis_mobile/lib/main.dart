import 'package:flutter/material.dart';
import 'package:train_vis_mobile/view/theme/my_colors.dart';

// /////////// //
// MAIN METHOD //
// /////////// //

void main() {
  runApp(const MyApp());
}

// ////// //
// MY APP //
// ////// //

/// Main application class
class MyApp extends StatelessWidget {
  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  const MyApp({super.key});

  // //////////// //
  // BUILD METHOD //
  // //////////// //

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      // CONFIGURATION //
      debugShowCheckedModeBanner: false, // hiding debug banner
      title: 'TrainVis',

      // THEME //
      theme: ThemeData(
        fontFamily: "Poppins",
        primaryColor: MyColors.primary,
        scaffoldBackgroundColor: MyColors.backgroundPrimary,
      ),

      // BUILDING APP //
      home: const Center(child: Text("Home")),
    );
  }
}
