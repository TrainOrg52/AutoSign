import 'package:auto_sign_mobile/view/routes/routes.dart';
import 'package:auto_sign_mobile/view/theme/data/my_colors.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:intl/intl.dart';

import 'firebase_options.dart';

// /////////// //
// MAIN METHOD //
// /////////// //

Future<void> main() async {
  // ///////////////// //
  // APP CONFIGURATION //
  // ///////////////// //

  // Firebase
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp(
    options: DefaultFirebaseOptions.currentPlatform,
  );

  // device orientation
  SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
  ]);

  // /////////// //
  // RUNNING APP //
  // /////////// //

  // running app
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
    return MaterialApp.router(
      // CONFIGURATION //
      debugShowCheckedModeBanner: false,

      // ROUTING //
      routerConfig: Routes.router,

      // THEME //
      theme: ThemeData(
        // scaffold
        scaffoldBackgroundColor: MyColors.backgroundPrimary,

        // app bar
        appBarTheme: const AppBarTheme(
          backgroundColor: MyColors.backgroundSecondary,
          foregroundColor: MyColors.textPrimary,
          elevation: 0,
        ),

        // text
        fontFamily: "Poppins",
      ),
    );
  }
}

// ////////////// //
// HELPER METHODS //
// ////////////// //

/// TODO
extension StringCasingExtension on String {
  /// TODO
  String toCapitalized() =>
      length > 0 ? '${this[0].toUpperCase()}${substring(1).toLowerCase()}' : '';

  /// TODO
  String toTitleCase() => replaceAll(RegExp(' +'), ' ')
      .split(' ')
      .map((str) => str.toCapitalized())
      .join(' ');
}

/// TODO
extension TimestampExtension on int {
  /// TODO
  String toDate() {
    // gathering date time object
    DateTime date = DateTime.fromMillisecondsSinceEpoch(this * 1000);

    // defining format of date time object
    DateFormat dateFormat = DateFormat('dd/MM/yy - HH:mm ');

    // formatting date-time object
    String formattedDate = dateFormat.format(date);

    // returning formatted date
    return formattedDate;
  }
}
