import 'package:firebase_core/firebase_core.dart';
import 'package:flutter/material.dart';
import 'package:train_vis_mobile/view/routes/routes.dart';

import 'firebase_options.dart';

// /////////// //
// MAIN METHOD //
// /////////// //

Future<void> main() async {
  // configuring app
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp(
    options: DefaultFirebaseOptions.currentPlatform,
  );

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
    // building app
    return MaterialApp.router(
      // CONFIGURATION //
      debugShowCheckedModeBanner: false,

      // ROUTING //
      routerConfig: router,
    );
  }
}
