import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:flutter/material.dart';
import 'package:train_vis_mobile/view/theme/my_colors.dart';

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
    // getting data from firestore
    var doc = FirebaseFirestore.instance
        .collection("trains")
        .doc("WEqEzn6V2LRtkTLEVy2u")
        .get();

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
      home: Center(
        child: FutureBuilder<DocumentSnapshot<Map<String, dynamic>>>(
          future: doc,
          initialData: null,
          builder: (BuildContext context, AsyncSnapshot snapshot) {
            if (snapshot.data != null) {
              return Text(snapshot.data["trainID"]);
            } else {
              return const Text("No data");
            }
          },
        ),
      ),
    );
  }
}
