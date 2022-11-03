import 'package:flutter/material.dart';

/// The home page of the application.
///
/// TODO
class HomePage extends StatelessWidget {
  // MEMBERS //

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  const HomePage({super.key});

  // //////////// //
  // BUILD METHOD //
  // //////////// //

  @override
  Widget build(BuildContext context) {
    return const Scaffold(
      body: Center(child: Text("Home")),
    );
  }
}
