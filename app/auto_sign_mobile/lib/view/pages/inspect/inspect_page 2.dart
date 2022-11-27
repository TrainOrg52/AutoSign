import 'package:flutter/material.dart';

/// Page to carry out inspection of a train vehicle.
///
/// TODO
class InspectPage extends StatelessWidget {
  // MEMBERS //
  final String vehicleID;

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  const InspectPage({
    super.key,
    required this.vehicleID,
  });

  // //////////// //
  // BUILD METHOD //
  // //////////// //

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(child: Text("Inspect for $vehicleID")),
    );
  }
}
