import 'package:flutter/material.dart';

/// Page to view the remediations that have occured on a train vehicle.
///
/// TODO
class RemediationsPage extends StatelessWidget {
  // MEMBERS //
  final String vehicleID;

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  const RemediationsPage({
    super.key,
    required this.vehicleID,
  });

  // //////////// //
  // BUILD METHOD //
  // //////////// //

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(child: Text("Remediations for $vehicleID")),
    );
  }
}
