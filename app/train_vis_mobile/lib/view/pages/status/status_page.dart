import 'package:flutter/material.dart';

/// Page to display the status of a train vehicle.
///
/// TODO
class StatusPage extends StatelessWidget {
  // MEMBERS //
  final String vehicleID;

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  const StatusPage({
    super.key,
    required this.vehicleID,
  });

  // //////////// //
  // BUILD METHOD //
  // //////////// //

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(child: Text("Status for $vehicleID")),
    );
  }
}
