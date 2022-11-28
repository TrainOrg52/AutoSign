import 'package:flutter/material.dart';

/// Page to carry out a remediation for a train vehicle.
///
/// TODO
class RemediatePage extends StatelessWidget {
  // MEMBERS //
  final String vehicleID;

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  const RemediatePage({
    super.key,
    required this.vehicleID,
  });

  // //////////// //
  // BUILD METHOD //
  // //////////// //

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(child: Text("Remediate for $vehicleID")),
    );
  }
}
