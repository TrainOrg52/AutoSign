import 'package:flutter/material.dart';

/// Page to view a specific checkpoint remediation that has occured on a train
/// vehicle.
///
/// TODO
class RemediationCheckpointPage extends StatelessWidget {
  // MEMBERS //
  final String remediationCheckpointID;

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  const RemediationCheckpointPage({
    super.key,
    required this.remediationCheckpointID,
  });

  // //////////// //
  // BUILD METHOD //
  // //////////// //

  @override
  Widget build(BuildContext context) {
    return const Scaffold(
      body: Center(child: Text("Individual Remediation Checkpoint")),
    );
  }
}
