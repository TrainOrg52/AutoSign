import 'package:flutter/material.dart';

/// Page to view a specific remediation that has occured on a train vehicle.
///
/// TODO
class RemediationWalkthroughPage extends StatelessWidget {
  // MEMBERS //
  final String remediationWalkthroughID;

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  const RemediationWalkthroughPage({
    super.key,
    required this.remediationWalkthroughID,
  });

  // //////////// //
  // BUILD METHOD //
  // //////////// //

  @override
  Widget build(BuildContext context) {
    return const Scaffold(
      body: Center(child: Text("Individual Remediation")),
    );
  }
}
