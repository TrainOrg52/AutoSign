import 'package:flutter/material.dart';

/// Page to display the profile of a train vehicle.
///
/// TODO
class ProfilePage extends StatelessWidget {
  // MEMBERS //
  final String vehicleID;

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  const ProfilePage({
    super.key,
    required this.vehicleID,
  });

  // //////////// //
  // BUILD METHOD //
  // //////////// //

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(child: Text("Profile for $vehicleID")),
    );
  }
}
