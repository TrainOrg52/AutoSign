import 'package:flutter/material.dart';
import 'package:train_vis_mobile/view/pages/profile/vehicle_overview.dart';
import 'package:train_vis_mobile/view/theme/data/my_text_styles.dart';
import 'package:train_vis_mobile/view/theme/widgets/my_icon_button.dart';
import 'package:train_vis_mobile/view/widgets/padded_custom_scroll_view.dart';

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
      // APP BAR //
      appBar: AppBar(
        leading: MyIconButton.back(
          onPressed: () {
            Navigator.of(context).pop();
          },
        ),
        title: Text(vehicleID, style: MyTextStyles.headerText1),
      ),
      body: PaddedCustomScrollView(
        slivers: [
          // /////////////// //
          // VEHICLE SUMMARY //
          // /////////////// //

          SliverToBoxAdapter(
            child: VehicleOverview(vehicleID: vehicleID),
          ),
        ],
      ),
    );
  }
}
