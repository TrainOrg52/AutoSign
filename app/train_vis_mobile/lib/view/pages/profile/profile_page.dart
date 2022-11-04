import 'package:flutter/material.dart';
import 'package:train_vis_mobile/view/pages/profile/vehicle_action_container.dart';
import 'package:train_vis_mobile/view/pages/profile/vehicle_activity_container.dart';
import 'package:train_vis_mobile/view/pages/profile/vehicle_overview_container.dart';
import 'package:train_vis_mobile/view/pages/profile/vehicle_status_container.dart';
import 'package:train_vis_mobile/view/theme/data/my_sizes.dart';
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
      // /////// //
      // APP BAR //
      // /////// //

      appBar: AppBar(
        leading: MyIconButton.back(
          onPressed: () {
            Navigator.of(context).pop();
          },
        ),
        title: Text(vehicleID, style: MyTextStyles.headerText1),
      ),

      // //// //
      // BODY //
      // //// //

      body: PaddedCustomScrollView(
        slivers: [
          // //////////////// //
          // VEHICLE OVERVIEW //
          // //////////////// //

          SliverToBoxAdapter(
            child: VehicleOverviewContainer(vehicleID: vehicleID),
          ),

          const SliverToBoxAdapter(child: SizedBox(height: MySizes.spacing)),

          // /////////////////////////// //
          // CONFORMANCE STATUS OVERVIEW //
          // /////////////////////////// //

          SliverToBoxAdapter(
            child: VehicleConformanceStatusOverview(
              vehicleID: vehicleID,
            ),
          ),

          const SliverToBoxAdapter(child: SizedBox(height: MySizes.spacing)),

          // ////// //
          // ACTION //
          // ////// //

          SliverToBoxAdapter(
            child: VehicleActionContainer(
              vehicleID: vehicleID,
            ),
          ),

          const SliverToBoxAdapter(child: SizedBox(height: MySizes.spacing)),

          // //////// //
          // ACTIVITY //
          // //////// //

          SliverToBoxAdapter(
            child: VehicleActivityContainer(
              vehicleID: vehicleID,
            ),
          ),
        ],
      ),
    );
  }
}
