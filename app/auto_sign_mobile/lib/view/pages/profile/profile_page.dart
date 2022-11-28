import 'package:auto_sign_mobile/controller/vehicle_controller.dart';
import 'package:auto_sign_mobile/view/pages/profile/vehicle_action_container.dart';
import 'package:auto_sign_mobile/view/pages/profile/vehicle_activity_container.dart';
import 'package:auto_sign_mobile/view/pages/profile/vehicle_overview_container.dart';
import 'package:auto_sign_mobile/view/pages/profile/vehicle_status_overview_container.dart';
import 'package:auto_sign_mobile/view/theme/data/my_sizes.dart';
import 'package:auto_sign_mobile/view/theme/data/my_text_styles.dart';
import 'package:auto_sign_mobile/view/theme/widgets/my_icon_button.dart';
import 'package:auto_sign_mobile/view/widgets/custom_stream_builder.dart';
import 'package:auto_sign_mobile/view/widgets/padded_custom_scroll_view.dart';
import 'package:flutter/material.dart';

/// Page to display the profile of a train vehicle.
///
/// Displays an overview of information about the train, a description of the
/// train's current status, controls to allow for inspection and remediation,
/// and controls to view a log of activity on the train.
class ProfilePage extends StatelessWidget {
  // MEMBERS //
  final String vehicleID; // ID of vehicle

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

      body: SafeArea(
        child: CustomStreamBuilder(
          stream: VehicleController.instance.getVehicle(vehicleID),
          builder: (context, vehicle) {
            return PaddedCustomScrollView(
              slivers: [
                // //////////////// //
                // VEHICLE OVERVIEW //
                // //////////////// //

                SliverToBoxAdapter(
                  child: VehicleOverviewContainer(vehicle: vehicle),
                ),

                const SliverToBoxAdapter(
                    child: SizedBox(height: MySizes.spacing)),

                // /////////////////////////// //
                // CONFORMANCE STATUS OVERVIEW //
                // /////////////////////////// //

                SliverToBoxAdapter(
                  child: VehicleStatusOverviewContainer(
                    vehicle: vehicle,
                  ),
                ),

                const SliverToBoxAdapter(
                    child: SizedBox(height: MySizes.spacing)),

                // ////// //
                // ACTION //
                // ////// //

                SliverToBoxAdapter(
                  child: VehicleActionContainer(
                    vehicleID: vehicleID,
                  ),
                ),

                const SliverToBoxAdapter(
                    child: SizedBox(height: MySizes.spacing)),

                // //////// //
                // ACTIVITY //
                // //////// //

                SliverToBoxAdapter(
                  child: VehicleActivityContainer(
                    vehicleID: vehicleID,
                  ),
                ),
              ],
            );
          },
        ),
      ),
    );
  }
}
