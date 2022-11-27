import 'package:auto_sign_mobile/controller/vehicle_controller.dart';
import 'package:auto_sign_mobile/model/vehicle/vehicle.dart';
import 'package:auto_sign_mobile/view/pages/status/checkpoint_status_list.dart';
import 'package:auto_sign_mobile/view/pages/status/vehicle_status_container.dart';
import 'package:auto_sign_mobile/view/theme/data/my_sizes.dart';
import 'package:auto_sign_mobile/view/theme/data/my_text_styles.dart';
import 'package:auto_sign_mobile/view/theme/widgets/my_icon_button.dart';
import 'package:auto_sign_mobile/view/widgets/custom_stream_builder.dart';
import 'package:auto_sign_mobile/view/widgets/padded_custom_scroll_view.dart';
import 'package:flutter/material.dart';

/// Page to display the status of a train vehicle.
///
/// Provides a breakdown of the conformance status of each of the train vehicle's
/// checkpoints. If the vehicle is non-confirming, an action button is also
/// provided to perform a remediation.
class StatusPage extends StatelessWidget {
  // MEMBERS //
  final String vehicleID; // ID of vehicle being displayed

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
      // /////// //
      // APP BAR //
      // /////// //

      appBar: AppBar(
        leading: MyIconButton.back(
          onPressed: () {
            Navigator.of(context).pop();
          },
        ),
        title: const Text("Status", style: MyTextStyles.headerText1),
      ),

      // //// //
      // BODY //
      // //// //

      body: CustomStreamBuilder<Vehicle>(
        stream: VehicleController.instance.getVehicle(vehicleID),
        builder: (context, vehicle) {
          return PaddedCustomScrollView(
            slivers: [
              // ////////////// //
              // VEHICLE STATUS //
              // ////////////// //

              SliverToBoxAdapter(
                child: VehicleStatusContainer(vehicle: vehicle),
              ),

              const SliverToBoxAdapter(
                  child: SizedBox(height: MySizes.spacing)),

              // ///////////////// //
              // CHECKPOINT STATUS //
              // ///////////////// //

              SliverToBoxAdapter(
                child: CheckpointStatusList(vehicle: vehicle),
              ),
            ],
          );
        },
      ),
    );
  }
}
