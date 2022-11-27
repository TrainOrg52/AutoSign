import 'package:flutter/material.dart';
import 'package:train_vis_mobile/controller/vehicle_controller.dart';
import 'package:train_vis_mobile/model/status/conformance_status.dart';
import 'package:train_vis_mobile/model/vehicle/vehicle.dart';
import 'package:train_vis_mobile/view/pages/status/checkpoint_status_list.dart';
import 'package:train_vis_mobile/view/pages/status/status_action_container.dart';
import 'package:train_vis_mobile/view/theme/data/my_sizes.dart';
import 'package:train_vis_mobile/view/theme/data/my_text_styles.dart';
import 'package:train_vis_mobile/view/theme/widgets/my_icon_button.dart';
import 'package:train_vis_mobile/view/widgets/custom_stream_builder.dart';
import 'package:train_vis_mobile/view/widgets/padded_custom_scroll_view.dart';

/// Page to display the status of a train vehicle.
///
/// Provides a breakdown of the conformance status of each of the train vehicle's
/// checkpoints.
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
              if (vehicle.conformanceStatus == ConformanceStatus.nonConforming)
                SliverToBoxAdapter(
                  child: StatusActionContainer(vehicle: vehicle),
                ),
              // ///////////// //
              // STATUS ACTION //
              // ///////////// //

              const SliverToBoxAdapter(
                  child: SizedBox(height: MySizes.spacing)),

              // /////////// //
              // CHECKPOINTS //
              // /////////// //

              SliverToBoxAdapter(
                child: CheckpointStatusList(checkpoints: vehicle.checkpoints),
              ),
            ],
          );
        },
      ),
    );
  }
}
