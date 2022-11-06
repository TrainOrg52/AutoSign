import 'package:flutter/material.dart';
import 'package:train_vis_mobile/view/pages/status/checkpoint_status_container.dart';
import 'package:train_vis_mobile/view/pages/status/status_action_container.dart';
import 'package:train_vis_mobile/view/theme/data/my_sizes.dart';
import 'package:train_vis_mobile/view/theme/data/my_text_styles.dart';
import 'package:train_vis_mobile/view/theme/widgets/my_icon_button.dart';
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

      body: PaddedCustomScrollView(
        slivers: [
          // ///////////// //
          // STATUS ACTION //
          // ///////////// //

          SliverToBoxAdapter(
            child: StatusActionContainer(vehicleID: vehicleID),
          ),

          const SliverToBoxAdapter(child: SizedBox(height: MySizes.spacing)),

          // /////////// //
          // CHECKPOINTS //
          // /////////// //

          const SliverToBoxAdapter(
            child: CheckpointStatusContainer(checkpointID: "checkpointID"),
          )
        ],
      ),
    );
  }
}
