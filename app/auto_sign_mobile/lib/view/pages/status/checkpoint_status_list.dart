import 'package:auto_sign_mobile/controller/vehicle_controller.dart';
import 'package:auto_sign_mobile/model/vehicle/checkpoint.dart';
import 'package:auto_sign_mobile/model/vehicle/vehicle.dart';
import 'package:auto_sign_mobile/view/pages/status/checkpoint_status_container.dart';
import 'package:auto_sign_mobile/view/theme/data/my_sizes.dart';
import 'package:auto_sign_mobile/view/widgets/custom_stream_builder.dart';
import 'package:flutter/material.dart';

/// Widget that displays the [ConformanceStatus] for all of the [Checkpoint]s
/// withini a given [Vehicle].
///
/// A [CheckpointStatusContainer] is used to display the [ConformanceStatus] of
/// each [Checkpoint].
class CheckpointStatusList extends StatefulWidget {
  // MEMBER VARIABLES //
  final Vehicle vehicle; // vehicle the status is being displayed for

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  const CheckpointStatusList({
    super.key,
    required this.vehicle,
  });

  // //////////// //
  // CREATE STATE //
  // //////////// //

  @override
  State<CheckpointStatusList> createState() => _CheckpointStatusListState();
}

/// State class for [CheckpointStatusList].
class _CheckpointStatusListState extends State<CheckpointStatusList> {
  // STATE VARIABLES //
  late int expandedCheckpointIndex; // index of checkpoint currently expanded

  // ////////// //
  // INIT STATE //
  // ////////// //

  @override
  void initState() {
    super.initState();

    // initializing state
    expandedCheckpointIndex = -1; // -1 as no checkpoint expanded initially
  }

  // //////////// //
  // BUILD METHOD //
  // //////////// //

  @override
  Widget build(BuildContext context) {
    return CustomStreamBuilder<List<Checkpoint>>(
      stream: VehicleController.instance
          .getCheckpointsWhereVehicleIs(widget.vehicle.id),
      builder: (context, checkpoints) {
        return ListView.builder(
          shrinkWrap: true,
          physics: const NeverScrollableScrollPhysics(),
          itemCount: checkpoints.length,
          itemBuilder: ((context, index) {
            return Column(
              children: [
                CheckpointStatusContainer(
                  checkpoint: checkpoints[index],
                  isExpanded: expandedCheckpointIndex == index,
                  onExpanded: () {
                    setState(() {
                      if (expandedCheckpointIndex == index) {
                        expandedCheckpointIndex = -1;
                      } else {
                        expandedCheckpointIndex = index;
                      }
                    });
                  },
                ),
                if (index != checkpoints.length - 1)
                  const SizedBox(height: MySizes.spacing),
              ],
            );
          }),
        );
      },
    );
  }
}
