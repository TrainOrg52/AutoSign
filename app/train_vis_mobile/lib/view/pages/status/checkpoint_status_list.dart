import 'package:flutter/material.dart';
import 'package:train_vis_mobile/controller/vehicle_controller.dart';
import 'package:train_vis_mobile/model/vehicle/checkpoint.dart';
import 'package:train_vis_mobile/model/vehicle/vehicle.dart';
import 'package:train_vis_mobile/view/pages/status/checkpoint_status_container.dart';
import 'package:train_vis_mobile/view/theme/data/my_sizes.dart';
import 'package:train_vis_mobile/view/widgets/custom_stream_builder.dart';

/// Widget that holds a list of [CheckpointStatusContainer] widgets for all of the
/// checkpoints within a given vehicle.
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
        });
  }
}
