import 'package:flutter/material.dart';
import 'package:train_vis_mobile/controller/checkpoint_controller.dart';
import 'package:train_vis_mobile/model/vehicle/checkpoint.dart';
import 'package:train_vis_mobile/view/pages/status/checkpoint_status_container.dart';
import 'package:train_vis_mobile/view/theme/data/my_sizes.dart';
import 'package:train_vis_mobile/view/widgets/custom_stream_builder.dart';

/// Widget that holds a list of [CheckpointStatusContainer] widgets for all of the
/// checkpoints within a given vehicle.
class CheckpointStatusList extends StatefulWidget {
  // MEMBER VARIABLES //
  final List<String> checkpoints; // IDs of the checkpoints being displayed

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  const CheckpointStatusList({
    super.key,
    required this.checkpoints,
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
    return ListView.builder(
      shrinkWrap: true,
      itemCount: widget.checkpoints.length,
      itemBuilder: ((context, index) {
        return Column(
          children: [
            CustomStreamBuilder<Checkpoint>(
              stream: CheckpointController.instance
                  .getCheckpoint(widget.checkpoints[index]),
              builder: (context, checkpoint) {
                return CheckpointStatusContainer(
                  checkpoint: checkpoint,
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
                );
              },
            ),
            if (index != widget.checkpoints.length - 1)
              const SizedBox(height: MySizes.spacing),
          ],
        );
      }),
    );
  }
}
