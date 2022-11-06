import 'package:flutter/material.dart';
import 'package:train_vis_mobile/view/pages/status/checkpoint_status_container.dart';
import 'package:train_vis_mobile/view/theme/data/my_sizes.dart';

/// Widget that holds a list of [CheckpointStatusContainer] widgets for all of the
/// checkpoints within a given walkthrough.
class WalkthroughStatusList extends StatefulWidget {
  // MEMBER VARIABLES //
  final String vehicleID; // ID of vehicle being displayed.

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  const WalkthroughStatusList({
    super.key,
    required this.vehicleID,
  });

  // //////////// //
  // CREATE STATE //
  // //////////// //

  @override
  State<WalkthroughStatusList> createState() => _WalkthroughStatusListState();
}

/// State class for [WalkthroughStatusList].
class _WalkthroughStatusListState extends State<WalkthroughStatusList> {
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
    return Column(
      children: [
        CheckpointStatusContainer(
          checkpointID: "checkpointID",
          isExpanded: expandedCheckpointIndex == 0,
          onExpanded: () {
            setState(() {
              if (expandedCheckpointIndex == 0) {
                expandedCheckpointIndex = -1;
              } else {
                expandedCheckpointIndex = 0;
              }
            });
          },
        ),
        const SizedBox(height: MySizes.spacing),
        CheckpointStatusContainer(
          checkpointID: "checkpointID",
          isExpanded: expandedCheckpointIndex == 1,
          onExpanded: () {
            setState(() {
              if (expandedCheckpointIndex == 1) {
                expandedCheckpointIndex = -1;
              } else {
                expandedCheckpointIndex = 1;
              }
            });
          },
        ),
        const SizedBox(height: MySizes.spacing),
        CheckpointStatusContainer(
          checkpointID: "checkpointID",
          isExpanded: expandedCheckpointIndex == 2,
          onExpanded: () {
            setState(() {
              if (expandedCheckpointIndex == 2) {
                expandedCheckpointIndex = -1;
              } else {
                expandedCheckpointIndex = 2;
              }
            });
          },
        ),
      ],
    );
  }
}
