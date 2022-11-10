import 'package:flutter/material.dart';
import 'package:train_vis_mobile/controller/vehicle_controller.dart';
import 'package:train_vis_mobile/model/inspection/checkpoint_inspection.dart';
import 'package:train_vis_mobile/model/vehicle/checkpoint.dart';
import 'package:train_vis_mobile/view/pages/inspect/capture/checkpoint_inspection_capture_page_view.dart';
import 'package:train_vis_mobile/view/widgets/custom_stream_builder.dart';

/// TODO
class VehicleInspectionCapturePageView extends StatefulWidget {
  // MEMBER VARIABLES //
  final String vehicleID;
  final Function(List<CheckpointInspection>) onCaptured;

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  /// TODO
  const VehicleInspectionCapturePageView({
    super.key,
    required this.vehicleID,
    required this.onCaptured,
  });

  // //////////// //
  // CREATE STATE //
  // //////////// //

  @override
  State<VehicleInspectionCapturePageView> createState() =>
      _VehicleInspectionCapturePageViewState();
}

/// TODO
class _VehicleInspectionCapturePageViewState
    extends State<VehicleInspectionCapturePageView> {
  // STATE VARIABLES //
  late PageController pageController; // controller for page view
  late int currentPage; // index of currently selected page
  late List<CheckpointInspection> checkpointInspections; // captured checkpoints

  // ////////// //
  // INIT STATE //
  // ////////// //

  @override
  void initState() {
    super.initState();

    // initializing state
    pageController = PageController();
    currentPage = 0;
    checkpointInspections = [];
  }

  // /////// //
  // DISPOSE //
  // /////// //

  @override
  void dispose() {
    super.dispose();

    // disposing of state
    pageController.dispose();
  }

  // //////////// //
  // BUILD METHOD //
  // //////////// //

  @override
  Widget build(BuildContext context) {
    return CustomStreamBuilder<List<Checkpoint>>(
      stream: VehicleController.instance
          .getCheckpointsWhereVehicleIs(widget.vehicleID),
      builder: (context, checkpoints) {
        return PageView(
          controller: pageController,
          physics: const NeverScrollableScrollPhysics(),
          children: [
            for (Checkpoint checkpoint in checkpoints)
              CheckpointInspectionCapturePageView(
                checkpoint: checkpoint,
                onCheckpointInspectionCaptured: (capturePath) {
                  // handling capture
                  _handleCheckpointInspectionCaptured(
                    checkpoints,
                    checkpoint,
                    capturePath,
                  );
                },
              )
          ],
        );
      },
    );
  }

  // ////////////// //
  // HELPER METHODS //
  // ////////////// //

  /// TODO
  void _handleCheckpointInspectionCaptured(
    List<Checkpoint> checkpoints,
    Checkpoint checkpoint,
    String capturePath,
  ) {
    // updating current page number
    currentPage = currentPage + 1;

    // creating new inspection checkpoint for capture
    checkpointInspections.add(
      CheckpointInspection.fromCheckpoint(
        checkpoint: checkpoint,
        capturePath: capturePath,
      ),
    );

    // checking if all checkpoints have been captured
    if (currentPage < checkpoints.length) {
      // not all checkpoints captured -> move to next checkpoint

      // navigating to next page
      pageController.nextPage(
        duration: const Duration(milliseconds: 500),
        curve: Curves.ease,
      );
    } else {
      // all checkpoints captured -> calling on inspection captured

      // calling on vehicle inspection captured
      widget.onCaptured(checkpointInspections);
    }
  }
}
