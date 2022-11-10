import 'package:flutter/material.dart';
import 'package:train_vis_mobile/controller/vehicle_controller.dart';
import 'package:train_vis_mobile/model/inspection/checkpoint_inspection.dart';
import 'package:train_vis_mobile/model/vehicle/checkpoint.dart';
import 'package:train_vis_mobile/view/pages/inspect/capture/checkpoint_inspection_capture_page_view.dart';
import 'package:train_vis_mobile/view/widgets/custom_stream_builder.dart';

/// A custom [PageView] for capturing an inspection of a [Vehicle].
///
/// The [PageView] consists of one page for each [Checkpoint] in the vehicle.
/// Each of these pages is a [CheckpointInspectionCapturePageView], which allow
/// for inspection of the each vehicle's [Checkpoint]s to be captured.
///
/// Each [CheckpointInspectionPageView]
class VehicleInspectionCapturePageView extends StatefulWidget {
  // MEMBER VARIABLES //
  final String vehicleID; // TODO
  final Function(List<CheckpointInspection>)
      onVehicleInspectionCaptured; // TODO

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  /// TODO
  const VehicleInspectionCapturePageView({
    super.key,
    required this.vehicleID,
    required this.onVehicleInspectionCaptured,
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
                onCheckpointInspectionCaptured: (checkpointInspection) {
                  // handling capture
                  _handleCheckpointInspectionCaptured(
                    checkpoints,
                    checkpointInspection,
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
    CheckpointInspection checkpointInspection,
  ) {
    // updating current page number
    currentPage = currentPage + 1;

    // adding the checkpoint inspection to the list of checkpoint inspections
    checkpointInspections.add(checkpointInspection);

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
      widget.onVehicleInspectionCaptured(checkpointInspections);
    }
  }
}
