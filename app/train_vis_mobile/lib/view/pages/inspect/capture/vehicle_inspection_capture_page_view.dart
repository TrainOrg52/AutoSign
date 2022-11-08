import 'package:flutter/material.dart';
import 'package:train_vis_mobile/controller/checkpoint_controller.dart';
import 'package:train_vis_mobile/model/inspection/checkpoint_inspection.dart';
import 'package:train_vis_mobile/model/vehicle/checkpoint.dart';
import 'package:train_vis_mobile/model/vehicle/vehicle.dart';
import 'package:train_vis_mobile/view/pages/inspect/capture/checkpoint_inspection_capture_page_view.dart';
import 'package:train_vis_mobile/view/widgets/custom_stream_builder.dart';

/// TODO
class VehicleInspectionCapturePageView extends StatefulWidget {
  // MEMBER VARIABLES //
  final Vehicle vehicle;
  final Function(List<CheckpointInspection>) onVehicleInspectionCaptured;

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  /// TODO
  const VehicleInspectionCapturePageView({
    super.key,
    required this.vehicle,
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
    return PageView(
      controller: pageController,
      physics: const NeverScrollableScrollPhysics(),
      children: _buildPages(),
    );
  }

  // ////////////////////// //
  // HELPER BUILDER METHODS //
  // ////////////////////// //

  /// TODO
  List<Widget> _buildPages() {
    // creating empty list of pages
    List<Widget> pages = [];

    // populating list of pages using vehicle object
    for (String checkpointID in widget.vehicle.checkpoints) {
      pages.add(
        CustomStreamBuilder<Checkpoint>(
          stream: CheckpointController.instance.getCheckpoint(checkpointID),
          builder: (context, checkpoint) {
            return CheckpointInspectionCapturePageView(
              checkpoint: checkpoint,
              onCheckpointInspectionCaptured: (capturePath) {
                // handling capture

                // updating current page number
                currentPage = currentPage + 1;

                // creating new inspection checkpoint for capture
                checkpointInspections.add(
                  CheckpointInspection.fromCheckpoint(
                    checkpoint: checkpoint,
                    vehicleInspectionID:
                        "inspectionWalkthroughID", // TODO add in real vehicle inspection id
                    capturePath: capturePath,
                  ),
                );

                // checking if all checkpoints have been captured
                if (currentPage < widget.vehicle.checkpoints.length) {
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
              },
            );
          },
        ),
      );
    }

    // returning list of pages
    return pages;
  }
}
