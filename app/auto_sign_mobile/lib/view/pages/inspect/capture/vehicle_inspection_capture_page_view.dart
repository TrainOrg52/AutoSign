import 'package:auto_sign_mobile/controller/vehicle_controller.dart';
import 'package:auto_sign_mobile/model/inspection/checkpoint_inspection.dart';
import 'package:auto_sign_mobile/model/vehicle/checkpoint.dart';
import 'package:auto_sign_mobile/view/pages/inspect/capture/checkpoint_inspection_capture_page_view.dart';
import 'package:auto_sign_mobile/view/widgets/custom_stream_builder.dart';
import 'package:flutter/material.dart';

/// A custom [PageView] for capturing an inspection of a [Vehicle].
///
/// The [PageView] consists of one page for each [Checkpoint] in the vehicle.
/// Each of these pages is a [CheckpointInspectionCapturePageView], which allows
/// for each [Checkpoint] in the [Vehicle] to be captured.
///
/// Each [CheckpointInspectionCapturePageView] returns a [CheckpointInspection]
/// object when the checkpoint is captured successfully. The list of
/// [CheckpointInspection]s for the [Vehicle] are maintained within this class.
///
/// The [VehicleInspectionCapturePageView] will iterate throught the [Checkpoint]s
/// for the [Vehicle], displaying a [CheckpointInspectionCapturePageView] and
/// gathering a [CheckpointInspection] for each. After all [CheckpointInspection]s
/// have been gathered, the [onVehicleInspectionCaptured] call back is run, with
/// the list of [CheckpointInspections] passed on.
class VehicleInspectionCapturePageView extends StatefulWidget {
  // MEMBER VARIABLES //
  final String vehicleID; // ID of vehicle being captured
  final Function(List<CheckpointInspection>)
      onVehicleInspectionCaptured; // callback when capture of vehicle complete

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

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

/// State class for [VehicleInspectionCapturePageView]
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

  /// Handles the capturing of a [CheckpointInspection] from the
  /// [CheckpointInspectionCapturePageView].
  ///
  /// The [CheckpointInspection] returned by the [CheckpointInspectionCapturePageView]
  /// is added to the class's list of [CheckpointInspection]s, and the next page
  /// is shown. The next page is either a [CheckpointInspectionCapturePageView] for
  /// the next [Checkpoint] in the [Vehicle], or the [onVehicleInspectionCaptured]
  /// callback is run with the gathered list of [VehicleInspection]s.
  void _handleCheckpointInspectionCaptured(
    List<Checkpoint> checkpoints,
    Checkpoint checkpoint,
    String capturePath,
  ) {
    // creating a new checkpoint inspection object
    CheckpointInspection checkpointInspection =
        CheckpointInspection.fromCheckpoint(
      checkpoint: checkpoint,
      capturePath: capturePath,
    );

    // adding the checkpoint inspection to the list of checkpoint inspections
    checkpointInspections.add(checkpointInspection);

    // checking if all checkpoints have been captured
    if (currentPage < checkpoints.length - 1) {
      // not all checkpoints captured -> move to next checkpoint

      // navigating to next page
      pageController.nextPage(
        duration: const Duration(milliseconds: 500),
        curve: Curves.ease,
      );

      // updating current page number (doesnt need to be in set state )
      currentPage = currentPage + 1;
    } else {
      // all checkpoints captured -> calling on inspection captured

      // calling on vehicle inspection captured
      widget.onVehicleInspectionCaptured(checkpointInspections);
    }
  }
}
