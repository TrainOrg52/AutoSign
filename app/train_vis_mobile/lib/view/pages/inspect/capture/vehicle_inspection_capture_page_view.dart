import 'package:flutter/material.dart';
import 'package:train_vis_mobile/view/pages/inspect/capture/checkpoint_inspection_capture_page_view.dart';

/// TODO
class VehicleInspectionCapturePageView extends StatelessWidget {
  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  /// TODO
  const VehicleInspectionCapturePageView({super.key});

  // //////////// //
  // BUILD METHOD //
  // //////////// //

  @override
  Widget build(BuildContext context) {
    // page view controller
    PageController pageController = PageController();

    return PageView(
      controller: pageController,
      physics: const NeverScrollableScrollPhysics(),
      children: const [
        // //////////////////// //
        // CHECKPOINT PAGE VIEW //
        // //////////////////// //

        CheckpointInspectionCapturePageView(),

        // TODO add pages for each checkpoint
      ],
    );
  }
}
