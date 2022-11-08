import 'package:flutter/material.dart';
import 'package:train_vis_mobile/view/theme/data/my_sizes.dart';
import 'package:train_vis_mobile/view/theme/data/my_text_styles.dart';
import 'package:train_vis_mobile/view/theme/widgets/my_text_button.dart';
import 'package:train_vis_mobile/view/widgets/bordered_container.dart';
import 'package:train_vis_mobile/view/widgets/camera_container.dart';

/// TODO
class CheckpointInspectionCapturePageView extends StatelessWidget {
  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  const CheckpointInspectionCapturePageView({super.key});

  // //////////// //
  // BUILD METHOD //
  // //////////// //

  @override
  Widget build(BuildContext context) {
    // page view controller
    PageController pageController = PageController();

    return Column(
      children: [
        // //////////////// //
        // CHECKPOINT TITLE //
        // //////////////// //
        const Text(
          "Capture: Entrance 1 Door",
          style: MyTextStyles.headerText1,
          textAlign: TextAlign.center,
        ),

        const SizedBox(height: MySizes.spacing),

        // ///////////////// //
        // CHECKPOINT PROMPT //
        // ///////////////// //

        const Text(
          "Please take a photo of the door within the first entrance",
          style: MyTextStyles.bodyText1,
          textAlign: TextAlign.center,
        ),

        const SizedBox(height: MySizes.spacing),

        // ////////////////////// //
        // INSTRUCTIONS + CAPTURE //
        // ////////////////////// //

        Expanded(
          child: PageView(
            controller: pageController,
            physics: const NeverScrollableScrollPhysics(),
            children: [
              // //////////// //
              // INSTRUCTIONS //
              // //////////// //

              _buildCheckpointInspectionPreview(pageController),

              // /////// //
              // CAPTURE //
              // /////// //

              CameraContainer(onCapture: () {}),
            ],
          ),
        ),
      ],
    );
  }

  // ////////////////////// //
  // HELPER BUILDER METHODS //
  // ////////////////////// //

  /// TODO
  Widget _buildCheckpointInspectionPreview(PageController pageController) {
    return Column(
      children: [
        // //////////////// //
        // CHECKPOINT IMAGE //
        // //////////////// //

        const Spacer(),

        BorderedContainer(
          isDense: true,
          backgroundColor: Colors.transparent,
          padding: const EdgeInsets.all(MySizes.paddingValue),
          child: Image.asset("resources/images/checkpoint 1.png"),
        ),

        const Spacer(),

        // //////////// //
        // READY BUTTON //
        // //////////// //

        MyTextButton.secondary(
          text: "I'm ready",
          onPressed: () {
            // navigating to the capture page
            pageController.animateToPage(
              1,
              duration: const Duration(milliseconds: 500),
              curve: Curves.ease,
            );
          },
        )
      ],
    );
  }
}
