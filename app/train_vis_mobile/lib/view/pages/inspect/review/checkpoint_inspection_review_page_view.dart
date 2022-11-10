import 'dart:io';

import 'package:flutter/material.dart';
import 'package:train_vis_mobile/model/inspection/checkpoint_inspection.dart';
import 'package:train_vis_mobile/view/theme/data/my_sizes.dart';
import 'package:train_vis_mobile/view/theme/data/my_text_styles.dart';
import 'package:train_vis_mobile/view/theme/widgets/my_text_button.dart';
import 'package:train_vis_mobile/view/widgets/bordered_container.dart';
import 'package:train_vis_mobile/view/widgets/camera_container.dart';

/// TODO
class CheckpointInspectionReviewPageView extends StatelessWidget {
  // MEMBER VARIABLES //
  final CheckpointInspection checkpointInspection; // checkpoint being reviewed

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  const CheckpointInspectionReviewPageView({
    super.key,
    required this.checkpointInspection,
  });

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
          "Are you happy with this photo?",
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

              CameraContainer(
                onCaptured: (capturePath) {
                  // handling capture

                  // TODO
                },
              ),

              // ////// //
              // REVIEW //
              // ////// //

              _buildCheckpointInspectionReview(pageController),
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
        // /////////////////////////// //
        // CHECKPOINT INSPECTION IMAGE //
        // /////////////////////////// //

        const Spacer(),

        BorderedContainer(
          isDense: true,
          backgroundColor: Colors.transparent,
          padding: const EdgeInsets.all(MySizes.paddingValue),
          child: Image.asset(checkpointInspection.capturePath),
        ),

        const Spacer(),

        // /////// //
        // ACTIONS //
        // /////// //

        Row(
          children: [
            // ////// //
            // RETAKE //
            // ////// //

            MyTextButton.secondary(
              text: "Retake",
              onPressed: () {
                // navigating to the (re) capture page
                pageController.animateToPage(
                  1,
                  duration: const Duration(milliseconds: 500),
                  curve: Curves.ease,
                );
              },
            ),

            // /////// //
            // CONFIRM //
            // /////// //

            MyTextButton.secondary(
              text: "Confirm",
              onPressed: () {
                // navigating back to review page
                // TODO
              },
            )
          ],
        ),
      ],
    );
  }

  /// TODO
  Widget _buildCheckpointInspectionReview(PageController pageController) {
    return Column(
      children: [
        const Spacer(),

        // //////////////// //
        // CHECKPOINT IMAGE //
        // //////////////// //

        Expanded(
          flex: 12,
          child: BorderedContainer(
            isDense: true,
            backgroundColor: Colors.transparent,
            padding: const EdgeInsets.all(MySizes.paddingValue),
            child: Image.file(File(checkpointInspection.capturePath)),
          ),
        ),

        const Spacer(),

        // /////// //
        // ACTIONS //
        // /////// //

        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            // ////// //
            // RETAKE //
            // ////// //

            MyTextButton.secondary(
              text: "Retake",
              onPressed: () {
                // navigating back to the capture page
                pageController.animateToPage(
                  1,
                  duration: const Duration(milliseconds: 500),
                  curve: Curves.ease,
                );
              },
            ),

            const SizedBox(width: MySizes.spacing),

            // //// //
            // NEXT //
            // //// //

            MyTextButton.primary(
              text: "Next",
              onPressed: () {
                // submitting the checkpoint inspection
                // TODO
              },
            )
          ],
        ),
      ],
    );
  }
}
