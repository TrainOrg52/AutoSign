import 'dart:io';

import 'package:flutter/material.dart';
import 'package:train_vis_mobile/model/inspection/checkpoint_inspection.dart';
import 'package:train_vis_mobile/view/theme/data/my_sizes.dart';
import 'package:train_vis_mobile/view/theme/data/my_text_styles.dart';
import 'package:train_vis_mobile/view/theme/widgets/my_text_button.dart';
import 'package:train_vis_mobile/view/widgets/bordered_container.dart';
import 'package:train_vis_mobile/view/widgets/camera_container.dart';

/// TODO
class CheckpointInspectionReviewPageView extends StatefulWidget {
  // MEMBER VARIABLES //
  final CheckpointInspection checkpointInspection; // checkpoint being reviewed
  final Function(CheckpointInspection)
      onReCaptured; // called when checkpoint re-captured
  final Function() onConfirmed; // called when CI confirmed

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  const CheckpointInspectionReviewPageView({
    super.key,
    required this.checkpointInspection,
    required this.onReCaptured,
    required this.onConfirmed,
  });

  @override
  State<CheckpointInspectionReviewPageView> createState() =>
      _CheckpointInspectionReviewPageViewState();
}

/// TODO
class _CheckpointInspectionReviewPageViewState
    extends State<CheckpointInspectionReviewPageView> {
  // STATE VARIABLES //
  late PageController pageController; // TODO
  late CheckpointInspection checkpointInspection; // TODO

  // ////////// //
  // INIT STATE //
  // ////////// //

  @override
  void initState() {
    super.initState();

    // initializing state
    pageController = PageController();
    checkpointInspection = widget.checkpointInspection;
  }

  // //////////// //
  // BUILD METHOD //
  // //////////// //

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        // //////////////// //
        // CHECKPOINT TITLE //
        // //////////////// //
        Text(
          checkpointInspection.title,
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

              _buildInstructionsContainer(pageController),

              // /////// //
              // CAPTURE //
              // /////// //

              CameraContainer(
                onCaptured: (capturePath) {
                  // handling capture

                  _handleCaptured(capturePath);
                },
              ),

              // ////// //
              // REVIEW //
              // ////// //

              _buildReviewContainer(pageController),
            ],
          ),
        ),
      ],
    );
  }

  /// TODO
  Widget _buildInstructionsContainer(PageController pageController) {
    return Column(
      children: [
        // /////////////////////////// //
        // CHECKPOINT INSPECTION IMAGE //
        // /////////////////////////// //

        const Spacer(),

        Expanded(
          flex: 12,
          child: BorderedContainer(
            isDense: true,
            backgroundColor: Colors.transparent,
            padding: const EdgeInsets.all(MySizes.paddingValue),
            child: Image.asset(checkpointInspection.capturePath),
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
                // navigating to the (re) capture page
                pageController.animateToPage(
                  1,
                  duration: const Duration(milliseconds: 500),
                  curve: Curves.ease,
                );
              },
            ),

            const SizedBox(width: MySizes.spacing),

            // /////// //
            // CONFIRM //
            // /////// //

            MyTextButton.secondary(
              text: "Confirm",
              onPressed: () {
                // handling the confirm (navigating back to main review page)
                widget.onConfirmed();
              },
            )
          ],
        ),
      ],
    );
  }

  /// TODO
  Widget _buildReviewContainer(PageController pageController) {
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

            // /////// //
            // CONFIRM //
            // /////// //

            MyTextButton.primary(
              text: "Confirm",
              onPressed: () {
                // handling the confirm
                widget.onReCaptured(checkpointInspection);
              },
            )
          ],
        ),
      ],
    );
  }

  /// TODO
  void _handleCaptured(String capturePath) {
    // updating checkpoint inspection
    setState(() {
      checkpointInspection.capturePath = capturePath;
    });

    // navigating to review page
    pageController.animateToPage(
      2,
      duration: const Duration(milliseconds: 500),
      curve: Curves.ease,
    );
  }
}
