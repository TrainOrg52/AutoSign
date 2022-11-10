import 'dart:io';

import 'package:flutter/material.dart';
import 'package:train_vis_mobile/controller/vehicle_controller.dart';
import 'package:train_vis_mobile/model/vehicle/checkpoint.dart';
import 'package:train_vis_mobile/view/theme/data/my_sizes.dart';
import 'package:train_vis_mobile/view/theme/data/my_text_styles.dart';
import 'package:train_vis_mobile/view/theme/widgets/my_text_button.dart';
import 'package:train_vis_mobile/view/widgets/bordered_container.dart';
import 'package:train_vis_mobile/view/widgets/camera_container.dart';
import 'package:train_vis_mobile/view/widgets/custom_future_builder.dart';

/// TODO
class CheckpointInspectionCapturePageView extends StatefulWidget {
  // MEMBER VARIABLES //
  final Checkpoint checkpoint; // checkpoint being displayed
  final Function(String)
      onCheckpointInspectionCaptured; //  ran when checkpoint is captured.

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  const CheckpointInspectionCapturePageView({
    super.key,
    required this.checkpoint,
    required this.onCheckpointInspectionCaptured,
  });

  // //////////// //
  // CREATE STATE //
  // //////////// //

  @override
  State<CheckpointInspectionCapturePageView> createState() =>
      _CheckpointInspectionCapturePageViewState();
}

/// TODO
class _CheckpointInspectionCapturePageViewState
    extends State<CheckpointInspectionCapturePageView> {
  // STATE VARIABLES //
  late PageController pageController; // controller for page view
  late String capturePath; // photo data for the checkpoint

  // ////////// //
  // INIT STATE //
  // ////////// //

  @override
  void initState() {
    super.initState();

    // initializing state
    pageController = PageController();
    capturePath = "";
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
          widget.checkpoint.title,
          style: MyTextStyles.headerText1,
          textAlign: TextAlign.center,
        ),

        const SizedBox(height: MySizes.spacing),

        // ///////////////// //
        // CHECKPOINT PROMPT //
        // ///////////////// //

        Text(
          widget.checkpoint.prompt,
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

              _buildCheckpointInspectionInstructions(pageController),

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
  Widget _buildCheckpointInspectionInstructions(PageController pageController) {
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
          child: CustomFutureBuilder(
            future: VehicleController.instance.getCheckpointImageDownloadURL(
              widget.checkpoint.vehicleID,
              widget.checkpoint.id,
            ),
            builder: (context, downloadURL) {
              return Image.network(downloadURL);
            },
          ),
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
            child: Image.file(File(capturePath)),
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
                widget.onCheckpointInspectionCaptured(capturePath);
              },
            )
          ],
        ),
      ],
    );
  }

  // ////////////// //
  // HELPER METHODS //
  // ////////////// //

  /// TODO
  void _handleCaptured(String capturePath) {
    // updating photo data
    setState(() {
      this.capturePath = capturePath;
    });

    // navigating to review page
    pageController.animateToPage(
      2,
      duration: const Duration(milliseconds: 500),
      curve: Curves.ease,
    );
  }
}
