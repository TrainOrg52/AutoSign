import 'dart:io';

import 'package:auto_sign_mobile/model/inspection/checkpoint_inspection.dart';
import 'package:auto_sign_mobile/view/theme/data/my_sizes.dart';
import 'package:auto_sign_mobile/view/theme/data/my_text_styles.dart';
import 'package:auto_sign_mobile/view/theme/widgets/my_text_button.dart';
import 'package:auto_sign_mobile/view/widgets/bordered_container.dart';
import 'package:auto_sign_mobile/view/widgets/camera_container.dart';
import 'package:flutter/material.dart';

/// A custom [PageView] for reviewing a single [CheckpointInspection]. The
/// [PageView] consists of three pages:
///
/// Page 1: A page to preview the current capture of the [CheckpointInspection].
/// From this page, the user can chose to 're-take' the inspection, which takes
/// them to Page 2, or confirm thee inspection, which takes theme back to the
/// main review page.
///
/// Page 2: A page to re-capture an image of the [CheckpointInspection]. Following
/// the capture, the user is taken to Page 3.
///
/// Page 3: A page to review the re-capture of the [CheckpointInspection]. The user
/// can chose to either re-take the image which takes them back to Page 2, or accept
/// the re-capture.
class CheckpointInspectionReviewPageView extends StatefulWidget {
  // MEMBER VARIABLES //
  final CheckpointInspection checkpointInspection; // checkpoint being reviewed
  final Function(String) onReCaptured; // called when checkpoint is re-captured
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

/// State class for [_CheckpointInspectionReviewPageViewState].
class _CheckpointInspectionReviewPageViewState
    extends State<CheckpointInspectionReviewPageView> {
  // STATE VARIABLES //
  late PageController pageController; // page controller
  late String capturePath; // capture path for checkpoint inspection

  // ////////// //
  // INIT STATE //
  // ////////// //

  @override
  void initState() {
    super.initState();

    // initializing state
    pageController = PageController();
    capturePath = widget.checkpointInspection.capturePath;
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
          widget.checkpointInspection.title,
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
              // /////// //
              // PREVIEW //
              // /////// //

              _buildPreviewPage(),

              // /////// //
              // CAPTURE //
              // /////// //

              _buildCapturePage(),

              // ////// //
              // REVIEW //
              // ////// //

              _buildeviewPage(),
            ],
          ),
        ),
      ],
    );
  }

  /// Builds a page that allows the user to preview their current capture for
  /// the [CheckpointInspection], and either re-take or confirm this capture.
  Widget _buildPreviewPage() {
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
            child: Image.asset(capturePath),
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

            MyTextButton.primary(
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

  /// Builds a page that allows for the user to re-capture the image for the
  /// [CheckpointInspection].
  Widget _buildCapturePage() {
    return CameraContainer(
      onCaptured: (capturePath) {
        // handling capture

        // updating checkpoint inspection
        setState(() {
          this.capturePath = capturePath;
        });

        // navigating to review page
        pageController.animateToPage(
          2,
          duration: const Duration(milliseconds: 500),
          curve: Curves.ease,
        );
      },
    );
  }

  /// Builds a page that allows the user to review the image they have re-captured
  /// for the checkpoint inspection.
  Widget _buildeviewPage() {
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

            // /////// //
            // CONFIRM //
            // /////// //

            MyTextButton.primary(
              text: "Confirm",
              onPressed: () {
                // handling the confirm
                widget.onReCaptured(capturePath);
              },
            )
          ],
        ),
      ],
    );
  }
}
