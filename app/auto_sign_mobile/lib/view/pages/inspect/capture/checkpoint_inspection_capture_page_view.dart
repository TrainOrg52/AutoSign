import 'package:auto_sign_mobile/controller/vehicle_controller.dart';
import 'package:auto_sign_mobile/model/vehicle/checkpoint.dart';
import 'package:auto_sign_mobile/view/theme/data/my_sizes.dart';
import 'package:auto_sign_mobile/view/theme/data/my_text_styles.dart';
import 'package:auto_sign_mobile/view/theme/widgets/my_text_button.dart';
import 'package:auto_sign_mobile/view/widgets/camera_container.dart';
import 'package:auto_sign_mobile/view/widgets/capture_preview.dart';
import 'package:auto_sign_mobile/view/widgets/custom_stream_builder.dart';
import 'package:flutter/material.dart';

/// A custom [PageView] for capturing an inspection of a [Checkpoint]. The title
/// of the checkpoint and its prompt are displayed as a title above the view
/// which consists of three pages that allow the user to preview, capture and review
/// the inspection. Once reviewed, an [String] is returned using the
/// [onCheckpointInspectionCaptured] method that contains the path to the image
/// captured of the [Checkpoint].
///
/// Page 1 - A page to display a preview for the checkpoint. This is an
/// example image of the checkpoint (i.e., the gold standard).
///
/// Page 2 - A page to allow the user to capture an image of the checkpoint. This
/// includes a camera preview.
///
/// Page 3 - A page to display the image of the checkpoint the user captured in
/// page 2, and retake/confirm this capture.
class CheckpointInspectionCapturePageView extends StatefulWidget {
  // MEMBER VARIABLES //
  final Checkpoint checkpoint; // checkpoint being displayed
  final Function(String)
      onCheckpointInspectionCaptured; // callback ran when checkpoint is captured.

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

/// State class for [CheckpointInspectionCapturePageView].
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

              _buildPreviewPage(),

              // /////// //
              // CAPTURE //
              // /////// //

              _buildCapturePage(),
            ],
          ),
        ),
      ],
    );
  }

  // ////////////////////// //
  // HELPER BUILDER METHODS //
  // ////////////////////// //

  /// Builds the page that shows the user a preview for the checkpoint.
  ///
  /// This is the gold-standard image of the checkpoint.
  Widget _buildPreviewPage() {
    return Column(
      children: [
        // ///////////// //
        // PREVIEW IMAGE //
        // ///////////// //

        const Spacer(),

        Expanded(
          flex: 15,
          child: CustomStreamBuilder(
            stream: VehicleController.instance.getCheckpointDemoDownloadURL(
              widget.checkpoint.vehicleID,
              widget.checkpoint.id,
              widget.checkpoint.captureType,
            ),
            builder: (context, downloadURL) {
              return CapturePreview(
                captureType: widget.checkpoint.captureType,
                path: downloadURL,
                isNetworkURL: true,
              );
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

  /// Builds the page that allows the user to capture an image of the checkpoing.
  Widget _buildCapturePage() {
    return CameraContainer(
      captureType: widget.checkpoint.captureType,
      onCaptured: (capturePath) {
        // handling capture

        // updating photo data
        setState(() {
          this.capturePath = capturePath;
        });

        // submitting the checkpoint inspection to the callback
        widget.onCheckpointInspectionCaptured(
          capturePath,
        );
      },
    );
  }
}