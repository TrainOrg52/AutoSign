import 'package:flutter/material.dart';
import 'package:train_vis_mobile/model/inspection/checkpoint_inspection.dart';
import 'package:train_vis_mobile/view/pages/inspect/review/checkpoint_inspection_review_container.dart';
import 'package:train_vis_mobile/view/pages/inspect/review/checkpoint_inspection_review_page_view.dart';
import 'package:train_vis_mobile/view/theme/data/my_sizes.dart';
import 'package:train_vis_mobile/view/theme/data/my_text_styles.dart';
import 'package:train_vis_mobile/view/theme/widgets/my_text_button.dart';

/// TODO
class VehicleInspectionReviewContainer extends StatefulWidget {
  // MEMBER VARIABLES //
  final List<CheckpointInspection> checkpointInspections; // TODO
  final Function(List<CheckpointInspection>) onSubmitted; // TODO

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  const VehicleInspectionReviewContainer({
    super.key,
    required this.checkpointInspections,
    required this.onSubmitted,
  });

  // //////////// //
  // CREATE STATE //
  // //////////// //

  @override
  State<VehicleInspectionReviewContainer> createState() =>
      _VehicleInspectionReviewContainerState();
}

/// TODO
class _VehicleInspectionReviewContainerState
    extends State<VehicleInspectionReviewContainer> {
  // STATE VARIABLES //
  late PageController pageController;
  late CheckpointInspection
      reviewCheckpointInspection; // checkpoint inspection being reviewed

  // ////////// //
  // INIT STATE //
  // ////////// //

  @override
  void initState() {
    super.initState();

    // initializing state
    pageController = PageController(initialPage: 1); // initial = review page
    reviewCheckpointInspection = CheckpointInspection();
  }

  // //////////// //
  // BUILD METHOD //
  // //////////// //

  @override
  Widget build(BuildContext context) {
    return PageView(
      controller: pageController,
      children: [
        // ////////////////////// //
        // CHECKPOINT REVIEW PAGE //
        // ////////////////////// //

        CheckpointInspectionReviewPageView(
          checkpointInspection: reviewCheckpointInspection,
          onConfirmed: () {
            // navigating to main review page
            pageController.nextPage(
              duration: const Duration(milliseconds: 500),
              curve: Curves.ease,
            );
          },
          onReCaptured: (checkpointInspection) {
            // updating the checkpoint inspection
            // TODO

            // navigating to main review page
            pageController.nextPage(
              duration: const Duration(milliseconds: 500),
              curve: Curves.ease,
            );
          },
        ),

        // //////////////// //
        // MAIN REVIEW PAGE //
        // //////////////// //

        Column(
          children: [
            // ///// //
            // TITLE //
            // ///// //
            const Text(
              "Review",
              style: MyTextStyles.headerText1,
              textAlign: TextAlign.center,
            ),

            const SizedBox(height: MySizes.spacing),

            // ////// //
            // PROMPT //
            // ////// //

            const Text(
              "Please review and submit your inspection",
              style: MyTextStyles.bodyText1,
              textAlign: TextAlign.center,
            ),

            const SizedBox(height: MySizes.spacing * 2),

            // //////////////////////////// //
            // CHECKPOINT REVIEW CONTAINERS //
            // //////////////////////////// //

            ListView.builder(
              shrinkWrap: true,
              itemCount: widget.checkpointInspections.length,
              itemBuilder: (context, index) {
                return Column(
                  children: [
                    CheckpointInspectionReviewContainer(
                      checkpointInspection: widget.checkpointInspections[index],
                      onReviewPressed: () {
                        // handling review
                        _onReviewPressed(widget.checkpointInspections[index]);
                      },
                    ),
                    if (index != widget.checkpointInspections.length - 1)
                      const SizedBox(height: MySizes.spacing * 2)
                  ],
                );
              },
            ),

            const SizedBox(height: MySizes.spacing * 2),

            // ///////////// //
            // SUBMIT BUTTON //
            // ///////////// //

            MyTextButton.primary(
              text: "Submit",
              onPressed: () {
                // handling the submit
                widget.onSubmitted(widget.checkpointInspections);
              },
            ),
          ],
        ),
      ],
    );
  }

  // ////////////// //
  // HELPER METHODS //
  // ////////////// //

  /// TODO
  void _onReviewPressed(CheckpointInspection checkpointInspection) {
    // updating the review checkpoint inspection
    setState(() {
      reviewCheckpointInspection = checkpointInspection;
    });

    // navigating to checkpoint inspection review page
    pageController.animateToPage(
      0,
      duration: const Duration(milliseconds: 500),
      curve: Curves.ease,
    );
  }
}
