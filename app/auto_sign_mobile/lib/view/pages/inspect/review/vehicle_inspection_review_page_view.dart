import 'package:auto_sign_mobile/model/inspection/checkpoint_inspection.dart';
import 'package:auto_sign_mobile/view/pages/inspect/review/checkpoint_inspection_review_container.dart';
import 'package:auto_sign_mobile/view/pages/inspect/review/checkpoint_inspection_review_page_view.dart';
import 'package:auto_sign_mobile/view/theme/data/my_sizes.dart';
import 'package:auto_sign_mobile/view/theme/data/my_text_styles.dart';
import 'package:auto_sign_mobile/view/theme/widgets/my_text_button.dart';
import 'package:flutter/material.dart';

/// A custom [PageView] for reviewing a [VehicleInspection]. It has two pages, with
/// Page 2 being shown by default.
///
/// Page 2: The main review page, which contains a [CheckpointInspectionRewviewContainer]
/// for each [Checkpoint] in the vehicle, and a button to submit the inspection.
///
/// Page 1: A [CheckpointInspectionReviewPageView], which is displayed for a given
/// [CheckpointInspection] when the user reviews it.
class VehicleInspectionReviewPageView extends StatefulWidget {
  // MEMBER VARIABLES //
  final List<CheckpointInspection>
      checkpointInspections; // the list of checkpoint inspections being reviewed
  final Function(List<CheckpointInspection>)
      onReviewed; // call back for when the user submits the inspections

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  const VehicleInspectionReviewPageView({
    super.key,
    required this.checkpointInspections,
    required this.onReviewed,
  });

  // //////////// //
  // CREATE STATE //
  // //////////// //

  @override
  State<VehicleInspectionReviewPageView> createState() =>
      _VehicleInspectionReviewPageViewState();
}

/// State class for [VehicleInspectionReviewPageView].
class _VehicleInspectionReviewPageViewState
    extends State<VehicleInspectionReviewPageView> {
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
      physics: const NeverScrollableScrollPhysics(),
      children: [
        // ////////////////////// //
        // CHECKPOINT REVIEW PAGE //
        // ////////////////////// //

        _buildCheckpointInspectionReviewPage(),

        // //////////////// //
        // MAIN REVIEW PAGE //
        // //////////////// //

        _buildVehicleInspectionReviewPage(),
      ],
    );
  }

  // ////////////////////// //
  // HELPER BUILDER METHODS //
  // ////////////////////// //

  /// Builds the page for reviewing a single [CheckpointInspection].
  Widget _buildCheckpointInspectionReviewPage() {
    return CheckpointInspectionReviewPageView(
      checkpointInspection: reviewCheckpointInspection,
      onConfirmed: () {
        // navigating to main review page
        pageController.nextPage(
          duration: const Duration(milliseconds: 500),
          curve: Curves.ease,
        );
      },
      onReCaptured: (capturePath) {
        // updating the checkpoint inspection
        reviewCheckpointInspection.capturePath = capturePath;

        // navigating to main review page
        pageController.nextPage(
          duration: const Duration(milliseconds: 500),
          curve: Curves.ease,
        );
      },
    );
  }

  /// Builds the page for reviewing the whole [VehicleInspection].
  Widget _buildVehicleInspectionReviewPage() {
    return ListView(
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
          physics: const NeverScrollableScrollPhysics(),
          itemCount: widget.checkpointInspections.length,
          itemBuilder: (context, index) {
            return Column(
              children: [
                CheckpointInspectionReviewContainer(
                  checkpointInspection: widget.checkpointInspections[index],
                  onReviewPressed: () {
                    // handling review

                    // updating the review checkpoint inspection
                    setState(() {
                      reviewCheckpointInspection =
                          widget.checkpointInspections[index];
                    });

                    // navigating to checkpoint inspection review page
                    pageController.animateToPage(
                      0,
                      duration: const Duration(milliseconds: 500),
                      curve: Curves.ease,
                    );
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
            widget.onReviewed(widget.checkpointInspections);
          },
        ),
      ],
    );
  }
}
