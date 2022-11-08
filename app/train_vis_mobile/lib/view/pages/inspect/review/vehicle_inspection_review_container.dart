import 'package:flutter/material.dart';
import 'package:train_vis_mobile/view/pages/inspect/review/checkpoint_inspection_review_container.dart';
import 'package:train_vis_mobile/view/theme/data/my_sizes.dart';
import 'package:train_vis_mobile/view/theme/data/my_text_styles.dart';
import 'package:train_vis_mobile/view/theme/widgets/my_text_button.dart';

/// TODO
class VehicleInspectionReviewContainer extends StatefulWidget {
  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  const VehicleInspectionReviewContainer({super.key});

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
  // TODO

  // //////////// //
  // BUILD METHOD //
  // //////////// //

  @override
  Widget build(BuildContext context) {
    return Column(
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

        const SizedBox(height: MySizes.spacing),

        // //////////////////////////// //
        // CHECKPOINT REVIEW CONTAINERS //
        // //////////////////////////// //

        const CheckpointInspectionReviewContainer(),

        const SizedBox(height: MySizes.spacing),

        // TODO add more review containers

        // ///////////// //
        // SUBMIT BUTTON //
        // ///////////// //

        MyTextButton.primary(
          text: "Submit",
          onPressed: () {
            // submitting the inspection
            // TODO
          },
        ),
      ],
    );
  }
}
