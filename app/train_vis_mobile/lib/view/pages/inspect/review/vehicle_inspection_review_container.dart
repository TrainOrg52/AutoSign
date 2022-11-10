import 'package:flutter/material.dart';
import 'package:train_vis_mobile/model/inspection/checkpoint_inspection.dart';
import 'package:train_vis_mobile/view/pages/inspect/review/checkpoint_inspection_review_container.dart';
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
    );
  }
}
