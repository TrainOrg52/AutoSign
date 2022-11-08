import 'package:flutter/material.dart';
import 'package:train_vis_mobile/view/theme/data/my_sizes.dart';
import 'package:train_vis_mobile/view/theme/data/my_text_styles.dart';
import 'package:train_vis_mobile/view/theme/widgets/my_text_button.dart';
import 'package:train_vis_mobile/view/widgets/bordered_container.dart';

/// TODO
class CheckpointInspectionReviewContainer extends StatefulWidget {
  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  const CheckpointInspectionReviewContainer({super.key});

  // //////////// //
  // CREATE STATE //
  // //////////// //

  @override
  State<CheckpointInspectionReviewContainer> createState() =>
      _CheckpointInspectionReviewContainerState();
}

/// TODO
class _CheckpointInspectionReviewContainerState
    extends State<CheckpointInspectionReviewContainer> {
  // STATE VARIABLES //
  late bool isReviewed; // review state of the checkpoint.

  // THEME-ING //
  // sizing
  final double containerHeight = 100; // height of the container

  // ////////// //
  // INIT STATE //
  // ////////// //

  @override
  void initState() {
    super.initState();

    // initializing state
    isReviewed = false;
  }

  // //////////// //
  // BUILD METHOD //
  // //////////// //

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: containerHeight,
      child: Row(
        mainAxisAlignment: MainAxisAlignment.start,
        children: [
          // //////////////// //
          // CHECKPOINT IMAGE //
          // //////////////// //

          BorderedContainer(
            isDense: true,
            backgroundColor: Colors.transparent,
            padding: const EdgeInsets.all(MySizes.paddingValue / 2),
            child: Image.asset("resources/images/checkpoint 1.png"),
          ),

          const SizedBox(width: MySizes.spacing),

          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // //////////////// //
                // CHECKPOINT TITLE //
                // //////////////// //

                const Text(
                  "Entrance 1: Door",
                  style: MyTextStyles.headerText3,
                ),

                const Spacer(),

                // ////////////// //
                // REVIEW ACTIONS //
                // ////////////// //

                _buildActionsContainer(),
              ],
            ),
          ),
        ],
      ),
    );
  }

  // ////////////////////// //
  // HELPER BUILDER METHODS //
  // ////////////////////// //

  /// TODO
  Widget _buildActionsContainer() {
    // building based on review status of checkpoint
    if (isReviewed) {
      // checkpoint reviewed -> display checkpoint reviewed button

      // /////////////// //
      // REVIEWED BUTTON //
      // /////////////// //

      return MyTextButton.primary(
        text: "Reviewed",
        onPressed: () {
          // undoing the review confirmation
          setState(() {
            isReviewed = false;
          });
        },
      );
    } else {
      // checkpoint not reviewed -> display buttons to review or confirm

      return Row(
        children: [
          // ///////////// //
          // REVIEW BUTTON //
          // ///////////// //

          MyTextButton.secondary(
            text: "Review",
            onPressed: () {
              // performing review
              // TODO
            },
          ),

          const SizedBox(width: MySizes.spacing),

          // ////////////// //
          // CONFIRM BUTTON //
          // ////////////// //

          MyTextButton.secondary(
            text: "Confirm",
            onPressed: () {
              // conforming the inspection
              setState(() {
                isReviewed = true;
              });
            },
          ),
        ],
      );
    }
  }
}
