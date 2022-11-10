import 'dart:io';

import 'package:flutter/material.dart';
import 'package:train_vis_mobile/model/inspection/checkpoint_inspection.dart';
import 'package:train_vis_mobile/view/theme/data/my_sizes.dart';
import 'package:train_vis_mobile/view/theme/data/my_text_styles.dart';
import 'package:train_vis_mobile/view/theme/widgets/my_text_button.dart';
import 'package:train_vis_mobile/view/widgets/bordered_container.dart';

/// TODO
class CheckpointInspectionReviewContainer extends StatefulWidget {
  // MEMBER VARIABLES //
  final CheckpointInspection checkpointInspection; // checkpoing being reviewed
  final Function() onReviewPressed; // call back when review is pressed

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  const CheckpointInspectionReviewContainer({
    super.key,
    required this.checkpointInspection,
    required this.onReviewPressed,
  });

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
            child: Image.file(File(widget.checkpointInspection.capturePath)),
          ),

          const SizedBox(width: MySizes.spacing),

          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // //////////////// //
                // CHECKPOINT TITLE //
                // //////////////// //

                Text(
                  widget.checkpointInspection.title,
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
              widget.onReviewPressed();
            },
          ),
        ],
      );
    }
  }
}
