import 'package:auto_sign_mobile/controller/vehicle_controller.dart';
import 'package:auto_sign_mobile/model/enums/capture_type.dart';
import 'package:auto_sign_mobile/model/inspection/checkpoint_inspection.dart';
import 'package:auto_sign_mobile/view/theme/data/my_sizes.dart';
import 'package:auto_sign_mobile/view/theme/data/my_text_styles.dart';
import 'package:auto_sign_mobile/view/theme/widgets/my_text_button.dart';
import 'package:auto_sign_mobile/view/widgets/capture_preview.dart';
import 'package:auto_sign_mobile/view/widgets/custom_stream_builder.dart';
import 'package:flutter/material.dart';

/// A custom [Container] that displays an overview of a [CheckpointInspection]
/// within the main 'Review' page, and allows the user to review the capture.
///
/// If the user choses to review the [CheckpointInspection] by pressing the
/// 'Review' button, the [onReviwePressed] callback is run.
class CheckpointInspectionReviewContainer extends StatelessWidget {
  // MEMBER VARIABLES //
  final CheckpointInspection checkpointInspection; // checkpoing being reviewed
  final Function() onReviewPressed; // call back when review is pressed

  // THEME-ING //
  // sizing
  final double containerHeight = 100; // height of the container

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  const CheckpointInspectionReviewContainer({
    super.key,
    required this.checkpointInspection,
    required this.onReviewPressed,
  });

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
          // /////////////////// //
          // CHECKPOINT SHOWCASE //
          // /////////////////// //

          CustomStreamBuilder<String>(
            stream: VehicleController.instance.getCheckpointShowcaseDownloadURL(
              checkpointInspection.vehicleID,
              checkpointInspection.checkpointID,
            ),
            builder: (context, downloadURL) {
              return CapturePreview(
                captureType: CaptureType.photo,
                path: downloadURL,
                isNetworkURL: true,
              );
            },
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
                  checkpointInspection.title,
                  style: MyTextStyles.headerText3,
                ),

                const Spacer(),

                // ///////////// //
                // REVIEW BUTTON //
                // ///////////// //

                MyTextButton.secondary(
                  text: "Review",
                  onPressed: () {
                    // performing review
                    onReviewPressed();
                  },
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}