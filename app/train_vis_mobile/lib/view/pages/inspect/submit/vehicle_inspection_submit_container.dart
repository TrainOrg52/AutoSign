import 'package:flutter/material.dart';
import 'package:train_vis_mobile/view/theme/data/my_colors.dart';
import 'package:train_vis_mobile/view/theme/data/my_sizes.dart';
import 'package:train_vis_mobile/view/theme/data/my_text_styles.dart';
import 'package:train_vis_mobile/view/theme/widgets/my_text_button.dart';

/// TODO
class VehicleInspectionSubmitContainer extends StatefulWidget {
  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  const VehicleInspectionSubmitContainer({super.key});

  // //////////// //
  // CREATE STATE //
  // //////////// //

  @override
  State<VehicleInspectionSubmitContainer> createState() =>
      _VehicleInspectionSubmitContainerState();
}

/// TODO
class _VehicleInspectionSubmitContainerState
    extends State<VehicleInspectionSubmitContainer> {
  // THEME-ING //
  // sizes

  // //////////// //
  // BUILD METHOD //
  // //////////// //

  @override
  Widget build(BuildContext context) {
    return _buildInspectionSubmittingContainer();
  }

  // ////////////////////// //
  // HELPER BUILDER METHODS //
  // ////////////////////// //

  /// TODO
  Widget _buildInspectionSubmittingContainer() {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: const [
        // ///// //
        // TITLE //
        // ///// //

        Text(
          "Submitting Inspection",
          style: MyTextStyles.headerText1,
          textAlign: TextAlign.center,
        ),

        SizedBox(height: MySizes.spacing),

        // ////// //
        // PROMPT //
        // ////// //

        Text(
          "Please wait for your inspection to be uploaded",
          style: MyTextStyles.bodyText1,
          textAlign: TextAlign.center,
        ),

        SizedBox(height: MySizes.spacing * 3),

        // ////////////////// //
        // PROGRESS INDICATOR //
        // ////////////////// //

        SizedBox(
          height: 45,
          width: 45,
          child: CircularProgressIndicator(
            color: MyColors.primaryAccent,
            strokeWidth: 5,
          ),
        ),
      ],
    );
  }

  /// TODO
  Widget _buildInspectionSubmittedContainer() {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        // ///// //
        // TITLE //
        // ///// //

        const Text(
          "Inspection Complete",
          style: MyTextStyles.headerText1,
          textAlign: TextAlign.center,
        ),

        const SizedBox(height: MySizes.spacing),

        // ////// //
        // PROMPT //
        // ////// //

        const Text(
          "Your inspection was successfully uploaded",
          style: MyTextStyles.bodyText1,
          textAlign: TextAlign.center,
        ),

        const SizedBox(height: MySizes.spacing * 3),

        // ///////////// //
        // FINISH BUTTON //
        // ///////////// //

        MyTextButton.primary(
          text: "Finish",
          onPressed: () {
            // navigating back to home screen
            // TODO
          },
        ),
      ],
    );
  }
}
