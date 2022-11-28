import 'package:auto_sign_mobile/view/theme/data/my_colors.dart';
import 'package:auto_sign_mobile/view/theme/data/my_sizes.dart';
import 'package:auto_sign_mobile/view/theme/data/my_text_styles.dart';
import 'package:auto_sign_mobile/view/theme/widgets/my_text_button.dart';
import 'package:flutter/material.dart';

/// A custom [Container] to be shown to the user when they are submitting
/// a [VehilceInspection].
///
/// The container takes a [isSubmitted] parameter which represents the submission
/// state of the inspection.
///
/// While submitting, an indeterminant progress indicator is shown. When submitted,
/// an message is shown informing the user the inspection has completed, and
/// a button is present that allows the user to return to the [ProfilePage].
class VehicleInspectionSubmitContainer extends StatelessWidget {
  // MEMBER VARIABLES //
  final bool isSubmitted; // submission status of inspection

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  const VehicleInspectionSubmitContainer({
    super.key,
    required this.isSubmitted,
  });

  // //////////// //
  // BUILD METHOD //
  // //////////// //

  @override
  Widget build(BuildContext context) {
    // building based on submission status
    if (isSubmitted) {
      // inspection submitted -> build submitted container
      return _buildSubmittedContainer(context);
    } else {
      // inspection not submitted -> build submitting container
      return _buildSubmittingContainer();
    }
  }

  // ////////////////////// //
  // HELPER BUILDER METHODS //
  // ////////////////////// //

  /// Builds the container for when the inspection is being submitted.
  ///
  /// This container contains a title, and an indeterminant progress indicator.
  Widget _buildSubmittingContainer() {
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

  /// Builds the container shown when the inspection has submitted.
  ///
  /// This container includes a title message, and a button to return to the
  /// [ProfilePage].
  Widget _buildSubmittedContainer(BuildContext context) {
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
            Navigator.of(context).pop();
          },
        ),
      ],
    );
  }
}
