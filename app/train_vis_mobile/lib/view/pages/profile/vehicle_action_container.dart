import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import 'package:go_router/go_router.dart';
import 'package:train_vis_mobile/view/routes/routes.dart';
import 'package:train_vis_mobile/view/theme/data/my_colors.dart';
import 'package:train_vis_mobile/view/theme/data/my_sizes.dart';
import 'package:train_vis_mobile/view/theme/data/my_text_styles.dart';
import 'package:train_vis_mobile/view/widgets/confirmation_dialog.dart';

/// A custom [Container] that displays actions that can be performed on a given
/// vehicle.
///
/// There are two controls - one for carrying out an inspection on the vehicle,
/// and one for carrying out a remediation.
class VehicleActionContainer extends StatelessWidget {
  // MEMBER VARIABLES //
  final String vehicleID; // ID of vehicle

  // THEME-ING //
  // sizing
  final double actionButtonHeight = 75; // height of the action button.

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  const VehicleActionContainer({
    super.key,
    required this.vehicleID,
  });

  // //////////// //
  // BUILD METHOD //
  // //////////// //

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // ///// //
        // TITLE //
        // ///// //

        const Text(
          "Action",
          style: MyTextStyles.headerText2,
        ),

        const SizedBox(height: MySizes.spacing),

        Row(
          children: [
            // /////// //
            // INSPECT //
            // /////// //

            Expanded(
              child: _buildActionButton(
                text: "Inspect",
                icon: FontAwesomeIcons.magnifyingGlass,
                onPressed: () {
                  // handling the action
                  _handleInspect(context);
                },
              ),
            ),

            // TODO removed for deliverable 1 (remediation)
            // const SizedBox(width: MySizes.spacing),

            // ///////// //
            // REMEDIATE //
            // ///////// //

            // Expanded(
            //   child: _buildActionButton(
            //     text: "Remediate",
            //     icon: FontAwesomeIcons.hammer,
            //     onPressed: () {
            //       // handling the action
            //       _handleRemediate(context);
            //     },
            //   ),
            // ),
          ],
        ),
      ],
    );
  }

  // ////////////////////// //
  // HELPER BUILDER METHODS //
  // ////////////////////// //

  /// Builds an instance of the 'action button' within the container.
  Widget _buildActionButton({
    required String text,
    required IconData icon,
    Function()? onPressed,
  }) {
    return OutlinedButton(
      style: OutlinedButton.styleFrom(
        foregroundColor: MyColors.textPrimary,
        backgroundColor: MyColors.backgroundSecondary,
        padding: MySizes.padding,
        side: const BorderSide(
          width: 0,
          color: MyColors.backgroundSecondary,
        ),
      ),
      onPressed: onPressed,
      child: SizedBox(
        height: actionButtonHeight,
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              icon,
              color: MyColors.textPrimary,
              size: MySizes.largeIconSize,
            ),
            const SizedBox(height: MySizes.spacing),
            Text(
              text,
              style: MyTextStyles.headerText1,
            ),
          ],
        ),
      ),
    );
  }

  // ////////////// //
  // HELPER METHODS //
  // ////////////// //

  /// Handles the user selecting the 'Inspect' action button.
  ///
  /// Displays a confirmation dialog to ensure the user would like to carry out
  /// an inspection, and if so, navigates the user to the inspect page.
  Future<void> _handleInspect(BuildContext context) async {
    // displaying confirmation dialog
    bool result = await showDialog(
      context: context,
      builder: (BuildContext context) {
        return const ConfirmationDialog(
          title: "Start Inspection",
          message:
              "Are you sure you want to start an inspection? This will overwrite the existing status of the vehicle.",
          falseText: "No",
          trueText: "Yes",
          trueBackgroundColor: MyColors.green,
          trueTextColor: MyColors.antiPrimary,
        );
      },
    );

    // acting based on result of dialog
    if (result) {
      // result true -> navigate to inspect

      // navigating to inspect page
      context.pushNamed(
        Routes.inspect,
        params: {"vehicleID": vehicleID},
      );
    } else {
      // result false -> do nothing

      // nothing
    }
  }

  /// Handles the user selecting the 'Remediate' action button.
  ///
  /// Displays a confirmation dialog to ensure the user would like to carry out
  /// a remediation, and if so, navigates the user to the remediate page.
  Future<void> _handleRemediate(BuildContext context) async {
    // displaying confirmation dialog
    bool result = await showDialog(
      context: context,
      builder: (BuildContext context) {
        return const ConfirmationDialog(
          title: "Start Remediation",
          message:
              "Are you sure you want to start a remediation? This will overwrite the existing status of the vehicle.",
          falseText: "No",
          trueText: "Yes",
          trueBackgroundColor: MyColors.green,
          trueTextColor: MyColors.antiPrimary,
        );
      },
    );

    // acting based on result of dialog
    if (result) {
      // result true -> navigate to inspect

      // navigating to remediate page
      context.pushNamed(
        Routes.remediate,
        params: {"vehicleID": vehicleID},
      );
    } else {
      // result false -> do nothing

      // nothing
    }
  }
}
