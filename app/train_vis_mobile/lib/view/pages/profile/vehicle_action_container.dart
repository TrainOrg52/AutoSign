import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import 'package:go_router/go_router.dart';
import 'package:train_vis_mobile/view/routes/routes.dart';
import 'package:train_vis_mobile/view/theme/data/my_colors.dart';
import 'package:train_vis_mobile/view/theme/data/my_sizes.dart';
import 'package:train_vis_mobile/view/theme/data/my_text_styles.dart';
import 'package:train_vis_mobile/view/widgets/confirmation_dialog.dart';

/// Widget that displays actions that can be performed on a given vehicle.
class VehicleActionContainer extends StatelessWidget {
  // MEMBER VARIABLES //
  final String vehicleID; // ID of vehicle

  // THEME-ING //
  // sizing
  final double actionButtonHeight = 75;

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

            const SizedBox(width: MySizes.spacing),

            // ///////// //
            // REMEDIATE //
            // ///////// //

            Expanded(
              child: _buildActionButton(
                text: "Remediate",
                icon: FontAwesomeIcons.hammer,
                onPressed: () {
                  // navigating to remediate page
                  context.pushNamed(
                    Routes.remediate,
                    params: {"vehicleID": vehicleID},
                  );
                },
              ),
            ),
          ],
        ),
      ],
    );
  }

  // ////////////////////// //
  // HELPER BUILDER METHODS //
  // ////////////////////// //

  /// Button to perform action on the train vehicle.
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

  /// TODO
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
}
