import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import 'package:go_router/go_router.dart';
import 'package:train_vis_mobile/view/routes/routes.dart';
import 'package:train_vis_mobile/view/theme/data/my_colors.dart';
import 'package:train_vis_mobile/view/theme/data/my_sizes.dart';
import 'package:train_vis_mobile/view/theme/data/my_text_styles.dart';

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
                  // navigating to inspect page
                  context.goNamed(
                    Routes.inspect,
                    params: {"vehicleID": vehicleID},
                  );
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
                  context.goNamed(
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
            Text(
              text,
              style: MyTextStyles.headerText1,
            ),
          ],
        ),
      ),
    );
  }
}
