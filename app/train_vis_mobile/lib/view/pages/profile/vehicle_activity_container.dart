import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import 'package:go_router/go_router.dart';
import 'package:train_vis_mobile/view/routes/routes.dart';
import 'package:train_vis_mobile/view/theme/data/my_colors.dart';
import 'package:train_vis_mobile/view/theme/data/my_sizes.dart';
import 'package:train_vis_mobile/view/theme/data/my_text_styles.dart';

/// Displays buttons to access activity log of the given vehicle in terms of
/// inspections and remediations.
class VehicleActivityContainer extends StatelessWidget {
  // MEMBER VARIABLES //
  final String vehicleID; // ID of vehicle

  // THEME-ING //
  // sizing
  final double buttonHeight = 40;

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  const VehicleActivityContainer({super.key, required this.vehicleID});

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
          "Activity",
          style: MyTextStyles.headerText2,
        ),

        const SizedBox(height: MySizes.spacing),

        // /////// //
        // REPORTS //
        // /////// //

        _buildButton(
          text: "Reports",
          icon: FontAwesomeIcons.clipboardList,
          onPressed: () {
            // navigating to reports page
            context.pushNamed(
              Routes.reports,
              params: {"vehicleID": vehicleID},
            );
          },
        ),

        const SizedBox(height: MySizes.spacing),

        // //////////// //
        // REMEDIATIONS //
        // //////////// //

        _buildButton(
          text: "Remediations",
          icon: FontAwesomeIcons.hammer,
          onPressed: () {
            // navigating to remediationss page
            context.pushNamed(
              Routes.remediations,
              params: {"vehicleID": vehicleID},
            );
          },
        ),
      ],
    );
  }

  // ////////////////////// //
  // HELPER BUILDER METHODS //
  // ////////////////////// //

  /// Button to view a specific activity log
  Widget _buildButton({
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
        height: buttonHeight,
        child: Row(
          mainAxisAlignment: MainAxisAlignment.start,
          children: [
            Icon(
              icon,
              color: MyColors.textPrimary,
              size: MySizes.largeIconSize,
            ),
            const SizedBox(width: MySizes.spacing),
            Expanded(
              child: Text(
                text,
                style: MyTextStyles.headerText1,
              ),
            ),
            const Icon(
              FontAwesomeIcons.circleChevronRight,
              size: MySizes.mediumIconSize,
            ),
          ],
        ),
      ),
    );
  }
}
