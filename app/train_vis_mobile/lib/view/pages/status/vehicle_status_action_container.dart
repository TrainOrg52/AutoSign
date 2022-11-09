import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'package:train_vis_mobile/model/vehicle/vehicle.dart';
import 'package:train_vis_mobile/view/routes/routes.dart';
import 'package:train_vis_mobile/view/theme/data/my_colors.dart';
import 'package:train_vis_mobile/view/theme/data/my_sizes.dart';
import 'package:train_vis_mobile/view/theme/data/my_text_styles.dart';
import 'package:train_vis_mobile/view/theme/widgets/my_text_button.dart';
import 'package:train_vis_mobile/view/widgets/bordered_container.dart';

/// Widget that displays the status of the vehicle along with any action that
/// should be taken.
class VehicleStatusActionContainer extends StatelessWidget {
  // MEMBER VARIABLES //
  final Vehicle vehicle; // vehicle being displayed

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  const VehicleStatusActionContainer({
    super.key,
    required this.vehicle,
  });

  // //////////// //
  // BUILD METHOD //
  // //////////// //

  @override
  Widget build(BuildContext context) {
    return BorderedContainer(
      borderColor: vehicle.conformanceStatus.color,
      backgroundColor: vehicle.conformanceStatus.accentColor,
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          // /////////////////// //
          // CONFORMANCE MESSAGE //
          // /////////////////// //
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(
                vehicle.conformanceStatus.iconData,
                size: MySizes.mediumIconSize,
                color: vehicle.conformanceStatus.color,
              ),
              const SizedBox(width: MySizes.spacing),
              Text(
                vehicle.conformanceStatus.description,
                style: MyTextStyles.headerText3,
              ),
            ],
          ),

          const SizedBox(height: MySizes.spacing),

          // // /////////////////////// //
          // // CONFORMANCE DESCRIPTION //
          // // /////////////////////// //

          // const Text(
          //   "There are currently 5 non-conformances present on this vehicle.",
          //   style: MyTextStyles.bodyText1,
          //   textAlign: TextAlign.center,
          // ),

          const SizedBox(height: MySizes.spacing),

          // //////////////////////// //
          // START REMEDIATION BUTTON //
          // //////////////////////// //

          MyTextButton.custom(
            backgroundColor: vehicle.conformanceStatus.color,
            borderColor: vehicle.conformanceStatus.color,
            textColor: MyColors.antiPrimary,
            text: "Start Remediation",
            onPressed: () {
              // navigating to status
              context.pushNamed(
                Routes.remediate,
                params: {"vehicleID": vehicle.id},
              );
            },
          ),
        ],
      ),
    );
  }
}
