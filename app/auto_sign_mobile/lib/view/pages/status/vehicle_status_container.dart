import 'package:auto_sign_mobile/model/enums/conformance_status.dart';
import 'package:auto_sign_mobile/model/vehicle/vehicle.dart';
import 'package:auto_sign_mobile/view/routes/routes.dart';
import 'package:auto_sign_mobile/view/theme/data/my_colors.dart';
import 'package:auto_sign_mobile/view/theme/data/my_sizes.dart';
import 'package:auto_sign_mobile/view/theme/data/my_text_styles.dart';
import 'package:auto_sign_mobile/view/theme/widgets/my_text_button.dart';
import 'package:auto_sign_mobile/view/widgets/bordered_container.dart';
import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

/// Widget that displays the status of the vehicle along with any action that
/// should be taken.
///
/// A custom [Container] that displays a message detailing the [ConformanceStatus]
/// of the [Vehicle], as well as a button to perform a remediation if needed.
class VehicleStatusContainer extends StatelessWidget {
  // MEMBER VARIABLES //
  final Vehicle vehicle; // vehicle being displayed

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  const VehicleStatusContainer({
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

          // //////////////////////// //
          // START REMEDIATION BUTTON //
          // //////////////////////// //

          if (vehicle.conformanceStatus == ConformanceStatus.nonConforming) ...[
            const SizedBox(height: MySizes.spacing),
            MyTextButton.custom(
              backgroundColor: vehicle.conformanceStatus.color,
              borderColor: vehicle.conformanceStatus.color,
              textColor: MyColors.antiPrimary,
              text: "Remediate",
              onPressed: () {
                // navigating to status
                context.pushNamed(
                  Routes.remediate,
                  params: {"vehicleID": vehicle.id},
                );
              },
            ),
          ],
        ],
      ),
    );
  }
}
