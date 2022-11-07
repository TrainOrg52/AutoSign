import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
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
class StatusActionContainer extends StatelessWidget {
  // MEMBER VARIABLES //
  final Vehicle vehicle; // vehicle being displayed

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  const StatusActionContainer({
    super.key,
    required this.vehicle,
  });

  // //////////// //
  // BUILD METHOD //
  // //////////// //

  @override
  Widget build(BuildContext context) {
    return BorderedContainer(
      borderColor: MyColors.negative,
      backgroundColor: MyColors.negativeAccent,
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          // /////////////////// //
          // CONFORMANCE MESSAGE //
          // /////////////////// //
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: const [
              Icon(
                FontAwesomeIcons.circleExclamation,
                size: MySizes.mediumIconSize,
                color: MyColors.red,
              ),
              SizedBox(width: MySizes.spacing),
              Text(
                "Non-conformances present.",
                style: MyTextStyles.headerText3,
              ),
            ],
          ),

          const SizedBox(height: MySizes.spacing),

          // /////////////////////// //
          // CONFORMANCE DESCRIPTION //
          // /////////////////////// //

          const Text(
            "There are currently 5 non-conformances present on this vehicle.",
            style: MyTextStyles.bodyText1,
            textAlign: TextAlign.center,
          ),

          const SizedBox(height: MySizes.spacing),

          // //////////////////////// //
          // START REMEDIATION BUTTON //
          // //////////////////////// //

          MyTextButton.custom(
            backgroundColor: MyColors.negative,
            borderColor: MyColors.negative,
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
