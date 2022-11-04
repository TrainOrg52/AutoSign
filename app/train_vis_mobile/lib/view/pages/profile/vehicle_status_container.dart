import 'package:flutter/widgets.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import 'package:go_router/go_router.dart';
import 'package:train_vis_mobile/view/routes/routes.dart';
import 'package:train_vis_mobile/view/theme/data/my_colors.dart';
import 'package:train_vis_mobile/view/theme/data/my_sizes.dart';
import 'package:train_vis_mobile/view/theme/data/my_text_styles.dart';
import 'package:train_vis_mobile/view/theme/widgets/my_text_button.dart';
import 'package:train_vis_mobile/view/widgets/bordered_container.dart';

/// Widget that displays an overview of the status of a given train vehicle.
class VehicleConformanceStatusOverview extends StatelessWidget {
  // MEMBER VARIABLES //
  final String vehicleID; // ID of vehicle

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  const VehicleConformanceStatusOverview({
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
          "Status",
          style: MyTextStyles.headerText2,
        ),

        const SizedBox(height: MySizes.spacing),

        // /////////////// //
        // STATUS OVERVIEW //
        // /////////////// //

        BorderedContainer(
          borderColor: MyColors.green,
          backgroundColor: MyColors.greenAcent,
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
                    FontAwesomeIcons.solidCircleCheck,
                    size: MySizes.mediumIconSize,
                    color: MyColors.green,
                  ),
                  SizedBox(width: MySizes.spacing),
                  Text(
                    "No non-conformances present.",
                    style: MyTextStyles.headerText3,
                  ),
                ],
              ),

              const SizedBox(height: MySizes.spacing),

              // /////////// //
              // VIEW BUTTON //
              // /////////// //

              MyTextButton.custom(
                backgroundColor: MyColors.green,
                borderColor: MyColors.green,
                textColor: MyColors.antiPrimary,
                text: "View",
                onPressed: () {
                  // navigating to status
                  context.pushNamed(
                    Routes.status,
                    params: {"vehicleID": vehicleID},
                  );
                },
              ),
            ],
          ),
        ),
      ],
    );
  }
}
