import 'package:flutter/widgets.dart';
import 'package:go_router/go_router.dart';
import 'package:train_vis_mobile/model/status/conformance_status.dart';
import 'package:train_vis_mobile/model/vehicle/vehicle.dart';
import 'package:train_vis_mobile/view/routes/routes.dart';
import 'package:train_vis_mobile/view/theme/data/my_colors.dart';
import 'package:train_vis_mobile/view/theme/data/my_sizes.dart';
import 'package:train_vis_mobile/view/theme/data/my_text_styles.dart';
import 'package:train_vis_mobile/view/theme/widgets/my_text_button.dart';
import 'package:train_vis_mobile/view/widgets/bordered_container.dart';

/// Widget that displays an overview of the status of a given train vehicle.
///
/// A custom [Container] that displays an overview of the status of a [Vehicle],
/// and a button to navigate to the [StatusPage].s
class VehicleStatusOverviewContainer extends StatelessWidget {
  // MEMBER VARIABLES //
  final Vehicle vehicle; // vehicle being displayed

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  const VehicleStatusOverviewContainer({
    super.key,
    required this.vehicle,
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

              // /////////// //
              // VIEW BUTTON //
              // /////////// //

              MyTextButton.custom(
                backgroundColor: vehicle.conformanceStatus.color,
                borderColor: vehicle.conformanceStatus.accentColor,
                textColor: MyColors.antiPrimary,
                text: "View",
                onPressed: () {
                  // navigating based on conformance status
                  if (vehicle.conformanceStatus == ConformanceStatus.pending) {
                    // conformance status pending -> go to most recent inspection

                    // navigating to inspection page
                    context.pushNamed(
                      Routes.vehicleInspection,
                      params: {
                        "vehicleID": vehicle.id,
                        "vehicleInspectionID": vehicle.lastVehicleInspectionID,
                      },
                    );
                  } else {
                    // conformance status not pending -> go to status page

                    // navigating to status page
                    context.pushNamed(
                      Routes.status,
                      params: {"vehicleID": vehicle.id},
                    );
                  }
                },
              ),
            ],
          ),
        ),
      ],
    );
  }
}
