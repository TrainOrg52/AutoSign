import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import 'package:percent_indicator/percent_indicator.dart';
import 'package:train_vis_mobile/controller/vehicle_controller.dart';
import 'package:train_vis_mobile/model/vehicle/vehicle.dart';
import 'package:train_vis_mobile/view/theme/data/my_colors.dart';
import 'package:train_vis_mobile/view/theme/data/my_sizes.dart';
import 'package:train_vis_mobile/view/theme/data/my_text_styles.dart';
import 'package:train_vis_mobile/view/widgets/colored_container.dart';
import 'package:train_vis_mobile/view/widgets/custom_future_builder.dart';

/// Widget that contains an overview of information about a given train vehicle.
class VehicleOverviewContainer extends StatelessWidget {
  // MEMBER VARIABLES //
  final Vehicle vehicle; // ID of vehicle

  // THEME-ING //
  // sizes
  final double avatarRadius = 50;
  final double avatarImageRadius = 40;
  final double avatarOutlineWidth = 5.0;

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  const VehicleOverviewContainer({
    super.key,
    required this.vehicle,
  });

  // //////////// //
  // BUILD METHOD //
  // //////////// //

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        // //////////// //
        // VEHICLE INFO //
        // //////////// //

        Align(
          child: Column(
            children: [
              // spacing for off-centered avatar
              SizedBox(height: avatarRadius),

              ColoredContainer(
                color: MyColors.backgroundSecondary,
                child: Stack(
                  children: [
                    // ////////////// //
                    // VEHICLE STATUS //
                    // ////////////// //
                    Align(
                      alignment: Alignment.topRight,
                      child: Padding(
                        padding: MySizes.padding,
                        child: Icon(
                          vehicle.conformanceStatus.iconData,
                          size: MySizes.mediumIconSize,
                          color: vehicle.conformanceStatus.color,
                        ),
                      ),
                    ),

                    Column(
                      mainAxisAlignment: MainAxisAlignment.end,
                      children: [
                        // spacing for off-centered avatar
                        SizedBox(height: avatarRadius),

                        // ////////// //
                        // VEHICLE ID //
                        // ////////// //

                        Text(
                          vehicle.id,
                          style: MyTextStyles.headerText1,
                        ),

                        const SizedBox(height: MySizes.spacing),

                        // ///////////// //
                        // VEHICLE TITLE //
                        // ///////////// //

                        Text(
                          vehicle.title,
                          style: MyTextStyles.headerText2
                              .copyWith(fontWeight: FontWeight.w400),
                        ),

                        const SizedBox(height: MySizes.spacing),

                        // //////////////// //
                        // VEHICLE LOCATION //
                        // //////////////// //

                        Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            const Icon(
                              FontAwesomeIcons.locationDot,
                              size: MySizes.smallIconSize,
                            ),
                            const SizedBox(width: MySizes.spacing),
                            Text(
                              vehicle.location,
                              style: MyTextStyles.bodyText1,
                            ),
                          ],
                        ),

                        const SizedBox(height: MySizes.spacing),
                      ],
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),

        // ///////////// //
        // VEHICLE IMAGE //
        // ///////////// //

        Align(
          alignment: Alignment.topCenter,
          child: CircularPercentIndicator(
            radius: avatarRadius,
            lineWidth: avatarOutlineWidth,
            percent: 1.0,
            progressColor: vehicle.conformanceStatus.color,
            center: CustomFutureBuilder(
              future: VehicleController.instance
                  .getVehicleAvatarDownloadURL(vehicle.id),
              builder: (context, downloadURL) {
                return CircleAvatar(
                  radius: avatarImageRadius,
                  backgroundImage: NetworkImage(downloadURL),
                );
              },
            ),
          ),
        ),
      ],
    );
  }
}
