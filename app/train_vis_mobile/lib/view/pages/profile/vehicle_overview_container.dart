import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import 'package:percent_indicator/percent_indicator.dart';
import 'package:train_vis_mobile/view/theme/data/my_colors.dart';
import 'package:train_vis_mobile/view/theme/data/my_sizes.dart';
import 'package:train_vis_mobile/view/theme/data/my_text_styles.dart';
import 'package:train_vis_mobile/view/widgets/colored_container.dart';

/// Widget that contains an overview of information about a given train vehicle.
class VehicleOverviewContainer extends StatelessWidget {
  // MEMBER VARIABLES //
  final String vehicleID;

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
    required this.vehicleID,
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
                    const Align(
                      alignment: Alignment.topRight,
                      child: Padding(
                        padding: MySizes.padding,
                        child: Icon(
                          FontAwesomeIcons.solidCircleCheck,
                          size: MySizes.mediumIconSize,
                          color: MyColors.green,
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
                          vehicleID,
                          style: MyTextStyles.headerText1,
                        ),

                        const SizedBox(height: MySizes.spacing),

                        // ///////////// //
                        // VEHICLE TITLE //
                        // ///////////// //

                        Text(
                          "Southeastern Type 707",
                          style: MyTextStyles.headerText2
                              .copyWith(fontWeight: FontWeight.w400),
                        ),

                        const SizedBox(height: MySizes.spacing),

                        // //////////////// //
                        // VEHICLE LOCATION //
                        // //////////////// //

                        Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: const [
                            Icon(
                              FontAwesomeIcons.locationDot,
                              size: MySizes.smallIconSize,
                            ),
                            SizedBox(width: MySizes.spacing),
                            Text(
                              "CAF, Newport",
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
            progressColor: MyColors.green,
            center: CircleAvatar(
              radius: avatarImageRadius,
              backgroundImage: const AssetImage("resources/images/707-012.png"),
            ),
          ),
        ),
      ],
    );
  }
}
