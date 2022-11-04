import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import 'package:percent_indicator/circular_percent_indicator.dart';
import 'package:train_vis_mobile/view/theme/data/my_colors.dart';
import 'package:train_vis_mobile/view/theme/data/my_sizes.dart';
import 'package:train_vis_mobile/view/theme/data/my_text_styles.dart';
import 'package:train_vis_mobile/view/theme/widgets/my_icon_button.dart';
import 'package:train_vis_mobile/view/widgets/colored_container.dart';
import 'package:train_vis_mobile/view/widgets/padded_custom_scroll_view.dart';

/// Page to display the profile of a train vehicle.
///
/// TODO
class ProfilePage extends StatelessWidget {
  // MEMBERS //
  final String vehicleID;

  // SIZING //
  final double _headerHeight = 180;

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  const ProfilePage({
    super.key,
    required this.vehicleID,
  });

  // //////////// //
  // BUILD METHOD //
  // //////////// //

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      // APP BAR //
      appBar: AppBar(
        leading: MyIconButton.back(
          onPressed: () {
            Navigator.of(context).pop();
          },
        ),
        title: Text(vehicleID, style: MyTextStyles.headerText1),
      ),
      body: PaddedCustomScrollView(
        slivers: [
          SliverToBoxAdapter(
            child: _buildHeader(),
          ),
        ],
      ),
    );
  }

  // ////////////////////// //
  // HELPER BUILDER METHODS //
  // ////////////////////// //

  /// Builds the widget used to display the
  Widget _buildHeader() {
    return Container(
      child: Stack(
        children: [
          // //////////// //
          // VEHICLE INFO //
          // //////////// //

          Column(
            children: [
              const SizedBox(height: 50),
              ColoredContainer(
                color: MyColors.backgroundSecondary,
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.end,
                  children: [
                    const SizedBox(height: 50),

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
              ),
            ],
          ),

          // ///////////// //
          // VEHICLE IMAGE //
          // ///////////// //

          Align(
            alignment: Alignment.topCenter,
            child: CircularPercentIndicator(
              radius: 50,
              lineWidth: 5.0,
              percent: 1.0,
              progressColor: MyColors.green,
              center: const CircleAvatar(
                radius: 40,
                backgroundImage: AssetImage("resources/images/707-012.png"),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
