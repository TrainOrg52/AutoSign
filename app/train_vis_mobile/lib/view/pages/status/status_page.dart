import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import 'package:go_router/go_router.dart';
import 'package:train_vis_mobile/view/routes/routes.dart';
import 'package:train_vis_mobile/view/theme/data/my_colors.dart';
import 'package:train_vis_mobile/view/theme/data/my_sizes.dart';
import 'package:train_vis_mobile/view/theme/data/my_text_styles.dart';
import 'package:train_vis_mobile/view/theme/widgets/my_icon_button.dart';
import 'package:train_vis_mobile/view/theme/widgets/my_text_button.dart';
import 'package:train_vis_mobile/view/widgets/bordered_container.dart';
import 'package:train_vis_mobile/view/widgets/padded_custom_scroll_view.dart';

/// Page to display the status of a train vehicle.
///
/// Provides a breakdown of the conformance status of each of the train vehicle's
/// checkpoints.
class StatusPage extends StatelessWidget {
  // MEMBERS //
  final String vehicleID; // ID of vehicle being displayed

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  const StatusPage({
    super.key,
    required this.vehicleID,
  });

  // //////////// //
  // BUILD METHOD //
  // //////////// //

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      // /////// //
      // APP BAR //
      // /////// //

      appBar: AppBar(
        leading: MyIconButton.back(
          onPressed: () {
            Navigator.of(context).pop();
          },
        ),
        title: const Text("Status", style: MyTextStyles.headerText1),
      ),

      // //// //
      // BODY //
      // //// //

      body: PaddedCustomScrollView(
        slivers: [
          // ///////////// //
          // STATUS ACTION //
          // ///////////// //

          SliverToBoxAdapter(
            child: StatusActionContainer(vehicleID: vehicleID),
          ),

          const SliverToBoxAdapter(child: SizedBox(height: MySizes.spacing)),

          // /////////// //
          // CHECKPOINTS //
          // /////////// //
        ],
      ),
    );
  }
}

/// Widget that displays the status of the vehicle along with any action that
/// should be taken.
class StatusActionContainer extends StatelessWidget {
  // MEMBER VARIABLES //
  final String vehicleID; // ID of vehicle being displayed

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  const StatusActionContainer({
    super.key,
    required this.vehicleID,
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
                "No non-conformances present.",
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
                params: {"vehicleID": vehicleID},
              );
            },
          ),
        ],
      ),
    );
  }
}
