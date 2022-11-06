import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import 'package:train_vis_mobile/view/theme/data/my_colors.dart';
import 'package:train_vis_mobile/view/theme/data/my_sizes.dart';
import 'package:train_vis_mobile/view/theme/data/my_text_styles.dart';
import 'package:train_vis_mobile/view/theme/widgets/my_icon_button.dart';
import 'package:train_vis_mobile/view/widgets/bordered_container.dart';
import 'package:train_vis_mobile/view/widgets/colored_container.dart';

/// Container that stores information on the status of a particular checkpoint
/// within a vehicle.
class CheckpointStatusContainer extends StatelessWidget {
  // MEMBER VARIABLES //
  final String checkpointID; // ID of vehicle being displayed
  final bool isExpanded;
  final Function() onExpanded;

  // THEME-ING
  // sizes
  final double containerHeight = 100; // height of the un-expanded container.

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  const CheckpointStatusContainer({
    super.key,
    required this.checkpointID,
    required this.isExpanded,
    required this.onExpanded,
  });

  // //////////// //
  // BUILD METHOD //
  // //////////// //

  @override
  Widget build(BuildContext context) {
    return AnimatedContainer(
      duration: const Duration(milliseconds: 500),
      child: ColoredContainer(
        color: MyColors.backgroundSecondary,
        padding: MySizes.padding,
        child: Column(
          children: [
            // //// //
            // BODY //
            // //// //

            _buildContainerBody(),

            // //////// //
            // DROPDOWN //
            // //////// //

            if (isExpanded) _buildContainerDropDown()
          ],
        ),
      ),
    );
  }

  // ////////////////////// //
  // HELPER BUILDER METHODS //
  // ////////////////////// //

  /// Information held within the main body of the container.
  Widget _buildContainerBody() {
    return SizedBox(
      height: containerHeight,
      child: Row(
        mainAxisAlignment: MainAxisAlignment.start,
        children: [
          // //////////////// //
          // CHECKPOINT IMAGE //
          // //////////////// //

          BorderedContainer(
            isDense: true,
            backgroundColor: Colors.transparent,
            padding: const EdgeInsets.all(MySizes.paddingValue / 2),
            child: Image.asset("resources/images/checkpoint 1.png"),
          ),

          const SizedBox(width: MySizes.spacing),

          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // //////////////// //
                // CHECKPOINT TITLE //
                // //////////////// //

                const Text(
                  "Entrance 1: Door",
                  style: MyTextStyles.headerText3,
                ),

                const Spacer(),

                // ///////////////// //
                // CHECKPOINT STATUS //
                // ///////////////// //

                BorderedContainer(
                  isDense: true,
                  borderColor: MyColors.green,
                  backgroundColor: MyColors.greenAcent,
                  padding: const EdgeInsets.all(MySizes.paddingValue / 2),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: const [
                      Icon(
                        FontAwesomeIcons.solidCircleCheck,
                        color: MyColors.green,
                        size: MySizes.smallIconSize,
                      ),
                      SizedBox(width: MySizes.spacing),
                      Text(
                        "Conforming",
                        style: MyTextStyles.bodyText2,
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
          MyIconButton.secondary(
            iconData: isExpanded
                ? FontAwesomeIcons.circleChevronUp
                : FontAwesomeIcons.circleChevronDown,
            iconSize: MySizes.mediumIconSize,
            onPressed: () {
              // calling the call back
              onExpanded();
            },
          ),
        ],
      ),
    );
  }

  /// Information held within the dropdown for the container.
  Widget _buildContainerDropDown() {
    return Column(
      children: [
        const Divider(
          color: MyColors.lineColor,
          thickness: MySizes.lineWidth,
          height: (MySizes.spacing * 2) + 1,
        ),

        // //// //
        // BODY //
        // //// //

        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            // /////////////// //
            // LAST INSPECTION //
            // /////////////// //

            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  "Last Inspection",
                  style: MyTextStyles.bodyText2,
                ),
                const SizedBox(height: MySizes.spacing),
                Row(
                  children: [
                    const BorderedContainer(
                      isDense: true,
                      borderColor: MyColors.negative,
                      backgroundColor: MyColors.negativeAccent,
                      padding: EdgeInsets.all(MySizes.paddingValue / 2),
                      child: Text(
                        "Non-Conforming",
                        style: MyTextStyles.bodyText2,
                      ),
                    ),
                    const SizedBox(width: MySizes.spacing),
                    MyIconButton.secondary(
                      iconData: FontAwesomeIcons.circleChevronRight,
                      onPressed: () {
                        // navigating to inspection
                        // TODO
                      },
                    ),
                  ],
                ),
              ],
            ),

            // //////////// //
            // ACTION TAKEN //
            // //////////// //

            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  "Action Taken",
                  style: MyTextStyles.bodyText2,
                ),
                const SizedBox(height: MySizes.spacing),
                Row(
                  children: [
                    const BorderedContainer(
                      isDense: true,
                      borderColor: MyColors.green,
                      backgroundColor: MyColors.greenAcent,
                      padding: EdgeInsets.all(MySizes.paddingValue / 2),
                      child: Text(
                        "Remediated",
                        style: MyTextStyles.bodyText2,
                      ),
                    ),
                    const SizedBox(width: MySizes.spacing),
                    MyIconButton.secondary(
                      iconData: FontAwesomeIcons.circleChevronRight,
                      onPressed: () {
                        // navigating to remediation
                        // TODO
                      },
                    ),
                  ],
                ),
              ],
            ),
          ],
        ),
      ],
    );
  }
}
