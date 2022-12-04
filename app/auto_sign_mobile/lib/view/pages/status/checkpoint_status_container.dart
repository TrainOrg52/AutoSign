import 'package:auto_sign_mobile/controller/vehicle_controller.dart';
import 'package:auto_sign_mobile/main.dart';
import 'package:auto_sign_mobile/model/vehicle/checkpoint.dart';
import 'package:auto_sign_mobile/view/routes/routes.dart';
import 'package:auto_sign_mobile/view/theme/data/my_colors.dart';
import 'package:auto_sign_mobile/view/theme/data/my_sizes.dart';
import 'package:auto_sign_mobile/view/theme/data/my_text_styles.dart';
import 'package:auto_sign_mobile/view/theme/widgets/my_icon_button.dart';
import 'package:auto_sign_mobile/view/widgets/bordered_container.dart';
import 'package:auto_sign_mobile/view/widgets/colored_container.dart';
import 'package:auto_sign_mobile/view/widgets/custom_stream_builder.dart';
import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import 'package:go_router/go_router.dart';

/// A custom [Container] that stores information on the status of a particular
/// [Checkpoint] within a [Vehicle].
///
/// The container displays an image of the checkpoint, the checkpoints title, and
/// a message that is either "conforming" or "non-conforming" depending on the
/// status of the checkpoint. The container also has a drop down button, which
/// reveals the result of the most recent inspection and remediation performed
/// on the vehicle.
///
/// The container has an [onExpanded] call-back, which is used to ensure that
/// only a single [CheckpointStatusContainer] can be expanded within a [CheckpointStatusList].
class CheckpointStatusContainer extends StatelessWidget {
  // MEMBER VARIABLES //
  final Checkpoint checkpoint; // checkpoint being displayed
  final bool isExpanded; // expansion state of the checkpoint.
  final Function() onExpanded; // call back for when container expanded

  // THEME-ING
  // sizes
  final double containerHeight = 100; // height of the un-expanded container.

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  const CheckpointStatusContainer({
    super.key,
    required this.checkpoint,
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

            if (isExpanded) _buildContainerDropDown(context)
          ],
        ),
      ),
    );
  }

  // ////////////////////// //
  // HELPER BUILDER METHODS //
  // ////////////////////// //

  /// Builds the widget that holds the main body of information within the
  /// container.
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
            child: CustomStreamBuilder(
              stream: VehicleController.instance.getCheckpointImageDownloadURL(
                checkpoint.vehicleID,
                checkpoint.id,
              ),
              builder: (context, downloadURL) {
                return Image.network(downloadURL);
              },
            ),
          ),

          const SizedBox(width: MySizes.spacing),

          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // //////////////// //
                // CHECKPOINT TITLE //
                // //////////////// //

                Text(
                  checkpoint.title,
                  style: MyTextStyles.headerText3,
                ),

                const Spacer(),

                // ///////////////// //
                // CHECKPOINT STATUS //
                // ///////////////// //

                BorderedContainer(
                  isDense: true,
                  borderColor: checkpoint.conformanceStatus.color,
                  backgroundColor: checkpoint.conformanceStatus.accentColor,
                  padding: const EdgeInsets.all(MySizes.paddingValue / 2),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(
                        checkpoint.conformanceStatus.iconData,
                        color: checkpoint.conformanceStatus.color,
                        size: MySizes.smallIconSize,
                      ),
                      const SizedBox(width: MySizes.spacing),
                      Text(
                        checkpoint.conformanceStatus.title.toTitleCase(),
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

  /// Builds the container that holds the information shown when the container
  /// is expanded.
  Widget _buildContainerDropDown(BuildContext context) {
    return Column(
      children: [
        const Divider(
          color: MyColors.lineColor,
          thickness: MySizes.lineWidth,
          height: (MySizes.spacing * 2) + 1,
        ),

        // ///////// //
        // DROP DOWN //
        // ///////// //

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
                    BorderedContainer(
                      isDense: true,
                      borderColor: checkpoint.lastVehicleInspectionResult.color,
                      backgroundColor:
                          checkpoint.lastVehicleInspectionResult.accentColor,
                      padding: const EdgeInsets.all(MySizes.paddingValue / 2),
                      child: Text(
                        checkpoint.lastVehicleInspectionResult.title
                            .toTitleCase(),
                        style: MyTextStyles.bodyText2,
                      ),
                    ),
                    const SizedBox(width: MySizes.spacing),
                    MyIconButton.secondary(
                      iconData: FontAwesomeIcons.circleChevronRight,
                      onPressed: () {
                        // navigating to inspection
                        context.pushNamed(
                          Routes.vehicleInspection,
                          params: {
                            "vehicleID": checkpoint.vehicleID,
                            "vehicleInspectionID":
                                checkpoint.lastVehicleInspectionID,
                          },
                        );
                      },
                    ),
                  ],
                ),
              ],
            ),

            // //////////// //
            // ACTION TAKEN //
            // //////////// //

            // TODO needs to be re-done for the proper remediate data model
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
                    if (checkpoint.lastVehicleRemediationID == "")
                      const BorderedContainer(
                        isDense: true,
                        borderColor: MyColors.lineColor,
                        backgroundColor: MyColors.greyAccent,
                        padding: EdgeInsets.all(MySizes.paddingValue / 2),
                        child: Text(
                          "None",
                          style: MyTextStyles.bodyText2,
                        ),
                      )
                    else ...[
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
                          context.pushNamed(
                            Routes.remediationWalkthrough,
                            params: {
                              "vehicleID": checkpoint.vehicleID,
                              "vehicleRemediationID":
                                  checkpoint.lastVehicleRemediationID!,
                            },
                          );
                        },
                      ),
                    ]
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
