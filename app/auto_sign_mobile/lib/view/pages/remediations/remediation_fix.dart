import 'package:auto_sign_mobile/controller/inspection_controller.dart';
import 'package:auto_sign_mobile/controller/remediation_controller.dart';
import 'package:auto_sign_mobile/controller/vehicle_controller.dart';
import 'package:auto_sign_mobile/model/remediation/vehicle_remediation.dart';
import 'package:auto_sign_mobile/view/pages/remediations/remediation_summary.dart';
import 'package:auto_sign_mobile/view/theme/data/my_colors.dart';
import 'package:auto_sign_mobile/view/theme/data/my_text_styles.dart';
import 'package:auto_sign_mobile/view/theme/widgets/my_text_button.dart';
import 'package:auto_sign_mobile/view/widgets/colored_container.dart';
import 'package:auto_sign_mobile/view/widgets/custom_stream_builder.dart';
import 'package:auto_sign_mobile/view/widgets/padded_custom_scroll_view.dart';
import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';

import '../../../model/enums/capture_type.dart';
import '../../theme/data/my_sizes.dart';
import '../../widgets/bordered_container.dart';
import '../../widgets/capture_preview.dart';

///Class for showing an image within the app
class RemediationFix extends StatefulWidget {
  String vehicleID;
  String vehicleRemediationID;
  String remediationCheckpointID;
  String signID;

  RemediationFix(this.vehicleID, this.vehicleRemediationID,
      this.remediationCheckpointID, this.signID);

  @override
  RemediationFixState createState() => RemediationFixState(
      vehicleID, vehicleRemediationID, remediationCheckpointID, signID);
}

///Stateful class showing the desired image.
class RemediationFixState extends State<RemediationFix> {
  String vehicleID;
  String vehicleRemediationID;
  String remediationCheckpointID;
  String signID;

  RemediationFixState(this.vehicleID, this.vehicleRemediationID,
      this.remediationCheckpointID, this.signID);
  final List<bool> toggleStates = <bool>[true, false, false];

  @override
  Widget build(BuildContext context) {
    return CustomStreamBuilder(
        stream: RemediationController.instance.getSignRemediation(signID),
        builder: (context, sign) {
          return Scaffold(
              appBar: AppBar(
                title: Text(
                  sign.checkpointTitle.toString(),
                  style: MyTextStyles.headerText1,
                ),
                backgroundColor: MyColors.antiPrimary,
                centerTitle: true,
              ),
              body: PaddedCustomScrollView(
                slivers: [
                  SliverToBoxAdapter(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.center,
                      children: [
                        Row(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: const [
                              Text(
                                "Issue",
                                style: MyTextStyles.headerText1,
                              )
                            ]),
                        SizedBox(
                          height: MySizes.spacing,
                        ),
                        Row(children: [
                          BorderedContainer(
                            isDense: true,
                            borderColor: MyColors.negative,
                            backgroundColor: MyColors.negativeAccent,
                            padding:
                                const EdgeInsets.all(MySizes.paddingValue / 2),
                            child: Row(
                              mainAxisAlignment: MainAxisAlignment.end,
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                Icon(
                                  FontAwesomeIcons.exclamation,
                                  size: MySizes.smallIconSize,
                                  color: MyColors.negative,
                                ),
                                SizedBox(width: MySizes.spacing),
                                Text(
                                  sign.preRemediationConformanceStatus
                                      .toString(),
                                  style: MyTextStyles.bodyText1,
                                ),
                              ],
                            ),
                          )
                        ]),
                        SizedBox(
                          height: MySizes.spacing,
                        ),
                        Row(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: const [
                              Text(
                                "Action",
                                style: MyTextStyles.headerText1,
                              )
                            ]),
                        SizedBox(
                          height: MySizes.spacing,
                        ),
                        Row(children: [
                          BorderedContainer(
                            isDense: true,
                            borderColor: MyColors.green,
                            backgroundColor: MyColors.greenAccent,
                            padding: EdgeInsets.all(MySizes.paddingValue / 2),
                            child: Row(
                              mainAxisAlignment: MainAxisAlignment.end,
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                Icon(
                                  FontAwesomeIcons.recycle,
                                  size: MySizes.smallIconSize,
                                  color: MyColors.green,
                                ),
                                SizedBox(width: MySizes.spacing),
                                Text(
                                  sign.remediationAction.toString(),
                                  style: MyTextStyles.bodyText1,
                                ),
                              ],
                            ),
                          )
                        ]),
                        SizedBox(
                          height: MySizes.spacing,
                        ),
                        Row(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: const [
                              Text(
                                "Photos",
                                style: MyTextStyles.headerText1,
                              )
                            ]),
                        ToggleButtons(
                          onPressed: (int index) {
                            setState(() {
                              for (int i = 0; i < toggleStates.length; i++) {
                                toggleStates[i] = i == index;
                              }
                            });
                          },
                          isSelected: toggleStates,
                          borderRadius:
                              const BorderRadius.all(Radius.circular(8)),
                          selectedBorderColor: MyColors.borderColor,
                          selectedColor: Colors.white,
                          fillColor: MyColors.primaryAccent,
                          constraints: const BoxConstraints(
                            minHeight: 40.0,
                            minWidth: 110,
                          ),
                          children: const [
                            Text(
                              "Inspection",
                              style: MyTextStyles.buttonTextStyle,
                            ),
                            Text(
                              "Remediation",
                              style: MyTextStyles.buttonTextStyle,
                            ),
                            Text(
                              "Expected",
                              style: MyTextStyles.buttonTextStyle,
                            )
                          ],
                        ),
                        const SizedBox(
                          height: MySizes.spacing,
                        ),
                        ColoredContainer(
                            color: MyColors.backgroundSecondary,
                            width: 300,
                            padding: MySizes.padding,
                            child: CustomStreamBuilder(
                                stream: RemediationController.instance
                                    .getVehicleRemediation(
                                        vehicleRemediationID),
                                builder: (context, vehicleRemediation) {
                                  return showImage(
                                      toggleStates,
                                      vehicleID,
                                      vehicleRemediationID,
                                      signID,
                                      vehicleRemediation.vehicleInspectionID,
                                      sign.checkpointInspectionID,
                                      sign.checkpointCaptureType,
                                      sign.checkpointID);
                                })),
                      ],
                    ),
                  )
                ],
              ));
        });
  }
}

Widget showImage(
  List<bool> toggleStates,
  String vehicleID,
  String vehicleRemediationID,
  String signRemediationID,
  String vehicleInspectionID,
  String checkpointInspectionID,
  CaptureType captureType,
  String checkpointID,
) {
  if (toggleStates[0]) {
    return CustomStreamBuilder<String>(
      stream: InspectionController.instance
          .getCheckpointInspectionCaptureDownloadURL(vehicleID,
              vehicleInspectionID, checkpointInspectionID, captureType),
      builder: (context, downloadURL) {
        return CapturePreview(
          key: GlobalKey(),
          captureType: captureType,
          path: downloadURL,
          isNetworkURL: true,
        );
      },
    );
  } else if (toggleStates[1]) {
    return CustomStreamBuilder(
        stream: RemediationController.instance.getSignRemediationDownloadURL(
            vehicleID, vehicleRemediationID, signRemediationID),
        builder: (context, imageURL) {
          return Image(image: NetworkImage(imageURL));
        });
  } else {
    return CustomStreamBuilder<String>(
      stream: VehicleController.instance.getCheckpointDemoDownloadURL(
        vehicleID,
        checkpointID,
        captureType,
      ),
      builder: (context, downloadURL) {
        return CapturePreview(
          key: GlobalKey(),
          captureType: captureType,
          path: downloadURL,
          isNetworkURL: true,
        );
      },
    );
  }
}
