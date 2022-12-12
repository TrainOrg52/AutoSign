import 'package:auto_sign_mobile/controller/inspection_controller.dart';
import 'package:auto_sign_mobile/controller/remediation_controller.dart';
import 'package:auto_sign_mobile/controller/vehicle_controller.dart';
import 'package:auto_sign_mobile/main.dart';
import 'package:auto_sign_mobile/view/theme/data/my_colors.dart';
import 'package:auto_sign_mobile/view/theme/data/my_text_styles.dart';
import 'package:auto_sign_mobile/view/theme/widgets/my_icon_button.dart';
import 'package:auto_sign_mobile/view/widgets/colored_container.dart';
import 'package:auto_sign_mobile/view/widgets/custom_stream_builder.dart';
import 'package:auto_sign_mobile/view/widgets/padded_custom_scroll_view.dart';
import 'package:flutter/material.dart';

import '../../../model/enums/capture_type.dart';
import '../../theme/data/my_sizes.dart';
import '../../widgets/bordered_container.dart';
import '../../widgets/capture_preview.dart';

///Class for showing an image within the app
class RemediationFix extends StatefulWidget {
  String vehicleID;
  String vehicleRemediationID;
  String signRemediationID;

  RemediationFix(
    this.vehicleID,
    this.vehicleRemediationID,
    this.signRemediationID,
  );

  @override
  RemediationFixState createState() => RemediationFixState(
        vehicleID,
        vehicleRemediationID,
        signRemediationID,
      );
}

///Stateful class showing the desired image.
class RemediationFixState extends State<RemediationFix> {
  String vehicleID;
  String vehicleRemediationID;
  String signRemediationID;

  RemediationFixState(
    this.vehicleID,
    this.vehicleRemediationID,
    this.signRemediationID,
  );
  final List<bool> toggleStates = <bool>[true, false, false];

  @override
  Widget build(BuildContext context) {
    return CustomStreamBuilder(
        stream: RemediationController.instance
            .getSignRemediation(signRemediationID),
        builder: (context, sign) {
          return Scaffold(
              appBar: AppBar(
                title: Text(
                  sign.checkpointTitle.toString(),
                  style: MyTextStyles.headerText1,
                ),
                backgroundColor: MyColors.antiPrimary,
                centerTitle: true,
                leading: MyIconButton.back(
                  onPressed: () {
                    Navigator.of(context).pop();
                  },
                ),
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
                                style: MyTextStyles.headerText2,
                              )
                            ]),
                        const SizedBox(
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
                                  sign.preRemediationConformanceStatus.iconData,
                                  size: MySizes.smallIconSize,
                                  color: sign
                                      .preRemediationConformanceStatus.color,
                                ),
                                const SizedBox(width: MySizes.spacing),
                                Text(
                                  sign.preRemediationConformanceStatus.title
                                      .toTitleCase(),
                                  style: MyTextStyles.bodyText1,
                                ),
                              ],
                            ),
                          )
                        ]),
                        const SizedBox(
                          height: MySizes.spacing,
                        ),
                        Row(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: const [
                              Text(
                                "Action",
                                style: MyTextStyles.headerText2,
                              )
                            ]),
                        const SizedBox(
                          height: MySizes.spacing,
                        ),
                        Row(children: [
                          BorderedContainer(
                            isDense: true,
                            borderColor: MyColors.green,
                            backgroundColor: MyColors.greenAccent,
                            padding:
                                const EdgeInsets.all(MySizes.paddingValue / 2),
                            child: Row(
                              mainAxisAlignment: MainAxisAlignment.end,
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                Icon(
                                  sign.remediationAction.iconData,
                                  size: MySizes.smallIconSize,
                                  color: sign.remediationAction.color,
                                ),
                                const SizedBox(width: MySizes.spacing),
                                Text(
                                  sign.remediationAction.title.toTitleCase(),
                                  style: MyTextStyles.bodyText1,
                                ),
                              ],
                            ),
                          )
                        ]),
                        const SizedBox(
                          height: MySizes.spacing,
                        ),
                        Row(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: const [
                              Text(
                                "Images",
                                style: MyTextStyles.headerText3,
                              )
                            ]),
                        const SizedBox(
                          height: MySizes.spacing,
                        ),
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
                                      signRemediationID,
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
          return CapturePreview(
            key: GlobalKey(),
            captureType: CaptureType.photo,
            path: imageURL,
            isNetworkURL: true,
          );
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
