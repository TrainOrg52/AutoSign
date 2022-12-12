import 'package:auto_sign_mobile/controller/inspection_controller.dart';
import 'package:auto_sign_mobile/controller/remediation_controller.dart';
import 'package:auto_sign_mobile/controller/vehicle_controller.dart';
import 'package:auto_sign_mobile/main.dart';
import 'package:auto_sign_mobile/model/enums/capture_type.dart';
import 'package:auto_sign_mobile/model/inspection/vehicle_inspection.dart';
import 'package:auto_sign_mobile/model/remediation/vehicle_remediation.dart';
import 'package:auto_sign_mobile/view/pages/inspections/inspections.dart';
import 'package:auto_sign_mobile/view/pages/remediations/remediations.dart';
import 'package:auto_sign_mobile/view/routes/routes.dart';
import 'package:auto_sign_mobile/view/theme/data/my_colors.dart';
import 'package:auto_sign_mobile/view/theme/data/my_text_styles.dart';
import 'package:auto_sign_mobile/view/theme/widgets/my_icon_button.dart';
import 'package:auto_sign_mobile/view/theme/widgets/my_text_button.dart';
import 'package:auto_sign_mobile/view/widgets/bordered_container.dart';
import 'package:auto_sign_mobile/view/widgets/capture_preview.dart';
import 'package:auto_sign_mobile/view/widgets/colored_container.dart';
import 'package:auto_sign_mobile/view/widgets/custom_stream_builder.dart';
import 'package:auto_sign_mobile/view/widgets/padded_custom_scroll_view.dart';
import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import 'package:go_router/go_router.dart';

import '../../../model/remediation/sign_remediation.dart';
import '../../theme/data/my_sizes.dart';

class RemediationSummary extends StatelessWidget {
  String vehicleID;
  String vehicleRemediationID;

  RemediationSummary(this.vehicleID, this.vehicleRemediationID);

  @override
  Widget build(BuildContext context) {
    double spacingVal = 5;
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          "Remediation",
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
            child: CustomStreamBuilder(
              stream: RemediationController.instance
                  .getVehicleRemediation(vehicleRemediationID),
              builder: (context, remediation) {
                return Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    _buildRemediationSummary(remediation),
                    const SizedBox(height: MySizes.spacing),
                    const Text(
                      "Inspection",
                      style: MyTextStyles.headerText2,
                    ),
                    const SizedBox(height: MySizes.spacing),
                    CustomStreamBuilder(
                      stream: InspectionController.instance
                          .getVehicleInspection(
                              remediation.vehicleInspectionID),
                      builder: (context, vehicleInspection) {
                        return _buildInspectionSummary(
                            context, vehicleInspection);
                      },
                    ),
                  ],
                );
              },
            ),
          ),
          const SliverToBoxAdapter(
            child: SizedBox(height: MySizes.spacing),
          ),
          SliverToBoxAdapter(
            child: CustomStreamBuilder(
                stream: RemediationController.instance
                    .getSignRemediationsWhereVehicleRemediationIs(
                        vehicleRemediationID),
                builder: (context, signRemediations) {
                  return Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        "Report",
                        style: MyTextStyles.headerText2,
                      ),
                      const SizedBox(height: MySizes.spacing),
                      _buildCheckpointList(
                          context, signRemediations, vehicleID),
                    ],
                  );
                }),
          ),
        ],
      ),
    );
  }
}

Widget _buildRemediationSummary(VehicleRemediation remediation) {
  return ColoredContainer(
    color: MyColors.backgroundSecondary,
    padding: MySizes.padding,
    child: Row(
      children: [
        const Icon(
          FontAwesomeIcons.hammer,
          color: MyColors.textPrimary,
          size: MySizes.largeIconSize,
        ),
        const SizedBox(width: MySizes.spacing),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                remediation.timestamp.toDateString().toString(),
                style: MyTextStyles.headerText1,
              ),
              const SizedBox(height: MySizes.spacing / 2),
              Row(
                children: [
                  locationWidget(remediation.location),
                  const SizedBox(width: MySizes.spacing * 2),
                  CustomStreamBuilder(
                    stream: RemediationController.instance
                        .getSignRemediationsWhereVehicleRemediationIs(
                            remediation.id),
                    builder: (context, signremediations) {
                      return numIssuesWidget(signremediations.length);
                    },
                  ),
                ],
              ),
            ],
          ),
        ),
      ],
    ),
  );
}

Widget _buildInspectionSummary(
    BuildContext context, VehicleInspection inspection) {
  return OutlinedButton(
    style: OutlinedButton.styleFrom(
      foregroundColor: MyColors.textPrimary,
      backgroundColor: MyColors.backgroundSecondary,
      padding: MySizes.padding,
      side: const BorderSide(
        width: 0,
        color: MyColors.backgroundSecondary,
      ),
    ),
    onPressed: () {
      context.pushNamed(
        Routes.vehicleInspection,
        params: {
          "vehicleInspectionID": inspection.id,
          "vehicleID": inspection.vehicleID
        },
      );
    },
    child: Row(
      children: [
        const Icon(
          FontAwesomeIcons.magnifyingGlass,
          color: MyColors.textPrimary,
          size: MySizes.largeIconSize,
        ),
        const SizedBox(width: MySizes.spacing),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                inspection.timestamp.toDateString().toString(),
                style: MyTextStyles.headerText1,
              ),
              const SizedBox(height: MySizes.spacing / 2),
              Row(
                children: [
                  locationWidget(inspection.location),
                  const SizedBox(width: MySizes.spacing * 2),
                  processingStatusWidget(inspection.processingStatus)
                ],
              ),
            ],
          ),
        ),
        const Icon(
          FontAwesomeIcons.circleChevronRight,
          size: MySizes.mediumIconSize,
          color: MyColors.textPrimary,
        ),
      ],
    ),
  );
}

// Widget _buildReport() {}

ListView _buildCheckpointList(BuildContext context,
    List<SignRemediation> signremediations, String vehicleID) {
  return ListView.builder(
    shrinkWrap: true,
    physics: const NeverScrollableScrollPhysics(),
    itemCount: signremediations.length,
    itemBuilder: (_, index) {
      return Column(
        children: [
          remediationCheckpoint(
            signremediations[index].checkpointTitle.toString(),
            signremediations[index],
            signremediations[index].preRemediationConformanceStatus.toString(),
            vehicleID,
            context,
          ),
          if (index != signremediations.length - 1)
            const SizedBox(height: MySizes.spacing),
        ],
      );
    },
  );
}

Widget remediationCheckpoint(
  String sectionName,
  SignRemediation signRemediation,
  String issueDescription,
  String vehicleID,
  BuildContext context,
) {
  return OutlinedButton(
    style: OutlinedButton.styleFrom(
      foregroundColor: MyColors.textPrimary,
      backgroundColor: MyColors.backgroundSecondary,
      padding: MySizes.padding,
      side: const BorderSide(
        width: 0,
        color: MyColors.backgroundSecondary,
      ),
    ),
    onPressed: () {
      // navigating to image view
      context.pushNamed(
        Routes.signRemdiation,
        params: {
          "vehicleID": vehicleID,
          "vehicleRemediationID": signRemediation.vehicleRemediationID,
          "signRemediationID": signRemediation.id,
        },
      );
    },
    child: SizedBox(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            signRemediation.checkpointTitle,
            style: MyTextStyles.headerText3,
          ),
          const SizedBox(height: MySizes.spacing),
          Row(
            mainAxisAlignment: MainAxisAlignment.start,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              CustomStreamBuilder<String>(
                stream:
                    VehicleController.instance.getCheckpointShowcaseDownloadURL(
                  vehicleID,
                  signRemediation.checkpointID,
                ),
                builder: (context, downloadURL) {
                  return SizedBox(
                    height: 100,
                    child: CapturePreview(
                      captureType: CaptureType.photo,
                      path: downloadURL,
                      isNetworkURL: true,
                    ),
                  );
                },
              ),
              const SizedBox(width: MySizes.spacing),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      "Issue",
                      style: MyTextStyles.bodyText2,
                    ),
                    BorderedContainer(
                      isDense: true,
                      borderColor:
                          signRemediation.preRemediationConformanceStatus.color,
                      backgroundColor: signRemediation
                          .preRemediationConformanceStatus.accentColor,
                      padding: const EdgeInsets.all(MySizes.paddingValue / 2),
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Icon(
                            signRemediation
                                .preRemediationConformanceStatus.iconData,
                            color: signRemediation
                                .preRemediationConformanceStatus.color,
                            size: MySizes.smallIconSize,
                          ),
                          const SizedBox(width: MySizes.spacing),
                          Text(
                            "${signRemediation.title} : ${signRemediation.preRemediationConformanceStatus.title}"
                                .toTitleCase(),
                            style: MyTextStyles.bodyText2,
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(height: MySizes.spacing - 5),
                    const Text(
                      "Action",
                      style: MyTextStyles.bodyText2,
                    ),
                    BorderedContainer(
                      isDense: true,
                      borderColor: signRemediation.remediationAction.color,
                      backgroundColor:
                          signRemediation.remediationAction.accentColor,
                      padding: const EdgeInsets.all(MySizes.paddingValue / 2),
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Icon(
                            signRemediation.remediationAction.iconData,
                            color: signRemediation.remediationAction.color,
                            size: MySizes.smallIconSize,
                          ),
                          const SizedBox(width: MySizes.spacing),
                          Text(
                            signRemediation.remediationAction.title
                                .toTitleCase(),
                            style: MyTextStyles.bodyText2,
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
              Column(
                mainAxisAlignment: MainAxisAlignment.center,
                mainAxisSize: MainAxisSize.max,
                children: const [
                  Center(
                    child: Icon(
                      FontAwesomeIcons.circleChevronRight,
                      size: MySizes.mediumIconSize,
                    ),
                  ),
                ],
              ),
            ],
          ),
        ],
      ),
    ),
  );

  return CustomStreamBuilder(
    stream: RemediationController.instance.getSignRemediationDownloadURL(
        vehicleID, signRemediation.vehicleRemediationID, signRemediation.id),
    builder: (context, imageURL) {
      return BorderedContainer(
        padding: const EdgeInsets.all(0),
        height: 230,
        borderRadius: 10,
        borderColor: MyColors.backgroundPrimary,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              sectionName,
              style: MyTextStyles.headerText1,
            ),
            Row(children: [
              SizedBox(
                height: 150,
                child: Image(image: NetworkImage(imageURL)),
              ),
              const SizedBox(
                width: 30,
              ),
              issueAction(issueDescription)
            ]),
            spacing(10),
            Center(
              child: MyTextButton.secondary(
                text: "View",
                onPressed: () {
                  context.pushNamed(
                    Routes.signRemdiation,
                    params: {
                      "vehicleID": vehicleID,
                      "vehicleRemediationID":
                          signRemediation.vehicleRemediationID,
                      "signRemediationID": signRemediation.id,
                    },
                  );
                },
              ),
            ),
          ],
        ),
      );
    },
  );
}

Widget issueAction(String issueDescription) {
  return Column(
    crossAxisAlignment: CrossAxisAlignment.start,
    children: [
      const Text(
        "Issue",
        style: MyTextStyles.headerText2,
      ),
      issue(issueDescription),
      const Text("Action", style: MyTextStyles.headerText2),
      remediatedWidget(),
    ],
  );
}

Widget issue(String issueDescription) {
  return BorderedContainer(
    isDense: true,
    borderColor: MyColors.negative,
    backgroundColor: MyColors.negativeAccent,
    padding: const EdgeInsets.all(MySizes.paddingValue / 2),
    child: Row(
      mainAxisAlignment: MainAxisAlignment.end,
      mainAxisSize: MainAxisSize.min,
      children: [
        const Icon(
          FontAwesomeIcons.exclamation,
          size: MySizes.smallIconSize,
          color: MyColors.negative,
        ),
        const SizedBox(width: MySizes.spacing),
        Text(
          issueDescription,
          style: MyTextStyles.bodyText1,
        ),
      ],
    ),
  );
}

Widget remediatedWidget() {
  return BorderedContainer(
    isDense: true,
    borderColor: MyColors.green,
    backgroundColor: MyColors.greenAccent,
    padding: const EdgeInsets.all(MySizes.paddingValue / 2),
    child: Row(
      mainAxisAlignment: MainAxisAlignment.end,
      mainAxisSize: MainAxisSize.min,
      children: const [
        Icon(
          FontAwesomeIcons.recycle,
          size: MySizes.smallIconSize,
          color: MyColors.green,
        ),
        SizedBox(width: MySizes.spacing),
        Text(
          "Replaced",
          style: MyTextStyles.bodyText1,
        ),
      ],
    ),
  );
}

Widget spacing(double size) {
  return SizedBox(
    height: size,
  );
}
