import 'package:auto_sign_mobile/controller/remediation_controller.dart';
import 'package:auto_sign_mobile/main.dart';
import 'package:auto_sign_mobile/view/pages/inspections/inspections.dart';
import 'package:auto_sign_mobile/view/routes/routes.dart';
import 'package:auto_sign_mobile/view/theme/data/my_colors.dart';
import 'package:auto_sign_mobile/view/theme/data/my_sizes.dart';
import 'package:auto_sign_mobile/view/theme/data/my_text_styles.dart';
import 'package:auto_sign_mobile/view/theme/widgets/my_icon_button.dart';
import 'package:auto_sign_mobile/view/widgets/custom_stream_builder.dart';
import 'package:auto_sign_mobile/view/widgets/padded_custom_scroll_view.dart';
import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import 'package:go_router/go_router.dart';

import '../../../model/remediation/vehicle_remediation.dart';

class RemediationsList extends StatelessWidget {
  String vehicleID;

  RemediationsList(this.vehicleID);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          "Remediations",
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
      body: CustomStreamBuilder(
        stream: RemediationController.instance
            .getVehicleRemediationsWhereVehicleIs(vehicleID),
        builder: (context, remediations) {
          return PaddedCustomScrollView(
            slivers: [
              SliverToBoxAdapter(
                child: _buildRemediationList(context, remediations, vehicleID),
              ),
            ],
          );
        },
      ),
    );
  }
}

ListView _buildRemediationList(BuildContext context,
    List<VehicleRemediation> remediations, String vehicleID) {
  return ListView.builder(
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      padding: EdgeInsets.zero,
      itemCount: remediations.length,
      itemBuilder: (context, index) {
        return Column(
          children: [
            remediationTile(remediations[index], vehicleID, context),
            if (index != remediations.length - 1)
              const SizedBox(height: MySizes.spacing),
          ],
        );
      });
}

Widget remediationTile(
    VehicleRemediation remediation, String vehicleID, BuildContext context) {
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
        Routes.vehicleRemediation,
        params: {
          "vehicleRemediationID": remediation.id,
          "vehicleID": vehicleID
        },
      );
    },
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
        const Icon(
          FontAwesomeIcons.circleChevronRight,
          size: MySizes.mediumIconSize,
          color: MyColors.textPrimary,
        ),
      ],
    ),
  );
}

Widget numIssuesWidget(int numIssues) {
  numIssues = 2;
  return Row(
    children: [
      const Icon(
        FontAwesomeIcons.solidCircleCheck,
        color: MyColors.green,
        size: MySizes.smallIconSize,
      ),
      const SizedBox(width: MySizes.spacing / 2),
      Text(
        "$numIssues remediation${numIssues > 1 ? 's' : ''}",
        style: MyTextStyles.bodyText1,
      ),
    ],
  );
}

class Remediation {
  String location;
  String date;
  int numRemediations;
  List<String> sectionNames;
  List<String> checkpointimages;
  List<String> descriptions;

  Remediation(this.location, this.date, this.numRemediations, this.sectionNames,
      this.checkpointimages, this.descriptions);
}
