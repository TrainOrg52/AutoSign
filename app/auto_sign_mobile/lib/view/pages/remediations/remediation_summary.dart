import 'package:auto_sign_mobile/controller/remediation_controller.dart';
import 'package:auto_sign_mobile/view/pages/remediations/remediations.dart';
import 'package:auto_sign_mobile/view/routes/routes.dart';
import 'package:auto_sign_mobile/view/theme/data/my_colors.dart';
import 'package:auto_sign_mobile/view/theme/data/my_text_styles.dart';
import 'package:auto_sign_mobile/view/theme/widgets/my_text_button.dart';
import 'package:auto_sign_mobile/view/widgets/bordered_container.dart';
import 'package:auto_sign_mobile/view/widgets/custom_stream_builder.dart';
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
    Remediation exampleRemediation = Remediation("Reading", "22/06/22", 3, [
      "Entrance 1: Door",
      "Entrance 2: Beside Door",
      "Walkway: Seats"
    ], [], [
      "Emergency Exit not found",
      "Call for Aid not found",
      "Seat Number not found"
    ]);
    double spacingVal = 5;
    return Scaffold(
        appBar: AppBar(
          title: const Text(
            "Remediation",
            style: MyTextStyles.headerText1,
          ),
          backgroundColor: MyColors.antiPrimary,
          centerTitle: true,
        ),
        body: CustomStreamBuilder(
          stream: RemediationController.instance
              .getSignRemediationsWhereVehicleRemediationIs(
                  vehicleRemediationID),
          builder: (context, signremediations) {
            return _buildCheckpointList(context, signremediations, vehicleID);
          },
        ));
  }
}

ListView _buildCheckpointList(BuildContext context,
    List<SignRemediation> signremediations, String vehicleID) {
  return ListView.builder(
      itemCount: signremediations.length + 4,
      itemBuilder: (_, index) {
        if (index == 0) {
          //return remediationTile(remediation, context);
          return const Text("TODO");
        } else if (index == 1) {
          return const Text(
            "Inspection",
            style: MyTextStyles.headerText1,
          );
        } else if (index == 2) {
          //return reportTile(
          //  Report("22/06/22", "Reading", false, true, []), context);
          return (const Text("FILLER"));
        } else if (index == 3) {
          return const Text(
            "Report",
            style: MyTextStyles.headerText1,
          );
        }

        return remediationCheckpoint(
            signremediations[index - 4].checkpointTitle.toString(),
            signremediations[index - 4],
            signremediations[index - 4]
                .preRemediationConformanceStatus
                .toString(),
            vehicleID,
            context);
      });
}

Widget remediationCheckpoint(
    String sectionName,
    SignRemediation signRemediation,
    String issueDescription,
    String vehicleID,
    BuildContext context) {
  return CustomStreamBuilder(
      stream: RemediationController.instance.getSignRemediationDownloadURL(
          vehicleID, signRemediation.vehicleRemediationID, signRemediation.id),
      builder: (context, imageURL) {
        return BorderedContainer(
            padding: const EdgeInsets.all(0),
            height: 230,
            borderRadius: 10,
            borderColor: MyColors.backgroundPrimary,
            child:
                Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
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
                    }),
              ),
            ]));
      });
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
