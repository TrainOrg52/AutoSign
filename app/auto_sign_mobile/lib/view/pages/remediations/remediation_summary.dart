import 'package:auto_sign_mobile/controller/remediation_controller.dart';
import 'package:auto_sign_mobile/view/pages/remediations/remediations.dart';
import 'package:auto_sign_mobile/view/routes/routes.dart';
import 'package:auto_sign_mobile/view/theme/data/my_colors.dart';
import 'package:auto_sign_mobile/view/theme/data/my_text_styles.dart';
import 'package:auto_sign_mobile/view/theme/widgets/my_text_button.dart';
import 'package:auto_sign_mobile/view/widgets/bordered_container.dart';
import 'package:auto_sign_mobile/view/widgets/custom_stream_builder.dart';
import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

import '../../../model/remediation/sign_remediation.dart';

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
            return _buildCheckpointList(context, signremediations);
          },
        ));
  }
}

ListView _buildCheckpointList(
    BuildContext context, List<SignRemediation> signremediations) {
  return ListView.builder(
      itemCount: 7,
      itemBuilder: (_, index) {
        if (index == 0) {
          //return remediationTile(remediation, context);
          return Text("TODO");
        } else if (index == 1) {
          return const Text(
            "Inspection",
            style: MyTextStyles.headerText1,
          );
        } else if (index == 2) {
          //return reportTile(
          //  Report("22/06/22", "Reading", false, true, []), context);
          return (Text("FILLER"));
        } else if (index == 3) {
          return const Text(
            "Report",
            style: MyTextStyles.headerText1,
          );
        }

        return remediationCheckpoint(
            signremediations[index - 4].checkpointTitle.toString(),
            "https://upload.wikimedia.org/wikipedia/commons/4/4a/100x100_logo.png",
            signremediations[index - 4].title.toString(),
            context);
      });
}

Widget remediationCheckpoint(String sectionName, String imageURL,
    String issueDescription, BuildContext context) {
  return BorderedContainer(
      padding: const EdgeInsets.all(0),
      height: 230,
      borderRadius: 10,
      borderColor: MyColors.backgroundPrimary,
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Text(
          sectionName,
          style: MyTextStyles.headerText1,
        ),
        Row(children: [
          SizedBox(
            width: 100,
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
                  Routes.remediationCheckpoint,
                  params: {
                    "remediationWalkthroughID": "2",
                    "vehicleID": "707-008",
                    "remediationCheckpointID": "2"
                  },
                );
              }),
        ),
      ]));
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
      width: 250,
      height: 45,
      backgroundColor: MyColors.negativeAccent,
      borderColor: MyColors.negative,
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          const Icon(
            Icons.warning,
            color: MyColors.negative,
          ),
          const SizedBox(
            width: 10,
          ),
          Text(
            issueDescription,
            style: MyTextStyles.buttonTextStyle,
          )
        ],
      ));
}

Widget remediatedWidget() {
  return BorderedContainer(
      width: 160,
      height: 45,
      backgroundColor: MyColors.greenAccent,
      borderColor: MyColors.green,
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: const [
          Icon(
            Icons.check_circle,
            color: MyColors.green,
          ),
          SizedBox(
            width: 10,
          ),
          Text(
            "Remediated",
            style: MyTextStyles.buttonTextStyle,
          )
        ],
      ));
}

Widget spacing(double size) {
  return SizedBox(
    height: size,
  );
}
