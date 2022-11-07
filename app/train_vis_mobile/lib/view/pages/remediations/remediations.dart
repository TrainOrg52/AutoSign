import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:train_vis_mobile/view/theme/data/my_colors.dart';
import "package:train_vis_mobile/view/theme/widgets/my_icon_button.dart";
import 'package:train_vis_mobile/view/theme/data/my_sizes.dart';
import 'package:train_vis_mobile/view/theme/data/my_text_styles.dart';
import 'package:train_vis_mobile/view/widgets/bordered_container.dart';
import 'package:train_vis_mobile/view/widgets/colored_container.dart';
import 'package:train_vis_mobile/view/pages/reports/reports.dart';

class RemediationsPage extends StatelessWidget {
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(
          "Remediations",
          style: MyTextStyles.headerText1,
        ),
        backgroundColor: MyColors.antiPrimary,
        centerTitle: true,
      ),
      body: _buildRemediationList(context),
    );
  }
}

ListView _buildRemediationList(BuildContext context) {
  List<Remediation> remediations = [
    Remediation("Reading", "22/06/22", 3, [], [], []),
    Remediation("Newport", "22/06/22", 1, [], [], []),
    Remediation("Leeds", "22/06/22", 2, [], [], [])
  ];

  return ListView.builder(
      padding: EdgeInsets.zero,
      itemCount: remediations.length * 2,
      itemBuilder: (_, index) {
        if (index.isEven) {
          return const Divider(
            height: 8,
          );
        }
        return remediationTile(remediations[index ~/ 2]);
      });
}

Widget remediationTile(Remediation remediation) {
  return BorderedContainer(
      padding: EdgeInsets.all(0),
      height: 70,
      borderRadius: 10,
      child: Center(
          child: ListTile(
              horizontalTitleGap: 0,
              title: Text(
                remediation.date,
                style: MyTextStyles.headerText1,
              ),
              subtitle: Row(
                children: [
                  locationWidget(remediation.location),
                  SizedBox(
                    width: 16,
                  ),
                  numIssuesWidget(remediation.numRemediations)
                ],
              ),
              leading: const Icon(
                Icons.build,
                size: 40,
                color: Colors.black,
              ),
              trailing: const Icon(
                Icons.navigate_next_sharp,
                color: Colors.black,
                size: 40,
              ))));
}

Widget numIssuesWidget(int numIssues) {
  return Row(
    children: [
      Icon(
        Icons.check_circle,
        color: MyColors.green,
      ),
      Text("$numIssues issues remediated")
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
