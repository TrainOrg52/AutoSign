import 'package:auto_sign_mobile/controller/remediation_controller.dart';
import 'package:auto_sign_mobile/main.dart';
import 'package:auto_sign_mobile/view/pages/inspections/inspections.dart';
import 'package:auto_sign_mobile/view/routes/routes.dart';
import 'package:auto_sign_mobile/view/theme/data/my_colors.dart';
import 'package:auto_sign_mobile/view/theme/data/my_text_styles.dart';
import 'package:auto_sign_mobile/view/widgets/bordered_container.dart';
import 'package:auto_sign_mobile/view/widgets/custom_stream_builder.dart';
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
      ),
      body: CustomStreamBuilder(
        stream: RemediationController.instance
            .getVehicleRemediationsWhereVehicleIs(vehicleID),
        builder: (context, remediations) {
          return _buildRemediationList(context, remediations, vehicleID);
        },
      ),
    );
  }
}

ListView _buildRemediationList(BuildContext context,
    List<VehicleRemediation> remediations, String vehicleID) {
  return ListView.builder(
      padding: EdgeInsets.zero,
      itemCount: remediations.length * 2,
      itemBuilder: (_, index) {
        if (index.isEven) {
          return const Divider(
            height: 8,
          );
        }
        return remediationTile(remediations[index ~/ 2], vehicleID, context);
      });
}

Widget remediationTile(
    VehicleRemediation remediation, String vehicleID, BuildContext context) {
  return BorderedContainer(
      padding: const EdgeInsets.all(0),
      height: 70,
      borderRadius: 10,
      child: Center(
          child: ListTile(
              horizontalTitleGap: 0,
              title: Text(
                remediation.timestamp.toDateString().toString(),
                style: MyTextStyles.headerText1,
              ),
              subtitle: Row(
                children: [
                  locationWidget(remediation.location),
                  const SizedBox(
                    width: 16,
                  ),
                  CustomStreamBuilder(
                      stream: RemediationController.instance
                          .getSignRemediationsWhereVehicleRemediationIs(
                              remediation.id),
                      builder: (context, signremediations) {
                        return numIssuesWidget(signremediations.length);
                      })
                ],
              ),
              leading: const SizedBox(
                height: 30,
                width: 30,
                child: Center(
                    child: Icon(
                  FontAwesomeIcons.hammer,
                  size: 25,
                  color: Colors.black,
                )),
              ),
              trailing: IconButton(
                icon: const Icon(
                  Icons.navigate_next_sharp,
                  color: Colors.black,
                  size: 40,
                ),
                onPressed: () {
                  context.pushNamed(
                    Routes.vehicleRemediation,
                    params: {
                      "vehicleID": vehicleID,
                      "vehicleRemediationID": remediation.id,
                    },
                  );
                },
              ))));
}

Widget numIssuesWidget(int numIssues) {
  return Row(
    children: [
      const Icon(
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
