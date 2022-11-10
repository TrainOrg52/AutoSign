import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'package:train_vis_mobile/view/pages/reports/reports.dart';
import 'package:train_vis_mobile/view/routes/routes.dart';
import 'package:train_vis_mobile/view/theme/data/my_colors.dart';
import 'package:train_vis_mobile/view/theme/data/my_text_styles.dart';
import 'package:train_vis_mobile/view/theme/widgets/my_text_button.dart';
import 'package:train_vis_mobile/view/widgets/bordered_container.dart';
import 'package:train_vis_mobile/view/widgets/colored_container.dart';

///Page showing the summary of the checkpoints for a given report
///Currently contains dummy data just to demonstrate the UI
class ReportSummary extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          "Reports",
          style: MyTextStyles.headerText1,
        ),
        backgroundColor: MyColors.antiPrimary,
        centerTitle: true,
      ),
      body: _buildReportSummary(context),
    );
  }
}

///Constructs a series of tiles for each of the checkpoints in the report
ListView _buildReportSummary(BuildContext context) {
  //Dummy checkpoints to populate the UI
  List<CheckPoint> checkpoints = [
    CheckPoint("Section 1", false),
    CheckPoint("Section 2", true),
    CheckPoint("Section 3", true),
    CheckPoint("Section 4", false)
  ];
  bool pending = false;
  bool current = false;

  return ListView.builder(
      itemCount: checkpoints.length + 1,
      itemBuilder: (_, index) {
        if (index == 0) {
          return reportTitleTile(pending, current);
        }

        CheckPoint currentPoint = checkpoints[index - 1];

        return checkpointViewWidget(currentPoint, pending, context);
      });
}

///Widget for building a list item for each checkpoint
Widget checkpointViewWidget(
    CheckPoint currentPoint, bool pending, BuildContext context) {
  return ColoredContainer(
      color: MyColors.backgroundPrimary,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            currentPoint.name,
            style: MyTextStyles.headerText2,
          ),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text("Image"),
              !pending
                  ? currentPoint.conforming
                      ? conforming()
                      : nonconforming()
                  : notProcessedWidget(),
              MyTextButton.secondary(
                  text: "View",
                  onPressed: () {
                    context.pushNamed(
                      Routes.checkpointInspection,
                      params: {
                        "vehicleInspectionID": "2",
                        "vehicleID": "707-008",
                        "checkpointInspectionID": "3"
                      },
                    );
                  })
            ],
          )
        ],
      ));
}

///Widget for showing that the current report hasn't been processed
Widget notProcessedWidget() {
  return BorderedContainer(
      width: 120,
      height: 45,
      backgroundColor: MyColors.grey500,
      borderColor: MyColors.grey1000,
      child: Row(
        children: const [
          Icon(
            Icons.warning,
            color: MyColors.grey1000,
          ),
          SizedBox(
            width: 10,
          ),
          Text(
            "Pending",
            style: MyTextStyles.buttonTextStyle,
          )
        ],
      ));
}

///Widget for showing that a checkpoint is non-conforming
Widget nonconforming() {
  return BorderedContainer(
      width: 190,
      height: 45,
      backgroundColor: MyColors.negativeAccent,
      borderColor: MyColors.negative,
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: const [
          Icon(
            Icons.warning,
            color: MyColors.negative,
          ),
          SizedBox(
            width: 10,
          ),
          Text(
            "Non-conforming",
            style: MyTextStyles.buttonTextStyle,
          )
        ],
      ));
}

///Widget for showing a checkpoint is conforming
Widget conforming() {
  return BorderedContainer(
      width: 160,
      height: 45,
      backgroundColor: MyColors.greenAcent,
      borderColor: MyColors.green,
      child: Row(
        children: const [
          Icon(
            Icons.check_circle,
            color: MyColors.green,
          ),
          SizedBox(
            width: 10,
          ),
          Text(
            "Conforming",
            style: MyTextStyles.buttonTextStyle,
          )
        ],
      ));
}

///Widget for building the title of the page showing metadata about the report
Widget reportTitleTile(bool pending, bool current) {
  return BorderedContainer(
      backgroundColor: MyColors.grey500,
      padding: const EdgeInsets.all(0),
      height: 70,
      borderRadius: 10,
      child: Center(
          child: ListTile(
              horizontalTitleGap: 0,
              title: const Text(
                "22/07/22",
                style: MyTextStyles.headerText1,
              ),
              subtitle: Row(
                children: [
                  const Icon(
                    Icons.location_on,
                    color: Colors.black,
                  ),
                  const Text("Reading"),
                  const SizedBox(
                    width: 16,
                  ),
                  pending ? pendingWidget() : processedWidget(),
                  const SizedBox(
                    width: 16,
                  ),
                  current ? upToDateWidget() : outdatedWidget()
                ],
              ),
              leading: const Icon(
                Icons.search,
                size: 40,
              ))));
}

///Class used in development to demonstrate UI
class CheckPoint {
  String name;
  bool conforming;

  CheckPoint(this.name, this.conforming);
}
