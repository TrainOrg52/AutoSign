import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'package:train_vis_mobile/view/routes/routes.dart';
import 'package:train_vis_mobile/view/theme/data/my_colors.dart';
import 'package:train_vis_mobile/view/theme/data/my_text_styles.dart';
import 'package:train_vis_mobile/view/widgets/bordered_container.dart';

///Page for showing the list of reports associated with a train
///Currently contains dummy data just to demonstrate the UI
class ReportsPage extends StatelessWidget {
  String vehicleID;

  ReportsPage(this.vehicleID);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(
          "Inspections - $vehicleID",
          style: MyTextStyles.headerText1,
        ),
        backgroundColor: MyColors.antiPrimary,
        centerTitle: true,
      ),
      body: _buildReportList(context),
    );
  }
}

///Constructs a series of tiles for each report
ListView _buildReportList(BuildContext context) {
  //Dummy reports to populate the UI
  List<Report> reports = [
    Report("22/06/22", "Reading", true, false, []),
    Report("22/05/22", "Newport", true, true, []),
    Report("22/05/22", "Leeds", true, true, []),
    Report("22/04/22", "Reading", true, true, []),
    Report("22/03/22", "London", true, false, [])
  ];

  return ListView.builder(
      padding: EdgeInsets.zero,
      itemCount: reports.length * 2,
      itemBuilder: (_, index) {
        if (index.isEven) {
          return const Divider(
            height: 8,
          );
        }
        return reportTile(reports[index ~/ 2], context);
      });
}

/// Widget which generates a tile for a given report object
Widget reportTile(Report report, BuildContext context) {
  return BorderedContainer(
      padding: const EdgeInsets.all(0),
      height: 70,
      borderRadius: 10,
      child: Center(
          child: ListTile(
              horizontalTitleGap: 0,
              title: Text(
                report.date,
                style: MyTextStyles.headerText1,
              ),
              subtitle: Row(
                children: [
                  locationWidget(report.location),
                  const SizedBox(
                    width: 16,
                  ),
                  report.processed ? processedWidget() : pendingWidget()
                ],
              ),
              leading: const Icon(
                Icons.search,
                size: 40,
              ),
              trailing: IconButton(
                  icon: Icon(
                    Icons.navigate_next_sharp,
                    color: Colors.black,
                    size: 40,
                  ),
                  onPressed: () {
                    context.pushNamed(
                      Routes.vehicleInspection,
                      params: {
                        "vehicleInspectionID": "2",
                        "vehicleID": "707-008"
                      },
                    );
                  }))));
}

///Shows location icon followed by name of the location
Widget locationWidget(String location) {
  return Row(
    children: [
      const Icon(
        Icons.location_on,
        color: Colors.black,
      ),
      SizedBox(width: 60, child: Text(location)),
    ],
  );
}

///Widget with a green checkmark for when a report has been processed
Widget processedWidget() {
  return Row(
    children: const [
      Icon(
        Icons.check_circle,
        color: MyColors.green,
      ),
      Text("Processed")
    ],
  );
}

///Widget with an amber warning for when a report is pending processing
Widget pendingWidget() {
  return Row(
    children: const [
      Icon(
        Icons.warning,
        color: MyColors.amber,
      ),
      Text("Pending")
    ],
  );
}

///Widget with an amber warning for when a report is outdated
Widget outdatedWidget() {
  return Row(
    children: const [
      Icon(
        Icons.warning,
        color: MyColors.amber,
      ),
      Text("Outdated")
    ],
  );
}

///Widget with a green checkmark for when a report is the most recent available
Widget upToDateWidget() {
  return Row(
    children: const [
      Icon(
        Icons.check_circle,
        color: MyColors.green,
      ),
      Text("Up to date")
    ],
  );
}

///Report class used for convenience in development
class Report {
  String location;
  String date;
  bool outdated;
  bool processed;
  List<TrainImage> imageList;

  Report(
      this.date, this.location, this.outdated, this.processed, this.imageList);
}

///Class used in development but no longer needed
class TrainImage {
  String sectionName;
  AssetImage sectionImg;
  AssetImage expectedImg;
  bool conforming;

  TrainImage(
      this.sectionName, this.conforming, this.sectionImg, this.expectedImg);
}
