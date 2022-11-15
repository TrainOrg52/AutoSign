import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import 'package:go_router/go_router.dart';
import 'package:train_vis_mobile/controller/inspection_controller.dart';
import 'package:train_vis_mobile/main.dart';
import 'package:train_vis_mobile/model/inspection/checkpoint_inspection.dart';
import 'package:train_vis_mobile/model/inspection/vehicle_inspection.dart';
import 'package:train_vis_mobile/view/pages/reports/reports.dart';
import 'package:train_vis_mobile/view/routes/routes.dart';
import 'package:train_vis_mobile/view/theme/data/my_colors.dart';
import 'package:train_vis_mobile/view/theme/data/my_sizes.dart';
import 'package:train_vis_mobile/view/theme/data/my_text_styles.dart';
import 'package:train_vis_mobile/view/theme/widgets/my_icon_button.dart';
import 'package:train_vis_mobile/view/widgets/bordered_container.dart';
import 'package:train_vis_mobile/view/widgets/colored_container.dart';
import 'package:train_vis_mobile/view/widgets/custom_stream_builder.dart';
import 'package:train_vis_mobile/view/widgets/padded_custom_scroll_view.dart';

///Page showing the summary of the checkpoints for a given report
///Currently contains dummy data just to demonstrate the UI
class ReportSummary extends StatelessWidget {
  String vehicleID;
  String vehicleInspectionID;

  ReportSummary({
    super.key,
    required this.vehicleID,
    required this.vehicleInspectionID,
  });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          "Inspection",
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
        stream: InspectionController.instance
            .getCheckpointInspectionsWhereVehicleInspectionIs(
                vehicleInspectionID),
        builder: (context, checkpoints) {
          return PaddedCustomScrollView(
            slivers: [
              SliverToBoxAdapter(
                child: dateStatusWidget(true),
              ),
              const SliverToBoxAdapter(
                child: SizedBox(height: MySizes.spacing),
              ),
              SliverToBoxAdapter(
                child: CustomStreamBuilder(
                  stream: InspectionController.instance
                      .getVehicleInspection(vehicleInspectionID),
                  builder: (context, inspection) {
                    return reportTitleTile(inspection);
                  },
                ),
              ),
              const SliverToBoxAdapter(
                child: SizedBox(height: MySizes.spacing),
              ),
              const SliverToBoxAdapter(
                child: Text(
                  "Report",
                  style: MyTextStyles.headerText2,
                ),
              ),
              const SliverToBoxAdapter(
                child: SizedBox(height: MySizes.spacing),
              ),
              SliverToBoxAdapter(
                child: CustomStreamBuilder(
                    stream: InspectionController.instance
                        .getVehicleInspection(vehicleInspectionID),
                    builder: (context, vehicleInspection) {
                      return inspectionStatusWidget(vehicleInspection);
                    }),
              ),
              const SliverToBoxAdapter(
                child: SizedBox(height: MySizes.spacing),
              ),
              SliverToBoxAdapter(
                child: _buildReportSummary(
                    context, checkpoints, vehicleInspectionID),
              ),
            ],
          );
        },
      ),
    );
  }
}

///Constructs a series of tiles for each of the checkpoints in the report
ListView _buildReportSummary(BuildContext context,
    List<CheckpointInspection> checkpoints, String vehicleInspectionID) {
  return ListView.builder(
    shrinkWrap: true,
    physics: const NeverScrollableScrollPhysics(),
    itemCount: checkpoints.length,
    itemBuilder: (_, index) {
      return Column(
        children: [
          checkpointViewWidget(checkpoints[index], context),
          if (index != checkpoints.length - 1)
            const SizedBox(height: MySizes.spacing),
        ],
      );
    },
  );
}

Widget dateStatusWidget(bool inDate) {
  // determining properties for container
  Color borderColor = inDate ? MyColors.green : MyColors.red;
  Color backgroundColor = inDate ? MyColors.greenAcent : MyColors.redAccent;
  IconData iconData = inDate
      ? FontAwesomeIcons.solidCircleCheck
      : FontAwesomeIcons.circleExclamation;
  String text = inDate ? "Inspection Up-to-date" : "Inspection Outdated";

  // returning container
  return BorderedContainer(
    borderColor: borderColor,
    backgroundColor: backgroundColor,
    child: Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Icon(
          iconData,
          size: MySizes.mediumIconSize,
          color: borderColor,
        ),
        const SizedBox(width: MySizes.spacing),
        Text(
          text,
          style: MyTextStyles.headerText3,
        ),
      ],
    ),
  );
}

///Widget for building the title of the page showing metadata about the report
Widget reportTitleTile(VehicleInspection inspection) {
  return ColoredContainer(
    color: MyColors.backgroundSecondary,
    padding: MySizes.padding,
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
                inspection.timestamp.toDate().toString(),
                style: MyTextStyles.headerText1,
              ),
              const SizedBox(height: MySizes.spacing / 2),
              Row(
                children: [
                  locationWidget(inspection.location),
                  const SizedBox(width: MySizes.spacing * 2),
                  processingStatusWidget(inspection.processingStatus),
                ],
              ),
            ],
          ),
        ),
      ],
    ),
  );
}

Widget inspectionStatusWidget(VehicleInspection vehicleInspection) {
  // returning container
  return BorderedContainer(
    borderColor: vehicleInspection.conformanceStatus.color,
    backgroundColor: vehicleInspection.conformanceStatus.accentColor,
    child: Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Icon(
          vehicleInspection.conformanceStatus.iconData,
          size: MySizes.mediumIconSize,
          color: vehicleInspection.conformanceStatus.color,
        ),
        const SizedBox(width: MySizes.spacing),
        Text(
          vehicleInspection.conformanceStatus.title.toCapitalized(),
          style: MyTextStyles.headerText3,
        ),
      ],
    ),
  );
}

Widget checkpointViewWidget(
  CheckpointInspection checkpointInspection,
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
        Routes.checkpointInspection,
        params: {
          "vehicleInspectionID": checkpointInspection.vehicleInspectionID,
          "vehicleID": checkpointInspection.vehicleID,
          "checkpointInspectionID": checkpointInspection.id,
          "checkpointID": checkpointInspection.checkpointID
        },
      );
    },
    child: SizedBox(
      height: 100,
      child: Row(
        mainAxisAlignment: MainAxisAlignment.start,
        children: [
          BorderedContainer(
            isDense: true,
            backgroundColor: Colors.transparent,
            padding: const EdgeInsets.all(MySizes.paddingValue / 2),
            child: CustomStreamBuilder(
              stream: InspectionController.instance
                  .getUnprocessedCheckpointInspectionImageDownloadURL(
                checkpointInspection.vehicleID,
                checkpointInspection.vehicleInspectionID,
                checkpointInspection.id,
              ),
              builder: (context, downloadURL) {
                return Image.network(downloadURL);
              },
            ),
          ),
          const SizedBox(width: MySizes.spacing),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  checkpointInspection.title,
                  style: MyTextStyles.headerText3,
                ),
                const Spacer(),
                BorderedContainer(
                  isDense: true,
                  borderColor: checkpointInspection.conformanceStatus.color,
                  backgroundColor:
                      checkpointInspection.conformanceStatus.accentColor,
                  padding: const EdgeInsets.all(MySizes.paddingValue / 2),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(
                        checkpointInspection.conformanceStatus.iconData,
                        color: checkpointInspection.conformanceStatus.color,
                        size: MySizes.smallIconSize,
                      ),
                      const SizedBox(width: MySizes.spacing),
                      Text(
                        checkpointInspection.conformanceStatus.title
                            .toTitleCase(),
                        style: MyTextStyles.bodyText2,
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
          const Center(
            child: Icon(
              FontAwesomeIcons.circleChevronRight,
              size: MySizes.mediumIconSize,
            ),
          ),
        ],
      ),
    ),
  );
}

///Widget with an amber warning for when a report is outdated
Widget outdatedWidget() {
  return Row(
    children: const [
      Icon(
        FontAwesomeIcons.circleExclamation,
        color: MyColors.amber,
      ),
      SizedBox(width: MySizes.spacing),
      Text(
        "Outdated",
        style: MyTextStyles.bodyText1,
      )
    ],
  );
}

///Widget with a green checkmark for when a report is the most recent available
Widget upToDateWidget() {
  return Row(
    children: const [
      Icon(
        FontAwesomeIcons.solidCircleCheck,
        size: MySizes.smallIconSize,
        color: MyColors.green,
      ),
      SizedBox(width: MySizes.spacing),
      Text(
        "Up to date",
        style: MyTextStyles.bodyText1,
      )
    ],
  );
}
