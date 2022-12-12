import 'package:auto_sign_mobile/controller/inspection_controller.dart';
import 'package:auto_sign_mobile/main.dart';
import 'package:auto_sign_mobile/model/enums/processing_status.dart';
import 'package:auto_sign_mobile/model/inspection/vehicle_inspection.dart';
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

///Page for showing the list of reports associated with a train
///Currently contains dummy data just to demonstrate the UI
class InspectionsPage extends StatelessWidget {
  String vehicleID;

  InspectionsPage(this.vehicleID);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          "Inspections",
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
            .getVehicleInspectionsWhereVehicleIs(vehicleID),
        builder: (context, inspections) {
          return PaddedCustomScrollView(
            slivers: [
              SliverToBoxAdapter(
                child: _buildReportList(context, inspections, vehicleID),
              ),
            ],
          );
        },
      ),
    );
  }
}

///Constructs a series of tiles for each report
ListView _buildReportList(BuildContext context,
    List<VehicleInspection> inspections, String vehicleID) {
  return ListView.builder(
    shrinkWrap: true,
    physics: const NeverScrollableScrollPhysics(),
    itemCount: inspections.length,
    itemBuilder: ((context, index) {
      return Column(
        children: [
          reportTile(inspections[index], context, vehicleID),
          if (index != inspections.length - 1)
            const SizedBox(height: MySizes.spacing),
        ],
      );
    }),
  );
}

/// Widget which generates a tile for a given report object
Widget reportTile(
    VehicleInspection inspection, BuildContext context, String vehicleID) {
  bool processed = inspection.processingStatus == ProcessingStatus.processed;

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
        params: {"vehicleInspectionID": inspection.id, "vehicleID": vehicleID},
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

///Shows location icon followed by name of the location
Widget locationWidget(String location) {
  return Row(
    children: [
      const Icon(
        FontAwesomeIcons.locationDot,
        size: MySizes.smallIconSize,
        color: MyColors.textPrimary,
      ),
      const SizedBox(width: MySizes.spacing / 2),
      SizedBox(
        child: Text(
          location,
          style: MyTextStyles.bodyText1,
        ),
      ),
    ],
  );
}

Widget processingStatusWidget(ProcessingStatus processingStatus) {
  return Row(
    children: [
      Icon(
        processingStatus.iconData,
        color: processingStatus.color,
        size: MySizes.smallIconSize,
      ),
      const SizedBox(width: MySizes.spacing / 2),
      Text(
        processingStatus.title.toTitleCase(),
        style: MyTextStyles.bodyText1,
      ),
    ],
  );
}
