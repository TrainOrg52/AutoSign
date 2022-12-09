//import 'dart:html';

import 'package:auto_sign_mobile/controller/inspection_controller.dart';
import 'package:auto_sign_mobile/controller/vehicle_controller.dart';
import 'package:auto_sign_mobile/main.dart';
import 'package:auto_sign_mobile/model/inspection/checkpoint_inspection.dart';
import 'package:auto_sign_mobile/view/theme/data/my_colors.dart';
import 'package:auto_sign_mobile/view/theme/data/my_sizes.dart';
import 'package:auto_sign_mobile/view/theme/data/my_text_styles.dart';
import 'package:auto_sign_mobile/view/theme/widgets/my_icon_button.dart';
import 'package:auto_sign_mobile/view/widgets/bordered_container.dart';
import 'package:auto_sign_mobile/view/widgets/capture_preview.dart';
import 'package:auto_sign_mobile/view/widgets/colored_container.dart';
import 'package:auto_sign_mobile/view/widgets/custom_stream_builder.dart';
import 'package:auto_sign_mobile/view/widgets/padded_custom_scroll_view.dart';
import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';

///Class for showing an image within the app
class ImageView extends StatefulWidget {
  String vehicleID;
  String vehicleInspectionID;
  String checkpointInspectionID;
  String checkpointID;

  ImageView(this.vehicleID, this.vehicleInspectionID,
      this.checkpointInspectionID, this.checkpointID);

  @override
  ImageViewState createState() => ImageViewState(
      vehicleID, vehicleInspectionID, checkpointInspectionID, checkpointID);
}

///Stateful class showing the desired image.
class ImageViewState extends State<ImageView> {
  final List<bool> toggleStates = <bool>[true, false];

  String vehicleID;
  String vehicleInspectionID;
  String checkpointInspectionID;
  String checkpointID;

  ImageViewState(this.vehicleID, this.vehicleInspectionID,
      this.checkpointInspectionID, this.checkpointID);

  @override
  Widget build(BuildContext context) {
    return CustomStreamBuilder(
      stream: InspectionController.instance
          .getCheckpointInspection(checkpointInspectionID),
      builder: (context, checkpointInspection) {
        return Scaffold(
          appBar: AppBar(
            title: Text(
              checkpointInspection.title,
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
                child: reportWidget(checkpointInspection),
              ),
              const SliverToBoxAdapter(
                child: SizedBox(height: MySizes.spacing),
              ),
              SliverToBoxAdapter(
                child: imageBuilder(context, checkpointInspection),
              ),
            ],
          ),
        );
      },
    );
  }

  Widget conformanceStatusWidget(CheckpointInspection checkpointInspection) {
    return BorderedContainer(
      borderColor: checkpointInspection.conformanceStatus.color,
      backgroundColor: checkpointInspection.conformanceStatus.accentColor,
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(
            checkpointInspection.conformanceStatus.iconData,
            size: MySizes.mediumIconSize,
            color: checkpointInspection.conformanceStatus.color,
          ),
          const SizedBox(width: MySizes.spacing),
          Text(
            checkpointInspection.conformanceStatus.title.toCapitalized(),
            style: MyTextStyles.headerText3,
          ),
        ],
      ),
    );
  }

  Future<void> updateCheckpoint(BuildContext context) async {
    return showDialog<void>(
      context: context,
      barrierDismissible: false, // user must tap button!
      builder: (BuildContext context) {
        return AlertDialog(
          title: const Text('AlertDialog Title'),
          content: SingleChildScrollView(
            child: ListBody(
              children: const <Widget>[
                Text('This is a demo alert dialog.'),
                Text('Would you like to approve of this message?'),
              ],
            ),
          ),
          actions: <Widget>[
            TextButton(
              child: const Text('Approve'),
              onPressed: () {
                Navigator.of(context).pop();
              },
            ),
          ],
        );
      },
    );
  }

  Widget reportWidget(CheckpointInspection checkpointInspection) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(children: [
          const Text(
            "Report",
            style: MyTextStyles.headerText2,
          ),
          const Spacer(),
          IconButton(
            icon: const Icon(
              FontAwesomeIcons.pencil,
              size: 25,
            ),
            onPressed: () {
              updateCheckpoint(context);
            },
          )
        ]),
        const SizedBox(height: MySizes.spacing),
        conformanceStatusWidget(checkpointInspection),
        const SizedBox(height: MySizes.spacing),
        ColoredContainer(
          color: MyColors.backgroundSecondary,
          padding: MySizes.padding,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: signReportWidgets(checkpointInspection),
          ),
        ),
      ],
    );
  }

  List<Widget> signReportWidgets(CheckpointInspection checkpointInspection) {
    List<Widget> widgets = [];
    int count = 0;

    for (var sign in checkpointInspection.signs) {
      widgets.add(
        BorderedContainer(
          isDense: true,
          borderColor: sign.conformanceStatus.color,
          backgroundColor: sign.conformanceStatus.accentColor,
          padding: const EdgeInsets.all(MySizes.paddingValue / 2),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(
                sign.conformanceStatus.iconData,
                size: MySizes.smallIconSize,
                color: sign.conformanceStatus.color,
              ),
              const SizedBox(width: MySizes.spacing),
              Text(
                "${sign.title} : ${sign.conformanceStatus.toString().toCapitalized()}",
                style: MyTextStyles.bodyText2,
              ),
            ],
          ),
        ),
      );

      if (count != checkpointInspection.signs.length - 1) {
        widgets.add(const SizedBox(height: MySizes.spacing));
      }

      count++;
    }

    return widgets;
  }

  Widget imageBuilder(
    BuildContext context,
    CheckpointInspection checkpointInspection,
  ) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          "Images",
          style: MyTextStyles.headerText2,
        ),
        const SizedBox(height: MySizes.spacing),
        Center(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              ToggleButtons(
                onPressed: (int index) {
                  setState(() {
                    for (int i = 0; i < toggleStates.length; i++) {
                      toggleStates[i] = i == index;
                    }
                  });
                },
                isSelected: toggleStates,
                borderRadius: const BorderRadius.all(Radius.circular(8)),
                selectedBorderColor: MyColors.borderColor,
                selectedColor: Colors.white,
                fillColor: MyColors.primaryAccent,
                constraints: const BoxConstraints(
                  minHeight: 40.0,
                  minWidth: 150,
                ),
                children: const [
                  Text(
                    "Inspection",
                    style: MyTextStyles.buttonTextStyle,
                  ),
                  Text(
                    "Expected",
                    style: MyTextStyles.buttonTextStyle,
                  )
                ],
              ),
              const SizedBox(height: MySizes.spacing),
              toggleStates[0] == false
                  ? expectedImage(
                      vehicleID,
                      checkpointID,
                      checkpointInspection.captureType,
                    )
                  : inspectionImage(
                      vehicleID,
                      vehicleInspectionID,
                      checkpointInspectionID,
                      checkpointInspection.captureType,
                    ),
            ],
          ),
        ),
      ],
    );
  }
}

Widget inspectionImage(
    vehicleID, vehicleInspectionID, checkpointInspectionID, captureType) {
  return ColoredContainer(
    color: MyColors.backgroundSecondary,
    width: 300,
    padding: MySizes.padding,
    child: CustomStreamBuilder<String>(
      stream: InspectionController.instance
          .getCheckpointInspectionCaptureDownloadURL(vehicleID,
              vehicleInspectionID, checkpointInspectionID, captureType),
      builder: (context, downloadURL) {
        return CapturePreview(
          captureType: captureType,
          path: downloadURL,
          isNetworkURL: true,
        );
      },
    ),
  );
}

Widget expectedImage(vehicleID, checkpointID, captureType) {
  return ColoredContainer(
    color: MyColors.backgroundSecondary,
    width: 300,
    padding: MySizes.padding,
    child: CustomStreamBuilder<String>(
      stream: VehicleController.instance.getCheckpointDemoDownloadURL(
        vehicleID,
        checkpointID,
        captureType,
      ),
      builder: (context, downloadURL) {
        return CapturePreview(
          captureType: captureType,
          path: downloadURL,
          isNetworkURL: true,
        );
      },
    ),
  );
}
