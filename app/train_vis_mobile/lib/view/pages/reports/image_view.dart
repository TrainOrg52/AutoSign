import 'package:flutter/material.dart';
import 'package:train_vis_mobile/controller/inspection_controller.dart';
import 'package:train_vis_mobile/controller/vehicle_controller.dart';
import 'package:train_vis_mobile/view/theme/data/my_colors.dart';
import 'package:train_vis_mobile/view/theme/data/my_text_styles.dart';
import 'package:train_vis_mobile/view/widgets/bordered_container.dart';
import 'package:train_vis_mobile/view/widgets/custom_stream_builder.dart';
import 'package:train_vis_mobile/view/widgets/padded_custom_scroll_view.dart';

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
          ),
          body: PaddedCustomScrollView(
            slivers: [
              SliverToBoxAdapter(child: imageBuilder(context)),
            ],
          ),
        );
      },
    );
  }

  Center imageBuilder(BuildContext context) {
    return Center(
        child: Column(
      children: [
        CustomStreamBuilder(
            stream: InspectionController.instance
                .getCheckpointInspection(checkpointInspectionID),
            builder: (context, checkpoint) {
              return Row(
                children: [
                  Expanded(
                      child: BorderedContainer(
                          backgroundColor:
                              checkpoint.conformanceStatus.accentColor,
                          borderColor: checkpoint.conformanceStatus.color,
                          child: Row(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                Icon(checkpoint.conformanceStatus.iconData),
                                const SizedBox(
                                  width: 5,
                                ),
                                Text(checkpoint.conformanceStatus.description)
                              ])))
                ],
              );
            }),
        const SizedBox(
          height: 30,
        ),
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
        toggleStates[0] == false
            ? expectedImage(vehicleID, checkpointID)
            : actualImage(
                vehicleID, vehicleInspectionID, checkpointInspectionID),
        CustomStreamBuilder(
            stream: InspectionController.instance
                .getCheckpointInspection(checkpointInspectionID),
            builder: (context, checkpoint) {
              return Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  Text(
                    checkpoint.title,
                    style: MyTextStyles.headerText2,
                  )
                ],
              );
            })
      ],
    ));
  }
}

CustomStreamBuilder expectedImage(vehicleID, checkpointID) {
  return CustomStreamBuilder(
    stream: VehicleController.instance
        .getCheckpointImageDownloadURL(vehicleID, checkpointID),
    builder: (context, url) {
      return SizedBox(
          width: 250,
          child: Image(
            image: NetworkImage(url),
            fit: BoxFit.fitWidth,
          ));
    },
  );
}

CustomStreamBuilder actualImage(
    vehicleID, vehicleInspectionID, checkpointInspectionID) {
  return CustomStreamBuilder(
    stream: InspectionController.instance
        .getUnprocessedCheckpointInspectionImageDownloadURL(
            vehicleID, vehicleInspectionID, checkpointInspectionID),
    builder: (context, url) {
      return SizedBox(
          width: 250,
          child: Image(
            image: NetworkImage(url),
            fit: BoxFit.fitWidth,
          ));
    },
  );
}
