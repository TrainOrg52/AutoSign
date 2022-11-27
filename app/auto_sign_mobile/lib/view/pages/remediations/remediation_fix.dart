import 'package:auto_sign_mobile/view/pages/remediations/remediation_summary.dart';
import 'package:auto_sign_mobile/view/theme/data/my_colors.dart';
import 'package:auto_sign_mobile/view/theme/data/my_text_styles.dart';
import 'package:flutter/material.dart';

///Class for showing an image within the app
class RemediationFix extends StatefulWidget {
  @override
  RemediationFixState createState() => RemediationFixState();
}

///Stateful class showing the desired image.
class RemediationFixState extends State<RemediationFix> {
  final List<bool> toggleStates = <bool>[true, false, false];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
        appBar: AppBar(
          title: const Text(
            "Remediation Fix",
            style: MyTextStyles.headerText1,
          ),
          backgroundColor: MyColors.antiPrimary,
          centerTitle: true,
        ),
        body: Center(
            child: Column(
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            Row(crossAxisAlignment: CrossAxisAlignment.start, children: const [
              Text(
                "Issue",
                style: MyTextStyles.headerText1,
              )
            ]),
            Row(children: [Expanded(child: issue("Emergency Exit not found"))]),
            Row(crossAxisAlignment: CrossAxisAlignment.start, children: const [
              Text(
                "Photos",
                style: MyTextStyles.headerText1,
              )
            ]),
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
                minWidth: 129.5,
              ),
              children: const [
                Text(
                  "Inspection",
                  style: MyTextStyles.buttonTextStyle,
                ),
                Text(
                  "Remediation",
                  style: MyTextStyles.buttonTextStyle,
                ),
                Text(
                  "Expected",
                  style: MyTextStyles.buttonTextStyle,
                )
              ],
            ),
            SizedBox(
                width: 360,
                child: const Image(
                  image: NetworkImage(
                      "https://thumbs.dreamstime.com/b/new-york-city-subway-doors-inside-car-no-one-around-98492229.jpg"),
                  fit: BoxFit.fitWidth,
                )),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: const [
                Text(
                  "Entrance 1: Door",
                  style: MyTextStyles.headerText1,
                )
              ],
            )
          ],
        )));
  }
}
