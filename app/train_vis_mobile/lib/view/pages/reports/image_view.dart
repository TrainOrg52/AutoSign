import 'package:flutter/material.dart';
import 'package:train_vis_mobile/view/theme/data/my_colors.dart';
import 'package:train_vis_mobile/view/theme/data/my_text_styles.dart';
import 'package:train_vis_mobile/view/widgets/bordered_container.dart';

///Class for showing an image within the app
class ImageView extends StatefulWidget {
  @override
  ImageViewState createState() => ImageViewState();
}

///Stateful class showing the desired image.
class ImageViewState extends State<ImageView> {
  final List<bool> toggleStates = <bool>[true, false];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
        appBar: AppBar(
          title: const Text(
            "Report",
            style: MyTextStyles.headerText1,
          ),
          backgroundColor: MyColors.antiPrimary,
          centerTitle: true,
        ),
        body: Center(
            child: Column(
          children: [
            const Text(
              "22/06/22",
              style: MyTextStyles.headerText1,
            ),
            Row(
              children: [
                Expanded(
                    child: BorderedContainer(
                        backgroundColor: MyColors.negativeAccent,
                        borderColor: MyColors.negative,
                        child: Row(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: const [
                              Icon(Icons.warning),
                              Text("Nonconforming")
                            ])))
              ],
            ),
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
                minWidth: 194.5,
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
            const Image(
              image: NetworkImage(
                  "https://thumbs.dreamstime.com/b/new-york-city-subway-doors-inside-car-no-one-around-98492229.jpg"),
              fit: BoxFit.fitWidth,
            ),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: const [
                Text(
                  "Entrance 1: Door",
                  style: MyTextStyles.headerText2,
                )
              ],
            )
          ],
        )));
  }
}
