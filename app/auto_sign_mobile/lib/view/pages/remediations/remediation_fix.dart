import 'package:auto_sign_mobile/view/pages/remediations/remediation_summary.dart';
import 'package:auto_sign_mobile/view/theme/data/my_colors.dart';
import 'package:auto_sign_mobile/view/theme/data/my_text_styles.dart';
import 'package:auto_sign_mobile/view/theme/widgets/my_text_button.dart';
import 'package:auto_sign_mobile/view/widgets/colored_container.dart';
import 'package:auto_sign_mobile/view/widgets/padded_custom_scroll_view.dart';
import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';

import '../../theme/data/my_sizes.dart';
import '../../widgets/bordered_container.dart';

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
            "Entrance 1 Door",
            style: MyTextStyles.headerText1,
          ),
          backgroundColor: MyColors.antiPrimary,
          centerTitle: true,
        ),
        body: PaddedCustomScrollView(
          slivers: [
            SliverToBoxAdapter(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.center,
                children: [
                  Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: const [
                        Text(
                          "Issue",
                          style: MyTextStyles.headerText1,
                        )
                      ]),
                  SizedBox(
                    height: MySizes.spacing,
                  ),
                  Row(children: [
                    BorderedContainer(
                      isDense: true,
                      borderColor: MyColors.negative,
                      backgroundColor: MyColors.negativeAccent,
                      padding: const EdgeInsets.all(MySizes.paddingValue / 2),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.end,
                        mainAxisSize: MainAxisSize.min,
                        children: const [
                          Icon(
                            FontAwesomeIcons.exclamation,
                            size: MySizes.smallIconSize,
                            color: MyColors.negative,
                          ),
                          SizedBox(width: MySizes.spacing),
                          Text(
                            "Emergency Exit Not Found",
                            style: MyTextStyles.bodyText1,
                          ),
                        ],
                      ),
                    )
                  ]),
                  SizedBox(
                    height: MySizes.spacing,
                  ),
                  Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: const [
                        Text(
                          "Action",
                          style: MyTextStyles.headerText1,
                        )
                      ]),
                  SizedBox(
                    height: MySizes.spacing,
                  ),
                  Row(children: [
                    BorderedContainer(
                      isDense: true,
                      borderColor: MyColors.green,
                      backgroundColor: MyColors.greenAccent,
                      padding: EdgeInsets.all(MySizes.paddingValue / 2),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.end,
                        mainAxisSize: MainAxisSize.min,
                        children: const [
                          Icon(
                            FontAwesomeIcons.recycle,
                            size: MySizes.smallIconSize,
                            color: MyColors.green,
                          ),
                          SizedBox(width: MySizes.spacing),
                          Text(
                            "Replaced",
                            style: MyTextStyles.bodyText1,
                          ),
                        ],
                      ),
                    )
                  ]),
                  SizedBox(
                    height: MySizes.spacing,
                  ),
                  Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: const [
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
                      minWidth: 110,
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
                    height: MySizes.spacing,
                  ),
                  const ColoredContainer(
                    color: MyColors.backgroundSecondary,
                    width: 300,
                    padding: MySizes.padding,
                    child: Image(
                        image: NetworkImage(
                            "https://thumbs.dreamstime.com/b/new-york-city-subway-doors-inside-car-no-one-around-98492229.jpg")),
                  ),
                ],
              ),
            )
          ],
        ));
  }
}
