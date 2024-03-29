import 'package:auto_sign_mobile/view/theme/data/my_colors.dart';
import 'package:auto_sign_mobile/view/theme/data/my_sizes.dart';
import 'package:auto_sign_mobile/view/theme/data/my_text_styles.dart';
import 'package:flutter/material.dart';
import 'package:percent_indicator/percent_indicator.dart';

/// A custom progress bar for displaying the progress of a [VehicleInspection].
class InspectProgressBar extends StatelessWidget {
  // MEMBER VARIABLES //
  final double progress; // the progress to be shown on the bar

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  const InspectProgressBar({
    super.key,
    required this.progress,
  });

  // //////////// //
  // BUILD METHOD //
  // //////////// //

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: const [
            Expanded(
              flex: 10,
              child: Text(
                "Capture",
                style: MyTextStyles.bodyText2,
              ),
            ),
            Expanded(
              flex: 3,
              child: Text(
                "Review",
                style: MyTextStyles.bodyText2,
              ),
            ),
            Flexible(
              flex: 2,
              child: Text(
                "Submit",
                style: MyTextStyles.bodyText2,
              ),
            ),
          ],
        ),
        const SizedBox(height: MySizes.spacing),
        LinearPercentIndicator(
          backgroundColor: MyColors.greyAccent,
          progressColor: MyColors.primaryAccent,
          percent: progress,
          lineHeight: 15,
          barRadius: const Radius.circular(100),
          padding: EdgeInsets.zero,
          animation: true,
          animateFromLastPercent: true,
          animationDuration: 1500,
        ),
      ],
    );
  }
}
