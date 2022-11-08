import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import 'package:train_vis_mobile/view/pages/inspect/capture/vehicle_inspection_capture_page_view.dart';
import 'package:train_vis_mobile/view/pages/inspect/inspect_progress_bar.dart';
import 'package:train_vis_mobile/view/pages/inspect/review/vehicle_inspection_review_container.dart';
import 'package:train_vis_mobile/view/pages/inspect/submit/vehicle_inspection_submit_container.dart';
import 'package:train_vis_mobile/view/theme/data/my_colors.dart';
import 'package:train_vis_mobile/view/theme/data/my_sizes.dart';
import 'package:train_vis_mobile/view/theme/data/my_text_styles.dart';
import 'package:train_vis_mobile/view/theme/widgets/my_icon_button.dart';
import 'package:train_vis_mobile/view/widgets/padded_custom_scroll_view.dart';

/// Page to carry out inspection of a train vehicle.
///
/// TODO
class InspectPage extends StatelessWidget {
  // MEMBERS //
  final String vehicleID;

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  const InspectPage({
    super.key,
    required this.vehicleID,
  });

  // //////////// //
  // BUILD METHOD //
  // //////////// //

  @override
  Widget build(BuildContext context) {
    // page view controller
    PageController pageController = PageController(initialPage: 1);

    return Scaffold(
      // ///////////// //
      // CONFIGURATION //
      // ///////////// //

      backgroundColor: MyColors.backgroundSecondary,

      // /////// //
      // APP BAR //
      // /////// //

      appBar: AppBar(
        leading: MyIconButton.secondary(
          iconData: FontAwesomeIcons.xmark,
          iconSize: MySizes.largeIconSize,
          onPressed: () {
            Navigator.of(context).pop();
          },
        ),
        title: const Text("Inspect", style: MyTextStyles.headerText1),
      ),

      // //// //
      // BODY //
      // //// //

      body: SafeArea(
        child: PaddedCustomScrollView(
          scrollPhysics: const NeverScrollableScrollPhysics(),
          topPadding: MySizes.paddingValue / 4,
          slivers: [
            // //////////// //
            // PROGRESS BAR //
            // //////////// //

            const SliverToBoxAdapter(
              child: InspectProgressBar(progress: 0.1),
            ),

            const SliverToBoxAdapter(child: SizedBox(height: MySizes.spacing)),

            // ///////////////// //
            // INSPECT PAGE VIEW //
            // ///////////////// //

            SliverFillRemaining(
              child: PageView(
                controller: pageController,
                physics: const NeverScrollableScrollPhysics(),
                children: const [
                  // //////////////////////////// //
                  // VEHICLE INSPECTION PAGE VIEW //
                  // //////////////////////////// //

                  VehicleInspectionCapturePageView(),

                  // ////// //
                  // REVIEW //
                  // ////// //

                  VehicleInspectionReviewContainer(),

                  // ////// //
                  // SUBMIT //
                  // ////// //

                  VehicleInspectionSubmitContainer(),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
