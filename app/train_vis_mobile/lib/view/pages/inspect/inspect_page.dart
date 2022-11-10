import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import 'package:go_router/go_router.dart';
import 'package:train_vis_mobile/controller/inspection_controller.dart';
import 'package:train_vis_mobile/model/inspection/checkpoint_inspection.dart';
import 'package:train_vis_mobile/model/inspection/vehicle_inspection.dart';
import 'package:train_vis_mobile/view/pages/inspect/capture/vehicle_inspection_capture_page_view.dart';
import 'package:train_vis_mobile/view/pages/inspect/inspect_progress_bar.dart';
import 'package:train_vis_mobile/view/pages/inspect/review/vehicle_inspection_review_container.dart';
import 'package:train_vis_mobile/view/pages/inspect/submit/vehicle_inspection_submit_container.dart';
import 'package:train_vis_mobile/view/theme/data/my_colors.dart';
import 'package:train_vis_mobile/view/theme/data/my_sizes.dart';
import 'package:train_vis_mobile/view/theme/data/my_text_styles.dart';
import 'package:train_vis_mobile/view/theme/widgets/my_icon_button.dart';
import 'package:train_vis_mobile/view/widgets/confirmation_dialog.dart';
import 'package:train_vis_mobile/view/widgets/padded_custom_scroll_view.dart';

/// Page to carry out inspection of a train vehicle.
///
/// TODO
class InspectPage extends StatefulWidget {
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
  // CREATE STATE //
  // //////////// //

  @override
  State<InspectPage> createState() => _InspectPageState();
}

/// TODO
class _InspectPageState extends State<InspectPage> {
  // STATE VARIABLES //
  late PageController pageController; // controller for pageview
  late List<CheckpointInspection> checkpointInspections; // TODO
  late bool isSubmitted; // if the inspection is being submitted
  late bool isOnSubmitPage; // if current page is submit

  // ////////// //
  // INIT STATE //
  // ////////// //

  @override
  void initState() {
    super.initState();

    // initializing state
    pageController = PageController();
    checkpointInspections = [];
    isSubmitted = false;
    isOnSubmitPage = false;
  }

  // //////////// //
  // BUILD METHOD //
  // //////////// //

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      // ///////////// //
      // CONFIGURATION //
      // ///////////// //

      backgroundColor: MyColors.backgroundSecondary,

      // /////// //
      // APP BAR //
      // /////// //

      appBar: AppBar(
        leading: !isSubmitted
            ? MyIconButton.secondary(
                iconData: FontAwesomeIcons.xmark,
                iconSize: MySizes.largeIconSize,
                onPressed: () {
                  // handling the close
                  _handleClose(context);
                },
              )
            : null,
        automaticallyImplyLeading: false,
        title: const Text("Inspect", style: MyTextStyles.headerText1),
      ),

      // //// //
      // BODY //
      // //// //

      body: SafeArea(
        child: WillPopScope(
          // disabling swipe to go back
          onWillPop: () async => false,
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

              const SliverToBoxAdapter(
                  child: SizedBox(height: MySizes.spacing)),

              // ///////////////// //
              // INSPECT PAGE VIEW //
              // ///////////////// //

              SliverFillRemaining(
                child: PageView(
                  controller: pageController,
                  physics: const NeverScrollableScrollPhysics(),
                  children: [
                    // //////////////////////////// //
                    // VEHICLE INSPECTION PAGE VIEW //
                    // //////////////////////////// //

                    VehicleInspectionCapturePageView(
                      vehicleID: widget.vehicleID,
                      onCaptured: (checkpointInspections) {
                        // handling the capture
                        _handleCaptured(checkpointInspections);
                      },
                    ),

                    // ////// //
                    // REVIEW //
                    // ////// //

                    VehicleInspectionReviewContainer(
                      checkpointInspections: checkpointInspections,
                      onSubmitted: (reviewedCheckpointInspections) {
                        // handing the submission
                        _handleSubmitted(reviewedCheckpointInspections);
                      },
                    ),

                    // ////// //
                    // SUBMIT //
                    // ////// //

                    VehicleInspectionSubmitContainer(
                      isSubmitted: isSubmitted,
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  // ////////////// //
  // HELPER METHODS //
  // ////////////// //

  /// TODO
  Future<void> _handleClose(BuildContext context) async {
    // displaying confirmation dialog
    bool result = await showDialog(
      context: context,
      builder: (BuildContext context) {
        return const ConfirmationDialog(
          title: "Cancel Inspection",
          message:
              "Are you sure you want to cancel this inspection? All progress will be lost.",
          falseText: "No",
          trueText: "Yes",
          trueBackgroundColor: MyColors.negative,
          trueTextColor: MyColors.antiNegative,
        );
      },
    );

    // acting based on result of dialog
    if (result && mounted) {
      // result true -> navigate to inspect

      // navigating to inspect page
      context.pop();
    } else {
      // result false -> do nothing

      // nothing
    }
  }

  /// TODO
  Future<void> _handleCaptured(
    List<CheckpointInspection> checkpointInspections,
  ) async {
    // updating state
    setState(() {
      this.checkpointInspections = checkpointInspections;
    });

    // navigating to review container
    pageController.nextPage(
      duration: const Duration(milliseconds: 500),
      curve: Curves.ease,
    );
  }

  /// TODO
  Future<void> _handleSubmitted(
    List<CheckpointInspection> reviewedCheckpointInspections,
  ) async {
    // updating state
    setState(() {
      isSubmitted = true;
      checkpointInspections = reviewedCheckpointInspections;
    });

    // creating the vehicle inspection object
    VehicleInspection vehicleInspection = VehicleInspection(
      vehicleID: widget.vehicleID,
    );

    // navigating to submit page
    pageController.nextPage(
      duration: const Duration(milliseconds: 500),
      curve: Curves.ease,
    );

    // updating state
    setState(() {
      isOnSubmitPage = true;
    });

    // adding the vehicle inspection to firestore
    await InspectionController.instance.addVehicleInspection(
      vehicleInspection,
      checkpointInspections,
    );

    // updating the submission status
    setState(() {
      isSubmitted = true;
    });
  }
}
