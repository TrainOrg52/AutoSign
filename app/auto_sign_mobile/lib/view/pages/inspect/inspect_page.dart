import 'package:auto_sign_mobile/controller/inspection_controller.dart';
import 'package:auto_sign_mobile/controller/vehicle_controller.dart';
import 'package:auto_sign_mobile/model/inspection/checkpoint_inspection.dart';
import 'package:auto_sign_mobile/model/inspection/vehicle_inspection.dart';
import 'package:auto_sign_mobile/model/vehicle/vehicle.dart';
import 'package:auto_sign_mobile/view/pages/inspect/capture/vehicle_inspection_capture_page_view.dart';
import 'package:auto_sign_mobile/view/pages/inspect/inspect_progress_bar.dart';
import 'package:auto_sign_mobile/view/pages/inspect/review/vehicle_inspection_review_page_view.dart';
import 'package:auto_sign_mobile/view/pages/inspect/submit/vehicle_inspection_submit_container.dart';
import 'package:auto_sign_mobile/view/theme/data/my_colors.dart';
import 'package:auto_sign_mobile/view/theme/data/my_sizes.dart';
import 'package:auto_sign_mobile/view/theme/data/my_text_styles.dart';
import 'package:auto_sign_mobile/view/theme/widgets/my_icon_button.dart';
import 'package:auto_sign_mobile/view/widgets/confirmation_dialog.dart';
import 'package:auto_sign_mobile/view/widgets/padded_custom_scroll_view.dart';
import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import 'package:go_router/go_router.dart';

/// Page to carry out inspection of a train vehicle.
///
/// The page consists of two elements:
///
/// 1: A progress bar to indicate the status of the inspection. This progress
/// bar is automatically updated during the inspection.
///
/// 2: A custom [PageView] that has one page for each of the stages of the inspection;
/// capture, review and submit.
class InspectPage extends StatefulWidget {
  // MEMBERS //
  final String vehicleID; // ID of vehicle being inspected

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

/// State class for [InspectPage].
class _InspectPageState extends State<InspectPage> {
  // STATE VARIABLES //
  late PageController pageController; // controller for pageview
  late double inspectionProgress; // progress value of the inspection
  late List<CheckpointInspection>
      checkpointInspections; // checkpoint inspections gathered during the inspection
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
    inspectionProgress = 0.05;
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

      // making the background all white (different to normal background)
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

              SliverToBoxAdapter(
                child: InspectProgressBar(progress: inspectionProgress),
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
                    // /////// //
                    // CAPTURE //
                    // /////// //

                    VehicleInspectionCapturePageView(
                      vehicleID: widget.vehicleID,
                      onVehicleInspectionCaptured: (checkpointInspections) {
                        // handling the capture
                        _handleVehicleInspectionCaptured(checkpointInspections);
                      },
                    ),

                    // ////// //
                    // REVIEW //
                    // ////// //

                    VehicleInspectionReviewPageView(
                      checkpointInspections: checkpointInspections,
                      onReviewed: (reviewedCheckpointInspections) {
                        // handing the submission
                        _handleReviewed(reviewedCheckpointInspections);
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

  /// Handles the closing of the [InspectPage].
  ///
  /// Displays a confirmation dialog to ensure the user wants to close, and if
  /// confirmed, returns the user to the [ProfilePage].
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

  /// Handles the capturing of the [VehicleInspection].
  ///
  /// Sets the [List] of [CheckpointInspection]s gathered from the
  /// [VehicleInspectionCapturePageView] into the state, and navigates to the
  /// next page (the review page).
  Future<void> _handleVehicleInspectionCaptured(
    List<CheckpointInspection> checkpointInspections,
  ) async {
    // updating state
    setState(() {
      this.checkpointInspections = checkpointInspections;
      inspectionProgress = 0.75; // TODO define better
    });

    // navigating to review container
    pageController.nextPage(
      duration: const Duration(milliseconds: 500),
      curve: Curves.ease,
    );
  }

  /// Handles the successful reviewing of the [VehicleInspection] within the
  /// review page.
  ///
  /// Sets the updated [List] of [CheckpointInspection]s gathered from the
  /// [VehicleInspectionReviewPageView] into the state, navigates to the submission
  /// page, and adds a new [VehicleInspection] into the system.
  Future<void> _handleReviewed(
    List<CheckpointInspection> reviewedCheckpointInspections,
  ) async {
    // updating state
    setState(() {
      checkpointInspections = reviewedCheckpointInspections;
      inspectionProgress = 1.0;
    });

    // getting the vehicle object
    Vehicle vehicle =
        await VehicleController.instance.getVehicleAtInstant(widget.vehicleID);

    // creating the vehicle inspection object
    VehicleInspection vehicleInspection = VehicleInspection.fromVehicle(
      vehicle: vehicle,
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
