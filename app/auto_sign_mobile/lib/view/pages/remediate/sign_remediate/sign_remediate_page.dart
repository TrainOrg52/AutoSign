import 'package:auto_sign_mobile/controller/remediation_controller.dart';
import 'package:auto_sign_mobile/controller/vehicle_controller.dart';
import 'package:auto_sign_mobile/main.dart';
import 'package:auto_sign_mobile/model/enums/capture_type.dart';
import 'package:auto_sign_mobile/model/enums/remediation_action.dart';
import 'package:auto_sign_mobile/model/remediation/sign_remediation.dart';
import 'package:auto_sign_mobile/model/vehicle/checkpoint.dart';
import 'package:auto_sign_mobile/model/vehicle/sign.dart';
import 'package:auto_sign_mobile/view/pages/remediate/sign_remediate/sign_remediate_progress_bar.dart';
import 'package:auto_sign_mobile/view/theme/data/my_colors.dart';
import 'package:auto_sign_mobile/view/theme/data/my_sizes.dart';
import 'package:auto_sign_mobile/view/theme/data/my_text_styles.dart';
import 'package:auto_sign_mobile/view/theme/widgets/my_icon_button.dart';
import 'package:auto_sign_mobile/view/theme/widgets/my_text_button.dart';
import 'package:auto_sign_mobile/view/widgets/bordered_container.dart';
import 'package:auto_sign_mobile/view/widgets/camera_container.dart';
import 'package:auto_sign_mobile/view/widgets/capture_preview.dart';
import 'package:auto_sign_mobile/view/widgets/confirmation_dialog.dart';
import 'package:auto_sign_mobile/view/widgets/custom_stream_builder.dart';
import 'package:auto_sign_mobile/view/widgets/padded_custom_scroll_view.dart';
import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import 'package:go_router/go_router.dart';

/// TODO
class SignRemediatePage extends StatefulWidget {
  // MEMBERS //
  final String vehicleID; // ID of vehicle being remediated
  final String checkpointID; // ID of checkpoint being remediated
  final String signID; // ID of sign being remediated

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  const SignRemediatePage({
    super.key,
    required this.vehicleID,
    required this.checkpointID,
    required this.signID,
  });

  // //////////// //
  // CREATE STATE //
  // //////////// //

  @override
  State<SignRemediatePage> createState() => _SignRemediatePageState();
}

/// State class for [SignRemediatePage].
class _SignRemediatePageState extends State<SignRemediatePage> {
  // STATE VARIABLES //
  late PageController pageController; // controller for pageview
  late Sign sign; // sign being remediated
  late double remediateProgress; // progress value of the remediation
  late RemediationAction remediationAction; // action taken to remediate
  late String capturePath; // capture path of photo of remediation
  late bool isSubmitted; // if the remediation is being submitted
  late bool isOnSubmitPage; // if current page is submit

  // THEME-ING //
  // sizes
  final double actionButtonHeight = 100;

  // ////////// //
  // INIT STATE //
  // ////////// //

  @override
  void initState() {
    super.initState();

    // initializing state
    pageController = PageController();
    remediateProgress = 0.05;
    capturePath = "";
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
        leading: !isOnSubmitPage
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
        title: const Text("Remediate", style: MyTextStyles.headerText1),
      ),

      // //// //
      // BODY //
      // //// //

      body: SafeArea(
        child: WillPopScope(
          // disabling swipe to go back
          onWillPop: () async => false,
          child: CustomStreamBuilder(
            stream:
                VehicleController.instance.getCheckpoint(widget.checkpointID),
            builder: (context, checkpoint) {
              // extracting the sign from the checkpoint
              sign = checkpoint.signs
                  .where((sign) => sign.id == widget.signID)
                  .first;

              // building the widget
              return PaddedCustomScrollView(
                scrollPhysics: const NeverScrollableScrollPhysics(),
                topPadding: MySizes.paddingValue / 4,
                slivers: [
                  // //////////// //
                  // PROGRESS BAR //
                  // //////////// //

                  SliverToBoxAdapter(
                    child:
                        SignRemediateProgressBar(progress: remediateProgress),
                  ),

                  const SliverToBoxAdapter(
                    child: SizedBox(height: MySizes.spacing),
                  ),

                  // ///// //
                  // TITLE //
                  // ///// //

                  SliverToBoxAdapter(child: _buildTitleContainer(checkpoint)),

                  const SliverToBoxAdapter(
                    child: SizedBox(height: MySizes.spacing),
                  ),

                  // ////// //
                  // PROMPT //
                  // ////// //

                  // //////////////////////// //
                  // SIGN REMEDIATE PAGE VIEW //
                  // //////////////////////// //

                  SliverFillRemaining(
                    child: PageView(
                      controller: pageController,
                      physics: const NeverScrollableScrollPhysics(),
                      children: [
                        // /// //
                        // LOG //
                        // /// //

                        _buildActionPage(),

                        // /////// //
                        // CAPTURE //
                        // /////// //

                        _buildCapturePage(),

                        // ////// //
                        // REVIEW //
                        // ////// //

                        _buildReviewPage(),

                        // ////// //
                        // SUBMIT //
                        // ////// //

                        // building based on submission status
                        if (isSubmitted)
                          // remediation submitted -> build submitted container
                          _buildSubmittedContainer(context)
                        else
                          // remediation not submitted -> build submitting container
                          _buildSubmittingContainer()
                      ],
                    ),
                  ),
                ],
              );
            },
          ),
        ),
      ),
    );
  }

  // ////////////////////// //
  // HELPER BUILDER METHODS //
  // ////////////////////// //

  /// TODO
  Widget _buildTitleContainer(Checkpoint checkpoint) {
    // building the widget
    return Column(
      children: [
        // //////////////// //
        // CHECKPOINT TITLE //
        // //////////////// //

        Text(
          checkpoint.title,
          style: MyTextStyles.headerText2,
        ),

        const SizedBox(height: MySizes.spacing),

        // ///////////////////// //
        // NON-CONFORMANCE TITLE //
        // ///////////////////// //

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
                color: sign.conformanceStatus.color,
                size: MySizes.smallIconSize,
              ),
              const SizedBox(width: MySizes.spacing),
              Text(
                "${sign.title} : ${sign.conformanceStatus.toString().toTitleCase()}",
                style: MyTextStyles.bodyText2,
              ),
            ],
          ),
        ),
      ],
    );
  }

  /// TODO
  Widget _buildActionPage() {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        // ////// //
        // PROMPT //
        // ////// //

        const Text(
          "Please select the remediation action.",
          style: MyTextStyles.bodyText1,
        ),

        const SizedBox(height: MySizes.spacing),

        // ////////////// //
        // ACTION BUTTONS //
        // ////////////// //

        Row(
          children: [
            // /////// //
            // CLEANED //
            // /////// //

            Expanded(
              child: _buildActionButton(
                text: "Cleaned",
                icon: FontAwesomeIcons.broom,
                onPressed: () {
                  // updating the state
                  setState(() {
                    remediateProgress += 0.31;
                    remediationAction = RemediationAction.cleaned;
                  });

                  // navigating to the next page
                  pageController.nextPage(
                    duration: const Duration(milliseconds: 500),
                    curve: Curves.ease,
                  );
                },
              ),
            ),

            const SizedBox(width: MySizes.spacing),

            // //////// //
            // REPLACED //
            // //////// //

            Expanded(
              child: _buildActionButton(
                text: "Replaced",
                icon: FontAwesomeIcons.arrowsRotate,
                onPressed: () {
                  // updating the state
                  setState(() {
                    remediateProgress += 0.31;
                    remediationAction = RemediationAction.replaced;
                  });

                  // navigating to the next page
                  pageController.nextPage(
                    duration: const Duration(milliseconds: 500),
                    curve: Curves.ease,
                  );
                },
              ),
            ),
          ],
        ),
      ],
    );
  }

  /// Builds an instance of the 'action button' within the container.
  Widget _buildActionButton({
    required String text,
    required IconData icon,
    Function()? onPressed,
  }) {
    return OutlinedButton(
      style: OutlinedButton.styleFrom(
        foregroundColor: MyColors.textPrimary,
        backgroundColor: MyColors.backgroundSecondary,
        padding: MySizes.padding,
        side: const BorderSide(
          width: 0,
          color: MyColors.lineColor,
        ),
      ),
      onPressed: onPressed,
      child: SizedBox(
        height: actionButtonHeight,
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              icon,
              color: MyColors.textPrimary,
              size: MySizes.largeIconSize,
            ),
            const SizedBox(height: MySizes.spacing),
            Text(
              text,
              style: MyTextStyles.headerText1,
            ),
          ],
        ),
      ),
    );
  }

  /// Builds the page that allows the user to capture an image of the checkpoing.
  Widget _buildCapturePage() {
    return CameraContainer(
      captureType: CaptureType.photo,
      onCaptured: (capturePath) {
        // handling capture

        // updating photo data
        setState(() {
          this.capturePath = capturePath;
        });

        // navigating to review page
        pageController.animateToPage(
          2,
          duration: const Duration(milliseconds: 500),
          curve: Curves.ease,
        );

        // updating state
        setState(() {
          remediateProgress += 0.31;
        });
      },
    );
  }

  /// Builds the page that allows the user to review the image of the checkpoint
  /// that they have captured and retake/confirm it.
  Widget _buildReviewPage() {
    return Column(
      children: [
        // ////// //
        // PROMPT //
        // ////// //

        const Text(
          "Are you happy with this photo?",
          style: MyTextStyles.bodyText1,
        ),

        const Spacer(),

        // ////////////// //
        // CAPTURED IMAGE //
        // ////////////// //

        Expanded(
          flex: 12,
          child: CapturePreview(
            captureType: CaptureType.photo,
            path: capturePath,
          ),
        ),

        const Spacer(),

        // /////// //
        // ACTIONS //
        // /////// //

        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            // ////// //
            // RETAKE //
            // ////// //

            MyTextButton.secondary(
              text: "Retake",
              onPressed: () {
                // navigating back to the capture page
                pageController.animateToPage(
                  1,
                  duration: const Duration(milliseconds: 500),
                  curve: Curves.ease,
                );

                // updating state
                setState(() {
                  remediateProgress -= 0.31;
                });
              },
            ),

            const SizedBox(width: MySizes.spacing),

            // //// //
            // NEXT //
            // //// //

            MyTextButton.primary(
              text: "Submit",
              onPressed: () {
                _handleSubmit();
              },
            )
          ],
        ),
      ],
    );
  }

  /// Builds the container for when the inspection is being submitted.
  ///
  /// This container contains a title, and an indeterminant progress indicator.
  Widget _buildSubmittingContainer() {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: const [
        // ///// //
        // TITLE //
        // ///// //

        Text(
          "Submitting Remediation",
          style: MyTextStyles.headerText1,
          textAlign: TextAlign.center,
        ),

        SizedBox(height: MySizes.spacing),

        // ////// //
        // PROMPT //
        // ////// //

        Text(
          "Please wait for your remediation to be submitted",
          style: MyTextStyles.bodyText1,
          textAlign: TextAlign.center,
        ),

        SizedBox(height: MySizes.spacing * 3),

        // ////////////////// //
        // PROGRESS INDICATOR //
        // ////////////////// //

        SizedBox(
          height: 45,
          width: 45,
          child: CircularProgressIndicator(
            color: MyColors.primaryAccent,
            strokeWidth: 5,
          ),
        ),
      ],
    );
  }

  /// Builds the container shown when the inspection has submitted.
  ///
  /// This container includes a title message, and a button to return to the
  /// [ProfilePage].
  Widget _buildSubmittedContainer(BuildContext context) {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        // ///// //
        // TITLE //
        // ///// //

        const Text(
          "Remediation Complete",
          style: MyTextStyles.headerText1,
          textAlign: TextAlign.center,
        ),

        const SizedBox(height: MySizes.spacing),

        // ////// //
        // PROMPT //
        // ////// //

        const Text(
          "Your remediation was successfully submitted",
          style: MyTextStyles.bodyText1,
          textAlign: TextAlign.center,
        ),

        const SizedBox(height: MySizes.spacing * 3),

        // ///////////// //
        // FINISH BUTTON //
        // ///////////// //

        MyTextButton.custom(
          text: "Finish",
          backgroundColor: MyColors.primary,
          borderColor: MyColors.primary,
          textColor: MyColors.antiPrimary,
          onPressed: () {
            // navigating back to remediate screen
            Navigator.of(context).pop();
          },
        ),
      ],
    );
  }

  // ////////////// //
  // HELPER METHODS //
  // ////////////// //

  /// Handles the closing of the [SignRemediatePage].
  ///
  /// Displays a confirmation dialog to ensure the user wants to close, and if
  /// confirmed, returns the user to the [RemediatePage].
  Future<void> _handleClose(BuildContext context) async {
    // displaying confirmation dialog
    bool result = await showDialog(
      context: context,
      builder: (BuildContext context) {
        return const ConfirmationDialog(
          title: "Cancel Remediation",
          message:
              "Are you sure you want to cancel this remediation? All progress will be lost.",
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

  /// Handles the submission of the sign remediation to the system.
  Future<void> _handleSubmit() async {
    // navigating to submit page
    pageController.nextPage(
      duration: const Duration(milliseconds: 500),
      curve: Curves.ease,
    );

    // updating state
    setState(() {
      isOnSubmitPage = true;
      remediateProgress += 0.32;
    });

    // getting checkpoint inspection object
    Checkpoint checkpoint = await VehicleController.instance
        .getCheckpointAtInstant(widget.checkpointID);

    // creating the sign remediation object
    SignRemediation signRemediation = SignRemediation.fromSign(
      sign: sign,
      checkpointID: checkpoint.id,
      checkpointTitle: checkpoint.title,
      checkpointCaptureType: checkpoint.captureType,
      checkpointInspectionID: checkpoint.lastCheckpointInspectionID,
      remediationAction: remediationAction,
      capturePath: capturePath,
    );

    // submitting the sign remediation
    await RemediationController.instance.addSignRemediation(
      signRemediation,
      checkpoint.vehicleID,
      checkpoint.lastVehicleInspectionID,
    );

    // updating state
    setState(() {
      isSubmitted = true;
    });
  }
}
