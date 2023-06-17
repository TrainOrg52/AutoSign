import 'package:auto_sign_mobile/model/enums/capture_type.dart';
import 'package:auto_sign_mobile/view/theme/data/my_colors.dart';
import 'package:auto_sign_mobile/view/theme/data/my_sizes.dart';
import 'package:auto_sign_mobile/view/theme/widgets/my_icon_button.dart';
import 'package:auto_sign_mobile/view/widgets/custom_future_builder.dart';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import 'package:percent_indicator/percent_indicator.dart';

// ////////// //
// MAIN CLASS //
// ////////// //

/// A container that allows for a certain type of camera media to be captured.
///
/// Either photo of video can be captured using the container, based by the
/// provided [captureType] property.
class CameraContainer extends StatelessWidget {
  // MEMBER VARIABLES //
  final CaptureType captureType; // capture type for the camera (photo of vid)
  final Function(String) onCaptured; // callback for when media captured

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  const CameraContainer({
    super.key,
    required this.captureType,
    required this.onCaptured,
  });

  // //////////// //
  // BUILD METHOD //
  // //////////// //

  @override
  Widget build(BuildContext context) {
    return CustomFutureBuilder<List<CameraDescription>>(
      future: availableCameras(),
      builder: (context, cameras) {
        return CameraContainerAux(
          captureType: captureType,
          cameras: cameras,
          onCapture: onCaptured,
        );
      },
    );
  }
}

// //////////////// //
// AUXILLIARY CLASS //
// //////////////// //

/// An auxilliary helper container that actually loads the camera using the provided
/// [camera] property.
class CameraContainerAux extends StatefulWidget {
  // MEMBER VARIABLES //
  final CaptureType captureType; // capture type for the camera.
  final List<CameraDescription> cameras; // cameras that can be used.
  final Function(String) onCapture; // callback for when media captured.

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  const CameraContainerAux({
    super.key,
    required this.captureType,
    required this.cameras,
    required this.onCapture,
  });

  // //////////// //
  // CREATE STATE //
  // //////////// //

  @override
  State<CameraContainerAux> createState() => _CameraContainerAuxState();
}

/// State class for [CameraContainerAux].
class _CameraContainerAuxState extends State<CameraContainerAux> {
  // STATE VARIABLES //
  late CameraController controller; // controller for camera
  late bool isInitialized; // initialization state of camera
  late bool isRecording; // current recording state of the capture
  late bool mediaCaptured; // if a photo/video has been captured or not
  late bool flashEnabled; // if flash is enabled

  // THEME-ING
  // sizes
  final double captureButtonOutlineRadius = 36;
  final double captureButtonOutlineWidth = 5;
  final double captureButtonRadius = 28;

  // ////////// //
  // INIT STATE //
  // ////////// //

  @override
  void initState() {
    super.initState();

    // initializing state
    isInitialized = false;
    isRecording = false;
    mediaCaptured = false;
    flashEnabled = false;

    // initializing controller
    controller = CameraController(
      widget.cameras.first, // first camera is rear camera
      ResolutionPreset.max,
      imageFormatGroup: ImageFormatGroup.bgra8888,
      enableAudio: false,
    );
    controller.initialize().then((_) {
      if (!mounted) {
        return;
      }
      setState(() async {
        isInitialized = true;
        if (widget.captureType == CaptureType.video) {
          controller.prepareForVideoRecording();
        }
      });
    }).catchError(
      (Object e) {
        if (e is CameraException) {
          switch (e.code) {
            case 'CameraAccessDenied':
              //print('User denied camera access.');
              break;
            default:
              //print('Handle other errors.');
              break;
          }
        }
      },
    );
  }

  // /////// //
  // DISPOSE //
  // /////// //

  @override
  void dispose() {
    super.dispose();

    // disposing state
    controller.dispose();
  }

  // //////////// //
  // BUILD METHOD //
  // //////////// //

  @override
  Widget build(BuildContext context) {
    // building UI based on initialization state
    if (!isInitialized) {
      // camera not initialized -> building error
      return const Center(child: CircularProgressIndicator());
    } else {
      // camera initialized -> building UI
      return Stack(
        children: [
          // ///////////// //
          // FLASH CONTROL //
          // //////////// //

          // Align(
          //   alignment: Alignment.topRight,
          //   child: _buildFlashControl(),
          // ),

          // ////////////// //
          // CAMERA PREVIEW //
          // ////////////// //

          Align(
            alignment: Alignment.bottomCenter,
            child: ClipRRect(
              borderRadius: BorderRadius.circular(MySizes.borderRadius * 2),
              child: CameraPreview(controller),
            ),
          ),

          // ////////////// //
          // CAPTURE BUTTON //
          // ////////////// //

          widget.captureType == CaptureType.photo
              ? _buildPhotoCaptureButton()
              : _buildVideoCaptureButton(),
        ],
      );
    }
  }

  // ////////////////////// //
  // HELPER BUILDER METHODS //
  // ////////////////////// //

  Widget _buildFlashControl() {
    return MyIconButton.secondary(
      iconData: FontAwesomeIcons.boltLightning,
      onPressed: () {
        setState(() {
          if (flashEnabled) {
            controller.setFlashMode(FlashMode.always);
          } else {
            controller.setFlashMode(FlashMode.off);
          }

          flashEnabled = !flashEnabled;
        });
      },
    );
  }

  /// Builds the capture button for the capture of a photo.
  Widget _buildPhotoCaptureButton() {
    return Align(
      alignment: Alignment.bottomCenter,
      child: Padding(
        padding: MySizes.padding,
        child: CircularPercentIndicator(
          radius: captureButtonOutlineRadius,
          lineWidth: captureButtonOutlineWidth,
          percent: 1.0,
          progressColor: MyColors.backgroundSecondary,
          center: OutlinedButton(
            style: OutlinedButton.styleFrom(
              shape: const CircleBorder(),
              backgroundColor: MyColors.grey100,
              foregroundColor: MyColors.textPrimary,
              side: const BorderSide(
                width: 0,
                color: MyColors.backgroundSecondary,
              ),
            ),
            onPressed: !mediaCaptured
                ? () async {
                    await _capturePhoto();
                  }
                : null,
            child: SizedBox(
              height: captureButtonRadius * 2,
              width: captureButtonRadius * 2,
            ),
          ),
        ),
      ),
    );
  }

  /// Builds the capture button for the capture of a video.
  ///
  /// The button changes dynamically based on the recording state of the container.
  Widget _buildVideoCaptureButton() {
    return Align(
      alignment: Alignment.bottomCenter,
      child: Padding(
        padding: MySizes.padding,
        child: CircularPercentIndicator(
          radius: captureButtonOutlineRadius,
          lineWidth: captureButtonOutlineWidth,
          backgroundColor: Colors.transparent,
          percent: 1.0,
          progressColor:
              isRecording ? Colors.transparent : MyColors.backgroundSecondary,
          center: SizedBox(
            height: captureButtonRadius * 2,
            width: captureButtonRadius * 2,
            child: OutlinedButton(
              style: OutlinedButton.styleFrom(
                shape: isRecording
                    ? RoundedRectangleBorder(
                        borderRadius:
                            BorderRadius.circular(MySizes.borderRadius * 2))
                    : const CircleBorder(),
                backgroundColor: MyColors.red,
                foregroundColor: MyColors.textPrimary,
                minimumSize: Size(captureButtonRadius, captureButtonRadius),
              ),
              onPressed: !mediaCaptured
                  ? () async {
                      await _captureVideo();
                    }
                  : null,
              child: Container(),
            ),
          ),
        ),
      ),
    );
  }

  // /////////////// //
  // CAPTURING MEDIA //
  // /////////////// //

  /// Handles the capturing of a photo.
  ///
  /// Takes a photo using the camera controller and passes it on to the handling
  /// method.
  Future<void> _capturePhoto() async {
    // updating state
    setState(() {
      mediaCaptured = true;
    });

    // capturing the photo
    XFile? photoFile = await controller.takePicture();

    // handling the capture
    widget.onCapture(photoFile.path);

    // resetting state
    setState(() {
      mediaCaptured = false;
    });
  }

  /// Handles the capturing of a video.
  ///
  /// Depending on the current state of the system (recording or not recording),
  /// this method will either start a recording or stop one.
  ///
  /// If it stops a recording, it passes the gathered video + sensor data onto
  /// the handling method.
  Future<void> _captureVideo() async {
    // checking if currently recording
    if (isRecording) {
      // currently recording -> need to stop recording

      // updating state
      setState(() {
        mediaCaptured = true;
        isRecording = false;
      });

      // gathering video
      XFile? videoFile = await controller.stopVideoRecording();

      // handling the capture
      widget.onCapture(videoFile.path);

      // resetting state
      setState(() {
        mediaCaptured = false;
      });
    } else {
      // not currently recording -> need to start recording

      // starting recording
      await controller.prepareForVideoRecording();
      await controller.startVideoRecording();

      // updating state
      setState(() {
        isRecording = true;
      });
    }
  }
}
