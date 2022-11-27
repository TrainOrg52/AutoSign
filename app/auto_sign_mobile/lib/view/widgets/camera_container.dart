import 'package:auto_sign_mobile/view/theme/data/my_colors.dart';
import 'package:auto_sign_mobile/view/theme/data/my_sizes.dart';
import 'package:auto_sign_mobile/view/widgets/custom_future_builder.dart';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:percent_indicator/percent_indicator.dart';

// ////////// //
// MAIN CLASS //
// ////////// //

/// TODO
class CameraContainer extends StatelessWidget {
  // MEMBER VARIABLES //
  final Function(String) onCaptured;

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  const CameraContainer({
    super.key,
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

/// TODO
class CameraContainerAux extends StatefulWidget {
  // MEMBER VARIABLES //
  final List<CameraDescription> cameras;
  final Function(String) onCapture;

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  const CameraContainerAux({
    super.key,
    required this.cameras,
    required this.onCapture,
  });

  // //////////// //
  // CREATE STATE //
  // //////////// //

  @override
  State<CameraContainerAux> createState() => _CameraContainerAuxState();
}

/// TODO
class _CameraContainerAuxState extends State<CameraContainerAux> {
  // STATE VARIABLES //
  late CameraController controller; // controller for camera
  late bool isInitialized; // initialization state of camera
  late bool photoCaptured; // if a photo has been captured or not

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
    photoCaptured = false;

    // initializing controller
    controller = CameraController(
      widget.cameras.first, // first camera is rear camera
      ResolutionPreset.max,
      imageFormatGroup: ImageFormatGroup.bgra8888,
    );
    controller.initialize().then((_) {
      if (!mounted) {
        return;
      }
      setState(() {
        isInitialized = true;
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

          Align(
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
                  onPressed: !photoCaptured
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
          ),
        ],
      );
    }
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
      photoCaptured = true;
    });

    // capturing the photo
    XFile? photoFile = await controller.takePicture();

    // handling the capture
    widget.onCapture(photoFile.path);

    // resetting state
    setState(() {
      photoCaptured = false;
    });
  }
}
