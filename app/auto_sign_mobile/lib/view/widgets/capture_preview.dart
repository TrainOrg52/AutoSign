import 'dart:io';

import 'package:auto_sign_mobile/model/enums/capture_type.dart';
import 'package:auto_sign_mobile/view/theme/data/my_sizes.dart';
import 'package:auto_sign_mobile/view/widgets/bordered_container.dart';
import 'package:flick_video_player/flick_video_player.dart';
import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';

/// TODO
class CapturePreview extends StatefulWidget {
  // MEMBER VARIABLES //
  final CaptureType captureType; // type of captured media (photo of video)
  final String path; // path to the captured media
  final bool isNetworkURL; // if the path to the media is a network URL

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  const CapturePreview({
    super.key,
    required this.captureType,
    required this.path,
    this.isNetworkURL = false,
  });

  // //////////// //
  // CREATE STATE //
  // //////////// //

  @override
  State<CapturePreview> createState() => _CapturePreviewState();
}

/// TODO
class _CapturePreviewState extends State<CapturePreview> {
  // STATE VARIABLES //
  late FlickManager flickManager; // manager for video player

  // ////////// //
  // INIT STATE //
  // ////////// //

  @override
  void initState() {
    // super state
    super.initState();

    // member state
    if (widget.captureType == CaptureType.video) {
      // initializing the video player
      flickManager = FlickManager(
        autoPlay: false,
        videoPlayerController: widget.isNetworkURL
            ? VideoPlayerController.network(
                widget.path,
              )
            : VideoPlayerController.file(File(widget.path)),
      );

      // muting the video player
      flickManager.flickControlManager?.mute();
    }
  }

  // ///////////// //
  // DISPOSE STATE //
  // ///////////// //

  @override
  void dispose() {
    // super state
    super.dispose();

    // member state
    if (widget.captureType == CaptureType.video) {
      flickManager.dispose();
    }
  }

  // //////////// //
  // BUILD METHOD //
  // //////////// //

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      child: BorderedContainer(
        isDense: true,
        backgroundColor: Colors.transparent,
        padding: const EdgeInsets.all(MySizes.paddingValue),
        child: widget.captureType == CaptureType.photo
            // ///////////// //
            // PHOTO PREVIEW //
            // ///////////// //

            ? widget.isNetworkURL
                ? Image.network(widget.path)
                : Image.file(File(widget.path))

            // ///////////// //
            // VIDEO PREVIEW //
            // ///////////// //

            : FlickVideoPlayer(
                flickManager: flickManager,
                flickVideoWithControls: SafeArea(
                  child: AspectRatio(
                    aspectRatio: 9 / 16,
                    child: FlickVideoWithControls(
                      videoFit: BoxFit.fitHeight,
                      controls: FlickPortraitControls(
                        iconSize: MySizes.largeIconSize,
                        progressBarSettings: FlickProgressBarSettings(),
                      ),
                    ),
                  ),
                ),
              ),
      ),
    );
  }
}
