/// TODO
class CaptureType {
  // MEMBER VARIABLES //
  final String title; // title for the capture type
  final String fileExtension; // file extension for media of this type

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  /// Constructs a new [CaptureType] with the provided title.
  ///
  /// Private so that only the pre-defined [CaptureType] instances
  /// can be used.
  const CaptureType._({
    required this.title,
    required this.fileExtension,
  });

  // ///////////////// //
  // STRING CONVERTION //
  // ///////////////// //

  /// Creates a [CaptureType] based on the given string.
  static CaptureType? fromString(String status) {
    // returning value based on string
    if (status == photo.title) {
      return photo;
    } else if (status == video.title) {
      return video;
    }

    // no matching value -> return null
    else {
      return null;
    }
  }

  /// Converts the [CaptureType] object to a string representation.
  @override
  String toString() {
    // returning the title of the status
    return title;
  }

  // ///////// //
  // INSTANCES //
  // ///////// //

  // all instances
  static const List<CaptureType> values = [
    photo,
    video,
  ];

  // photo
  static const CaptureType photo = CaptureType._(
    title: "photo",
    fileExtension: "png",
  );

  // video
  static const CaptureType video = CaptureType._(
    title: "video",
    fileExtension: "mp4",
  );
}
