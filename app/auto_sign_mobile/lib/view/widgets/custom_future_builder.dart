import 'package:flutter/material.dart';

// ////////// //
// MAIN CLASS //
// ////////// //

/// A [FutureBuilder] that only shows the [builder] when the [future] has loaded,
/// otherwise it shows a "null widget" (progress indicator).
class CustomFutureBuilder<T> extends StatefulWidget {
  // MEMBER VARIABLES //
  final Future<T> future;
  final Duration nullWidgetDelay; // time to wait before showing loading widget
  final Widget Function(BuildContext, T) builder;

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  /// Constructs a new [CustomFutureBuilder] with the provided data.
  const CustomFutureBuilder({
    Key? key,
    required this.future,
    this.nullWidgetDelay = const Duration(milliseconds: 100),
    required this.builder,
  }) : super(key: key);

  // //////////// //
  // CREATE STATE //
  // //////////// //

  @override
  State<CustomFutureBuilder<T>> createState() => _CustomFutureBuilderState<T>();
}
// /////////// //
// STATE CLASS //
// /////////// //

/// State class for [CustomFutureBuilder].
class _CustomFutureBuilderState<T> extends State<CustomFutureBuilder<T>> {
  // MEMBER VARIABLES //
  late bool nullWidgetDelayTimerStarted;
  late bool nullWidgetDelayPassed;

  // ////////// //
  // INIT STATE //
  // ////////// //

  @override
  void initState() {
    // initializing super state
    super.initState();

    // initializing member state
    nullWidgetDelayTimerStarted = false;
    nullWidgetDelayPassed = false;
  }

  // //////////// //
  // BUILD METHOD //
  // //////////// //

  @override
  Widget build(BuildContext context) {
    // starting the timer (if it has not already been started)
    if (!nullWidgetDelayTimerStarted) {
      startNullWidgetDelayTimer();
    }

    // returning the streambuilder
    return FutureBuilder<T>(
      future: widget.future,
      builder: (context, snapshot) {
        // declaring the builder based on the state of the snapshot

        // snapshot has no data & delay not passed -> build empty container
        if (!snapshot.hasData && !nullWidgetDelayPassed) {
          return Container();
        }
        // snapshot has no data & delay passed -> build circular progress indicator
        else if (!snapshot.hasData && nullWidgetDelayPassed) {
          return const Center(child: CircularProgressIndicator());
        }

        // snapshot has data -> return normal builder
        return widget.builder(context, snapshot.data as T);
      },
    );
  }

  // ////////////// //
  // HELPER METHODS //
  // ////////////// //

  /// Starts the null widget delay timer.
  Future<void> startNullWidgetDelayTimer() async {
    // updating the state
    setState(() {
      nullWidgetDelayTimerStarted = true;
    });

    // waiting for delay time
    await Future.delayed(widget.nullWidgetDelay);

    // updating the state
    setState(() {
      nullWidgetDelayPassed = true;
    });
  }
}
