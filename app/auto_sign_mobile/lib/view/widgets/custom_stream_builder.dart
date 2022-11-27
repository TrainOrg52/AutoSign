import 'package:flutter/material.dart';

// ////////// //
// MAIN CLASS //
// ////////// //

/// A [StreamBuilder] that only shows the [builder] when the latest snapshot
/// of the stream has data, otherwise it shows a "null widget" (progress
/// indicator).
class CustomStreamBuilder<T> extends StatefulWidget {
  // MEMBER VARIABLES //
  final Stream<T> stream;
  final Duration nullWidgetDelay; // time to wait before showing null widget
  final Widget Function(BuildContext, T) builder;

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  /// Constructs a new [CustomStreamBuilder] with the provided data.
  const CustomStreamBuilder({
    Key? key,
    required this.stream,
    this.nullWidgetDelay = const Duration(milliseconds: 100),
    required this.builder,
  }) : super(key: key);

  // //////////// //
  // CREATE STATE //
  // //////////// //

  @override
  State<CustomStreamBuilder<T>> createState() => _CustomStreamBuilderState<T>();
}
// /////////// //
// STATE CLASS //
// /////////// //

/// State class for [CustomStreamBuilder].
class _CustomStreamBuilderState<T> extends State<CustomStreamBuilder<T>> {
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
    return StreamBuilder<T>(
      stream: widget.stream,
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

    // updating the state if still mounted
    if (mounted) {
      setState(() {
        nullWidgetDelayPassed = true;
      });
    }
  }
}
