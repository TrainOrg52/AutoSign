import 'package:flutter/material.dart';
import 'package:train_vis_mobile/view/theme/data/my_sizes.dart';

/// A wrapper round [CustomScrollView] to add some padding within the scrollable
/// area of the view, not just around the whole view.
class PaddedCustomScrollView extends StatelessWidget {
  // MEMBER VARIABLES //
  // slivers
  final List<Widget> slivers; // the slivers to go into the scroll view
  // configuration
  final ScrollPhysics? scrollPhysics;
  // sizing information
  final double topPadding; // padding at the top of the scroll view
  final double bottomPadding; // padding at the bottom of the scroll view
  final double sidePadding; // padding at the sides of the scroll view

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  /// Constructs a new [PaddedCustomScrollView] with the provided information.
  const PaddedCustomScrollView({
    Key? key,
    // member variables
    required this.slivers,
    // configuration
    this.scrollPhysics,
    // sizing
    this.topPadding = MySizes.paddingValue * 2,
    this.bottomPadding = MySizes.paddingValue * 3,
    this.sidePadding = MySizes.paddingValue * 2,
  }) : super(key: key);

  // //////////// //
  // BUILD METHOD //
  // //////////// //

  @override
  Widget build(BuildContext context) {
    // building the padded custom scroll view based on the number of slivers
    if (slivers.length == 1) {
      return _buildOneWidgetView();
    } else if (slivers.length == 2) {
      return _buildTwoWidgetView();
    } else {
      return _buildMultiWidgetView();
    }
  }

  // ////////////////////// //
  // HELPER BUILDER METHODS //
  // ////////////////////// //

  /// Builds a [PaddedCustomScrollView] that contains 1 sliver.
  Widget _buildOneWidgetView() {
    // getting the sliver to be shown
    Widget sliver = slivers.first;

    // building the custom scroll view using the sliver.
    return CustomScrollView(
      controller: ScrollController(),
      physics: scrollPhysics,
      slivers: [
        SliverPadding(
          padding: EdgeInsets.only(
            top: topPadding,
            left: sidePadding,
            right: sidePadding,
            bottom: bottomPadding,
          ),
          sliver: sliver,
        ),
      ],
    );
  }

  /// Builds a [PaddedCustomScrollView] that contains 2 slivers.
  Widget _buildTwoWidgetView() {
    // getting the slivers to be shown
    Widget topSliver = slivers.first;
    Widget bottomSliver = slivers.last;

    // building the custom scroll view using the slivers
    return CustomScrollView(
      controller: ScrollController(),
      physics: scrollPhysics,
      slivers: [
        // top sliver
        SliverPadding(
          padding: EdgeInsets.only(
            top: topPadding,
            left: sidePadding,
            right: sidePadding,
          ),
          sliver: topSliver,
        ),
        // bottom sliver
        SliverPadding(
          padding: EdgeInsets.only(
            left: sidePadding,
            right: sidePadding,
            bottom: bottomPadding,
          ),
          sliver: bottomSliver,
        ),
      ],
    );
  }

  /// Builds a [PaddedCustomScrollView] that contains 3 or more slivers.
  Widget _buildMultiWidgetView() {
    // getting the slivers to go into the view
    Widget topSliver = slivers.first;
    List<Widget> middleSlivers = [...slivers];
    middleSlivers.removeAt(0); // removing first sliver (already in top sliver)
    middleSlivers.removeLast(); // removing last sliver (already in last sliver)
    Widget bottomSliver = slivers.last;

    // building the custom scroll view using the slivers
    return CustomScrollView(
      controller: ScrollController(),
      physics: scrollPhysics,
      slivers: [
        // top sliver
        SliverPadding(
          padding: EdgeInsets.only(
            top: topPadding,
            left: sidePadding,
            right: sidePadding,
          ),
          sliver: topSliver,
        ),
        // middle slivers
        for (Widget middleSliver in middleSlivers)
          SliverPadding(
            padding: EdgeInsets.only(
              left: sidePadding,
              right: sidePadding,
            ),
            sliver: middleSliver,
          ),
        // bottom sliver
        SliverPadding(
          padding: EdgeInsets.only(
            left: sidePadding,
            right: sidePadding,
            bottom: bottomPadding,
          ),
          sliver: bottomSliver,
        ),
      ],
    );
  }
}
