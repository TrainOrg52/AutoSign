import 'package:auto_sign_mobile/controller/vehicle_controller.dart';
import 'package:auto_sign_mobile/view/theme/data/my_colors.dart';
import 'package:auto_sign_mobile/view/theme/data/my_sizes.dart';
import 'package:auto_sign_mobile/view/theme/data/my_text_styles.dart';
import 'package:auto_sign_mobile/view/theme/widgets/my_icon_button.dart';
import 'package:auto_sign_mobile/view/theme/widgets/my_text_button.dart';
import 'package:auto_sign_mobile/view/widgets/bordered_container.dart';
import 'package:auto_sign_mobile/view/widgets/colored_container.dart';
import 'package:auto_sign_mobile/view/widgets/custom_stream_builder.dart';
import 'package:auto_sign_mobile/view/widgets/padded_custom_scroll_view.dart';
import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';

/// Page to carry out a remediation for a train vehicle.
///
/// TODO
class RemediatePage extends StatelessWidget {
  // MEMBERS //
  final String vehicleID;

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  const RemediatePage({
    super.key,
    required this.vehicleID,
  });

  // //////////// //
  // BUILD METHOD //
  // //////////// //

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      // /////// //
      // APP BAR //
      // /////// //

      appBar: AppBar(
        leading: MyIconButton.back(
          onPressed: () {
            Navigator.of(context).pop();
          },
        ),
        title: const Text("Remediate", style: MyTextStyles.headerText1),
      ),

      // //// //
      // BODY //
      // //// //

      body: SafeArea(
        child: CustomStreamBuilder(
          stream: VehicleController.instance.getVehicle(vehicleID),
          builder: (context, vehicle) {
            return Stack(
              children: [
                PaddedCustomScrollView(
                  slivers: [
                    // /////////////// //
                    // ADD ALL TO CART //
                    // /////////////// //

                    SliverToBoxAdapter(child: _buildAddAllToCartContainer()),

                    const SliverToBoxAdapter(
                        child: SizedBox(height: MySizes.spacing)),

                    // /////////////////////////////// //
                    // CHECKPOINT REMEDIATE CONTAINERS //
                    // /////////////////////////////// //

                    SliverToBoxAdapter(
                        child: _buildCheckpointRemediateContainer()),
                  ],
                ),

                // ////////////////// //
                // CHECKOUT CONTAINER //
                // ////////////////// //

                Align(
                  alignment: Alignment.bottomCenter,
                  child: Padding(
                    padding: const EdgeInsets.all(MySizes.paddingValue * 2),
                    child: _buildCheckoutContainer(),
                  ),
                ),
              ],
            );
          },
        ),
      ),
    );
  }

  // ////////////////////// //
  // HELPER BUILDER METHODS //
  // ////////////////////// //

  /// TODO
  Widget _buildAddAllToCartContainer() {
    return BorderedContainer(
      borderColor: MyColors.blue,
      backgroundColor: MyColors.blueAccent,
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          // /////// //
          // MESSAGE //
          // /////// //
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: const [
              Icon(
                FontAwesomeIcons.circleInfo,
                size: MySizes.mediumIconSize,
                color: MyColors.blue,
              ),
              SizedBox(width: MySizes.spacing),
              Text(
                "Add All Signs to Cart",
                style: MyTextStyles.headerText3,
              ),
            ],
          ),

          const SizedBox(height: MySizes.spacing),

          // ////// //
          // BUTTON //
          // ////// //

          MyTextButton.custom(
            backgroundColor: MyColors.blue,
            borderColor: MyColors.blue,
            textColor: MyColors.antiPrimary,
            text: "Add All to Cart",
            onPressed: () {
              // addding all of the signs to the cart
              // TODO
            },
          ),
        ],
      ),
    );
  }

  /// TODO
  Widget _buildCheckpointRemediateContainer() {
    return ColoredContainer(
      color: MyColors.backgroundSecondary,
      padding: MySizes.padding,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            height: 100,
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // ///// //
                // IMAGE //
                // ///// //

                BorderedContainer(
                  isDense: true,
                  backgroundColor: Colors.transparent,
                  padding: const EdgeInsets.all(MySizes.paddingValue / 2),
                  child: CustomStreamBuilder(
                    stream: VehicleController.instance
                        .getCheckpointImageDownloadURL(
                      "707-008",
                      "JlFY9HXxzEoqYQfs6ZXk",
                    ),
                    builder: (context, downloadURL) {
                      return Image.network(downloadURL);
                    },
                  ),
                ),

                const SizedBox(width: MySizes.spacing),

                // ///// //
                // TITLE //
                // ///// //

                const Text(
                  "Platform 1 Door",
                  style: MyTextStyles.headerText3,
                ),
              ],
            ),
          ),

          const SizedBox(height: MySizes.spacing),

          // ////// //
          // ISSUES //
          // ////// //

          const Text("Issues", style: MyTextStyles.bodyText1),

          const SizedBox(height: MySizes.spacing),

          Row(
            children: [
              // /////////// //
              // SIGN STATUS //
              // /////////// //

              BorderedContainer(
                isDense: true,
                borderColor: MyColors.red,
                backgroundColor: MyColors.redAccent,
                padding: const EdgeInsets.all(MySizes.paddingValue / 2),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: const [
                    Icon(
                      FontAwesomeIcons.circleExclamation,
                      color: MyColors.red,
                      size: MySizes.smallIconSize,
                    ),
                    SizedBox(width: MySizes.spacing),
                    Text(
                      "Evacuation Instructions Missing",
                      style: MyTextStyles.bodyText2,
                    ),
                  ],
                ),
              ),

              const Spacer(),

              // //////////// //
              // SIGN ACTIONS //
              // //////////// //

              MyIconButton.secondary(
                iconData: FontAwesomeIcons.cartPlus,
                onPressed: () {
                  // adding the sign to the cart
                  // TODO
                },
              ),

              const SizedBox(width: MySizes.spacing),

              MyIconButton.secondary(
                iconData: FontAwesomeIcons.hammer,
                onPressed: () {
                  // remediating the issue
                  // TODO
                },
              ),
            ],
          ),
        ],
      ),
    );
  }

  /// TODO
  Widget _buildCheckoutContainer() {
    return BorderedContainer(
      borderColor: MyColors.blue,
      backgroundColor: MyColors.blueAccent,
      child: Row(
        children: [
          // //// //
          // ICON //
          // //// //
          const Icon(
            FontAwesomeIcons.cartShopping,
            color: MyColors.blue,
            size: MySizes.mediumIconSize,
          ),

          const SizedBox(width: MySizes.spacing),

          // /////// //
          // MESSAGE //
          // /////// //

          const Text("4 signs in cart", style: MyTextStyles.bodyText1),

          const Spacer(),

          // /////////////// //
          // CHECKOUT BUTTON //
          // /////////////// //

          MyTextButton.custom(
            backgroundColor: MyColors.blue,
            borderColor: MyColors.blue,
            textColor: MyColors.antiPrimary,
            text: "Checkout",
            onPressed: () {
              // going to the checkout
              // TODO
            },
          ),
        ],
      ),
    );
  }
}
