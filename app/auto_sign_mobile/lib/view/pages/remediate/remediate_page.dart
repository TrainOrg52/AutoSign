import 'package:auto_sign_mobile/controller/shop_controller.dart';
import 'package:auto_sign_mobile/controller/vehicle_controller.dart';
import 'package:auto_sign_mobile/main.dart';
import 'package:auto_sign_mobile/model/enums/conformance_status.dart';
import 'package:auto_sign_mobile/model/vehicle/checkpoint.dart';
import 'package:auto_sign_mobile/model/vehicle/sign.dart';
import 'package:auto_sign_mobile/view/routes/routes.dart';
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
import 'package:go_router/go_router.dart';
import 'package:provider/provider.dart';

/// Page to carry out a remediation for a train vehicle.
///
/// TODO
class RemediatePage extends StatefulWidget {
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
  // CREATE STATE //
  // //////////// //

  @override
  State<RemediatePage> createState() => _RemediatePageState();
}

/// TODO
class _RemediatePageState extends State<RemediatePage> {
  // STATE VARIABLES //
  late Map<String, List<String>>
      cart; // signs in the cart (checkpointID -> sign ID)
  late bool allSignsAddedToCart;

  // ////////// //
  // INIT STATE //
  // ////////// //

  @override
  void initState() {
    // super state
    super.initState();

    // member state
    cart = {};
    allSignsAddedToCart = false;
  }

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
          stream: VehicleController.instance
              .getNonConformingCheckpointsWhereVehicleIs(widget.vehicleID),
          builder: (context, checkpoints) {
            // building the widget
            return Stack(
              children: [
                PaddedCustomScrollView(
                  slivers: [
                    // /////////////// //
                    // ADD ALL TO CART //
                    // /////////////// //

                    SliverToBoxAdapter(
                      child: _buildAddAllToCartContainer(checkpoints),
                    ),

                    const SliverToBoxAdapter(
                        child: SizedBox(height: MySizes.spacing)),

                    // /////////////////////////////// //
                    // CHECKPOINT REMEDIATE CONTAINERS //
                    // /////////////////////////////// //

                    for (Checkpoint checkpoint in checkpoints)
                      SliverToBoxAdapter(
                        child: Column(
                          children: [
                            _buildCheckpointRemediateContainer(
                              context,
                              checkpoint,
                            ),
                            const SizedBox(height: MySizes.spacing),
                          ],
                        ),
                      ),

                    // //////////////////////// //
                    // EXAMPLE DROP DOWN BUTTON //
                    // //////////////////////// //
                  ],
                ),

                // ////////////////// //
                // CHECKOUT CONTAINER //
                // ////////////////// //

                if (cart.isNotEmpty)
                  Align(
                    alignment: Alignment.bottomCenter,
                    child: Padding(
                      padding: const EdgeInsets.all(MySizes.paddingValue * 2),
                      child: _buildCheckoutContainer(context, checkpoints),
                    ),
                  ),
              ],
            );
          },
        ),
      ),
    );
  }

  /// TODO
  Widget _buildAddAllToCartContainer(List<Checkpoint> checkpoints) {
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
            children: [
              const Icon(
                FontAwesomeIcons.circleInfo,
                size: MySizes.mediumIconSize,
                color: MyColors.blue,
              ),
              const SizedBox(width: MySizes.spacing),
              Text(
                allSignsAddedToCart && cart.isNotEmpty
                    ? "All Signs Added To Cart"
                    : "Add Signs All to Cart",
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
            text: allSignsAddedToCart && cart.isNotEmpty
                ? "Undo"
                : "Add All to Cart",
            onPressed: () {
              // addding all of the signs to the cart
              _handleAddAllToCartPressed(checkpoints);
            },
          ),
        ],
      ),
    );
  }

  /// TODO
  Widget _buildCheckpointRemediateContainer(
    BuildContext context,
    Checkpoint checkpoint,
  ) {
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
                        .getCheckpointShowcaseDownloadURL(
                      checkpoint.vehicleID,
                      checkpoint.id,
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

                Text(
                  checkpoint.title,
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

          for (var sign in checkpoint.signs)
            if (sign.conformanceStatus == ConformanceStatus.nonConforming)
              Row(
                children: [
                  // /////////// //
                  // SIGN STATUS //
                  // /////////// //

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
                          size: MySizes.smallIconSize,
                          color: sign.conformanceStatus.color,
                        ),
                        const SizedBox(width: MySizes.spacing),
                        Text(
                          "${sign.title} : ${sign.conformanceStatus.toString().toCapitalized()}",
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
                    iconData: cart[checkpoint.id] != null
                        ? cart[checkpoint.id]!.contains(sign.id)
                            ? FontAwesomeIcons.circleCheck
                            : FontAwesomeIcons.cartPlus
                        : FontAwesomeIcons.cartPlus,
                    onPressed: () {
                      // handling the action
                      _handleAddToCartPressed(checkpoint, sign);
                    },
                  ),

                  const SizedBox(width: MySizes.spacing),

                  MyIconButton.secondary(
                    iconData: FontAwesomeIcons.hammer,
                    onPressed: () {
                      // remediating the issue
                      context.pushNamed(
                        Routes.signRemediate,
                        params: {
                          "vehicleID": checkpoint.vehicleID,
                          "checkpointID": checkpoint.id,
                          "signID": sign.id,
                        },
                      );
                    },
                  ),
                ],
              ),
        ],
      ),
    );
  }

  /// TODO
  Widget _buildCheckoutContainer(
    BuildContext context,
    List<Checkpoint> checkpoints,
  ) {
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
              // putting the cart into the controller
              Provider.of<ShopController>(context, listen: false).cart =
                  _flattenCart(checkpoints);

              // navigating to the checkout page
              context.pushNamed(
                Routes.checkout,
                params: {"vehicleID": widget.vehicleID},
              );
            },
          ),
        ],
      ),
    );
  }

  // ////////////// //
  // HELPER METHODS //
  // ////////////// //

  /// Handles the user pressing the toggle button to add/remove a sign to/from
  /// the cart.
  void _handleAddToCartPressed(Checkpoint checkpoint, Sign sign) {
    // updating the state of the cart
    setState(() {
      // checking if this checkpoint is in the cart
      if (cart[checkpoint.id] != null) {
        // checkpoint in the cart -> need to add/remove the sign to it

        // checking if the sign is in the cart
        if (cart[checkpoint.id]!.contains(sign.id)) {
          // sign in cart -> need to remove it

          // removing the sign from the cart
          cart[checkpoint.id]!.remove(sign.id);

          // seeing if the checkpoint needs to be removed from the cart
          if (cart[checkpoint.id]!.isEmpty) {
            // checkpoint has no signs in cart -> need to remove it form cart

            // removing checkpoint from cart
            cart.remove(checkpoint.id);
          }
        } else {
          // sign not in cart -> need to add it to cart

          // adding the sign to the cart
          cart[checkpoint.id]!.add(sign.id);
        }
      } else {
        // checkpoint not in cart -> need to initialize it with empty list

        // initializing the checkpoing with the sign
        cart[checkpoint.id] = [sign.id];
      }
    });
  }

  /// Adds the signs from all of the non-conformances in the checkpoints to the
  /// cart.
  void _handleAddAllToCartPressed(List<Checkpoint> checkpoints) {
    setState(() {
      // checking if add all to cart pressed already
      if (allSignsAddedToCart) {
        // add all to cart pressed already -> need to clear the cart

        // clearing the cart and resetting state
        cart = {};
        allSignsAddedToCart = false;
      } else {
        // not all signs added to cart yet -> need to add all signs to cart

        // iterating over the checkpoints
        for (Checkpoint checkpoint in checkpoints) {
          // adding the checkpoints signs to the cart
          cart[checkpoint.id] = [for (Sign sign in checkpoint.signs) sign.id];
        }

        // updating state
        allSignsAddedToCart = true;
      }
    });
  }

  /// Flattens the string into a single map.
  Map<String, int> _flattenCart(List<Checkpoint> checkpoints) {
    // creating empty map
    Map<String, int> flattenedCart = {};

    // creating the flattened cart

    // iterating over checkpoints in cart
    for (MapEntry<String, List<String>> checkpointSigns in cart.entries) {
      // converting the sign id to a title
      Checkpoint checkpoint = checkpoints
          .where((checkpoint) => checkpoint.id == checkpointSigns.key)
          .first;

      // iterating over the signs in the checkpoint
      for (String signID in checkpointSigns.value) {
        // getting the title of the sign
        Sign sign = checkpoint.signs.where((sign) => sign.id == signID).first;

        // checking if this sign is the flattened cart
        if (flattenedCart[sign.title] == null) {
          // sign not in cart -> need to add it

          // adding the sign to the cart
          flattenedCart[sign.title] = 1;
        } else {
          // sign in cart -> need to increment it

          // incrementing the quantity of the sign in the cart
          flattenedCart[sign.title] = flattenedCart[sign.title]! + 1;
        }
      }
    }

    // returning the flattened cart
    return flattenedCart;
  }
}
