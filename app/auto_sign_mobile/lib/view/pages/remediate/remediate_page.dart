import 'package:auto_sign_mobile/controller/vehicle_controller.dart';
import 'package:auto_sign_mobile/main.dart';
import 'package:auto_sign_mobile/model/enums/conformance_status.dart';
import 'package:auto_sign_mobile/model/vehicle/checkpoint.dart';
import 'package:auto_sign_mobile/view/routes/routes.dart';
import 'package:auto_sign_mobile/view/theme/data/my_colors.dart';
import 'package:auto_sign_mobile/view/theme/data/my_sizes.dart';
import 'package:auto_sign_mobile/view/theme/data/my_text_styles.dart';
import 'package:auto_sign_mobile/view/theme/widgets/my_icon_button.dart';
import 'package:auto_sign_mobile/view/theme/widgets/my_text_button.dart';
import 'package:auto_sign_mobile/view/widgets/bordered_container.dart';
import 'package:auto_sign_mobile/view/widgets/colored_container.dart';
import 'package:auto_sign_mobile/view/widgets/custom_dropdown_button.dart';
import 'package:auto_sign_mobile/view/widgets/custom_stream_builder.dart';
import 'package:auto_sign_mobile/view/widgets/padded_custom_scroll_view.dart';
import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import 'package:go_router/go_router.dart';

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

class _RemediatePageState extends State<RemediatePage> {
  // STATE VARIABLES //
  late final Map<Checkpoint, Map<String, bool>> signsInCart;

  // ////////// //
  // INIT STATE //
  // ////////// //

  @override
  void initState() {
    // super state
    super.initState();

    signsInCart = {};
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
            // initializing the signs in cart object
            for (Checkpoint checkpoint in checkpoints) {
              // adding empty map for the checkpoint
              signsInCart[checkpoint] = {};

              // populating map for the checkpoint
              for (var sign in checkpoint.signs) {
                if (sign.entries.first.value ==
                    ConformanceStatus.nonConforming) {
                  signsInCart[checkpoint]?[sign.entries.first.key] = false;
                }
              }
            }

            // building the widget
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

                    for (Checkpoint checkpoint in checkpoints)
                      SliverToBoxAdapter(
                        child: _buildCheckpointRemediateContainer(
                          context,
                          checkpoint,
                        ),
                      ),

                    // //////////////////////// //
                    // EXAMPLE DROP DOWN BUTTON //
                    // //////////////////////// //

                    SliverToBoxAdapter(
                      child: CustomDropdownButton<ConformanceStatus>(
                        // value
                        value: ConformanceStatus
                            .conforming, // TODO change this to be the current conformance status of the sign
                        // on changed
                        onChanged:
                            (ConformanceStatus? conformanceStatus) async {
                          if (conformanceStatus != null) {
                            // updating the conformance status
                            // TODO
                          }
                        },
                        // items
                        items: ConformanceStatus.userSelectableValues
                            .map<DropdownMenuItem<ConformanceStatus>>(
                          (conformanceStatus) {
                            return DropdownMenuItem(
                              value: conformanceStatus,
                              child: BorderedContainer(
                                isDense: true,
                                borderColor: conformanceStatus.color,
                                backgroundColor: conformanceStatus.accentColor,
                                padding: const EdgeInsets.all(
                                    MySizes.paddingValue / 2),
                                child: Row(
                                  mainAxisSize: MainAxisSize.min,
                                  children: [
                                    Icon(
                                      conformanceStatus.iconData,
                                      size: MySizes.smallIconSize,
                                      color: conformanceStatus.color,
                                    ),
                                    const SizedBox(width: MySizes.spacing),
                                    Text(
                                      conformanceStatus.title.toCapitalized(),
                                      style: MyTextStyles.bodyText2,
                                    ),
                                  ],
                                ),
                              ),
                            );
                          },
                        ).toList(),
                      ),
                    ),
                  ],
                ),

                // ////////////////// //
                // CHECKOUT CONTAINER //
                // ////////////////// //

                Align(
                  alignment: Alignment.bottomCenter,
                  child: Padding(
                    padding: const EdgeInsets.all(MySizes.paddingValue * 2),
                    child: _buildCheckoutContainer(context),
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
            if (sign.entries.first.value == ConformanceStatus.nonConforming)
              Row(
                children: [
                  // /////////// //
                  // SIGN STATUS //
                  // /////////// //

                  BorderedContainer(
                    isDense: true,
                    borderColor: sign.entries.first.value.color,
                    backgroundColor: sign.entries.first.value.accentColor,
                    padding: const EdgeInsets.all(MySizes.paddingValue / 2),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(
                          sign.entries.first.value.iconData,
                          size: MySizes.smallIconSize,
                          color: sign.entries.first.value.color,
                        ),
                        const SizedBox(width: MySizes.spacing),
                        Text(
                          "${sign.entries.first.key} : ${sign.entries.first.value.toString().toCapitalized()}",
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
                      context.pushNamed(
                        Routes.signRemediate,
                        params: {"vehicleID": checkpoint.vehicleID},
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
  Widget _buildCheckoutContainer(BuildContext context) {
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

  /// Adds the given checkpoint sign to the cart.
  void _addSignToCart(Checkpoint checkpoint, String signID) {
    // updating the cart
  }

  /// Removes the given checkpoint sign from the cart.
  void _removeSignFromCart(Checkpoint checkpoint, String signID) {
    // updating the cart
  }

  /// Adds the signs from all of the non-conformances in the checkpoints to the
  /// cart.
  void _addAllSignsToCart(List<Checkpoint> checkpoints) {
    /// TODO
  }
}
