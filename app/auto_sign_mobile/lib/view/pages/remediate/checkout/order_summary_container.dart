import 'package:auto_sign_mobile/controller/shop_controller.dart';
import 'package:auto_sign_mobile/view/theme/data/my_colors.dart';
import 'package:auto_sign_mobile/view/theme/data/my_sizes.dart';
import 'package:auto_sign_mobile/view/theme/data/my_text_styles.dart';
import 'package:auto_sign_mobile/view/theme/widgets/my_icon_button.dart';
import 'package:auto_sign_mobile/view/theme/widgets/my_text_button.dart';
import 'package:auto_sign_mobile/view/widgets/colored_container.dart';
import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import 'package:provider/provider.dart';

/// TODO
class OrderSummaryContainer extends StatelessWidget {
  // MEMBER VARIABLES //
  final Function(BuildContext context)
      onSubmit; // called when the submit button is pressed

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  const OrderSummaryContainer({
    super.key,
    required this.onSubmit,
  });

  // //////////// //
  // BUILD METHOD //
  // //////////// //

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // ///// //
        // TITLE //
        // ///// //

        const Text("Order Summary", style: MyTextStyles.headerText2),

        const SizedBox(height: MySizes.spacing),

        // ///////////// //
        // ORDER SUMMARY //
        // ///////////// //

        _buildOrderSummaryContainer(context),

        const SizedBox(height: MySizes.spacing),

        // /////// //
        // ACTIONS //
        // /////// //

        _buildActionsContainer(context),

        const SizedBox(height: MySizes.spacing),
      ],
    );
  }

  // ////////////////////// //
  // HELPER BUILDER METHODS //
  // ////////////////////// //

  /// TODO
  Widget _buildOrderSummaryContainer(BuildContext context) {
    return ColoredContainer(
      color: MyColors.backgroundSecondary,
      child: ListView.builder(
        physics: const NeverScrollableScrollPhysics(),
        shrinkWrap: true,
        itemCount: Provider.of<ShopController>(context).cart.entries.length,
        itemBuilder: ((context, index) {
          return Column(
            children: [
              _buildSignOrderRow(
                context,
                Provider.of<ShopController>(context)
                    .cart
                    .entries
                    .toList()[index],
              ),
              if (index !=
                  Provider.of<ShopController>(context).cart.entries.length - 1)
                const SizedBox(height: MySizes.spacing),
            ],
          );
        }),
      ),
    );
  }

  /// TODO
  Widget _buildSignOrderRow(
      BuildContext context, MapEntry<String, int> signOrder) {
    return Row(
      children: [
        // ///// //
        // TITLE //
        // ///// //

        Expanded(
          child: Text(
            signOrder.key,
            style: MyTextStyles.headerText3,
          ),
        ),

        // ////////// //
        // - QUANTITY //
        // ////////// //

        MyIconButton.secondary(
          iconData: FontAwesomeIcons.squareMinus,
          onPressed: () {
            // reducing the quantity
            Provider.of<ShopController>(context, listen: false)
                .decrementSignQuantity(signOrder.key);
          },
        ),

        const SizedBox(width: MySizes.spacing / 2),

        // //////// //
        // QUANTITY //
        // //////// //

        Text(
          "${signOrder.value}",
          style: MyTextStyles.headerText3,
        ),

        const SizedBox(width: MySizes.spacing / 2),

        // ////////// //
        // + QUANTITY //
        // ////////// //

        MyIconButton.secondary(
          iconData: FontAwesomeIcons.squarePlus,
          onPressed: () {
            // increasing the quantity
            Provider.of<ShopController>(context, listen: false)
                .incrementSignQuantity(signOrder.key);
          },
        ),

        const SizedBox(width: MySizes.spacing / 2),

        // ////// //
        // DELETE //
        // ////// //

        MyIconButton.negative(
          iconData: FontAwesomeIcons.trash,
          onPressed: () {
            // removing the sign
            Provider.of<ShopController>(context, listen: false)
                .removeSignFromCart(signOrder.key);
          },
        ),
      ],
    );
  }

  /// TODO
  Widget _buildActionsContainer(BuildContext context) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        // ////// //
        // CANCEL //
        // ////// //

        MyTextButton.secondary(
          text: "Cancel",
          onPressed: () {
            // cancelling the purchase -> popping
            Navigator.of(context).pop(false);
          },
        ),

        const SizedBox(width: MySizes.spacing),

        // ////// //
        // SUBMIT //
        // ////// //

        MyTextButton.custom(
          text: "Submit",
          backgroundColor: MyColors.blue,
          borderColor: MyColors.blue,
          textColor: MyColors.antiPrimary,
          onPressed: () {
            // submitting the order using the callback
            onSubmit(context);
          },
        ),
      ],
    );
  }
}
