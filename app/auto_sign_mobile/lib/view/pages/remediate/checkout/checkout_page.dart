import 'package:auto_sign_mobile/view/pages/remediate/checkout/order_submit_container.dart';
import 'package:auto_sign_mobile/view/pages/remediate/checkout/order_summary_container.dart';
import 'package:auto_sign_mobile/view/theme/data/my_text_styles.dart';
import 'package:auto_sign_mobile/view/theme/widgets/my_icon_button.dart';
import 'package:auto_sign_mobile/view/widgets/padded_custom_scroll_view.dart';
import 'package:flutter/material.dart';

/// Page to carry out a remediation for a train vehicle.
///
/// TODO
class CheckoutPage extends StatefulWidget {
  // MEMBER VARIABLES //
  final String vehicleID;

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  const CheckoutPage({
    super.key,
    required this.vehicleID,
  });

  // //////////// //
  // CREATE STATE //
  // ///////////// //

  @override
  State<CheckoutPage> createState() => _CheckoutPageState();
}

/// TODO
class _CheckoutPageState extends State<CheckoutPage> {
  // STATE VARIABLES //
  late PageController pageController; // controller for pageview
  late bool isSubmitted; // if the order is being submitted
  late bool isOnSubmitPage; // if current page is submit

  // ////////// //
  // INIT STATE //
  // ////////// //

  @override
  void initState() {
    super.initState();

    // initializing state
    pageController = PageController();
    isSubmitted = false;
    isOnSubmitPage = false;
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
        leading: !isOnSubmitPage
            ? MyIconButton.back(
                onPressed: () {
                  Navigator.of(context).pop();
                },
              )
            : null,
        automaticallyImplyLeading: false,
        title: const Text("Checkout", style: MyTextStyles.headerText1),
      ),

      // //// //
      // BODY //
      // //// //

      body: SafeArea(
        child: PaddedCustomScrollView(
          scrollPhysics: const NeverScrollableScrollPhysics(),
          slivers: [
            SliverFillRemaining(
              child: PageView(
                controller: pageController,
                physics: const NeverScrollableScrollPhysics(),
                children: [
                  // /////// //
                  // SUMMARY //
                  // /////// //

                  OrderSummaryContainer(
                    onSubmit: _handleSubmit,
                  ),

                  // ////// //
                  // SUBMIT //
                  // ////// //

                  OrderSubmitContainer(isSubmitted: isSubmitted),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ////////////// //
  // HELPER METHODS //
  // ////////////// //

  // TODO
  Future<void> _handleSubmit() async {
    // navigating to the submit page
    pageController.nextPage(
      duration: const Duration(milliseconds: 500),
      curve: Curves.ease,
    );

    // updating state
    setState(() {
      isOnSubmitPage = true;
    });

    // submiting order
    // TODO Do this properly - just a dummy wait at the moment
    await Future.delayed(const Duration(seconds: 2));

    // updating state
    setState(() {
      isSubmitted = true;
    });
  }
}
