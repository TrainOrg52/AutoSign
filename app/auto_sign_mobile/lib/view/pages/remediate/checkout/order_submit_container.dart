import 'package:auto_sign_mobile/controller/vehicle_controller.dart';
import 'package:auto_sign_mobile/model/enums/conformance_status.dart';
import 'package:auto_sign_mobile/model/vehicle/vehicle.dart';
import 'package:auto_sign_mobile/view/routes/routes.dart';
import 'package:auto_sign_mobile/view/theme/data/my_colors.dart';
import 'package:auto_sign_mobile/view/theme/data/my_sizes.dart';
import 'package:auto_sign_mobile/view/theme/data/my_text_styles.dart';
import 'package:auto_sign_mobile/view/theme/widgets/my_text_button.dart';
import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

/// TODO
class OrderSubmitContainer extends StatelessWidget {
  // MEMBER VARIABLES //
  final String vehicleID; // id of vehicle order is for
  final bool isSubmitted; // submission status of inspection

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  const OrderSubmitContainer({
    super.key,
    required this.vehicleID,
    required this.isSubmitted,
  });

  // //////////// //
  // BUILD METHOD //
  // //////////// //

  @override
  Widget build(BuildContext context) {
    // building based on submission status
    if (isSubmitted) {
      // inspection submitted -> build submitted container
      return _buildSubmittedContainer(context);
    } else {
      // inspection not submitted -> build submitting container
      return _buildSubmittingContainer();
    }
  }

  // ////////////////////// //
  // HELPER BUILDER METHODS //
  // ////////////////////// //

  /// Builds the container for when the inspection is being submitted.
  ///
  /// This container contains a title, and an indeterminant progress indicator.
  Widget _buildSubmittingContainer() {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: const [
        // ///// //
        // TITLE //
        // ///// //

        Text(
          "Submitting Order",
          style: MyTextStyles.headerText1,
          textAlign: TextAlign.center,
        ),

        SizedBox(height: MySizes.spacing),

        // ////// //
        // PROMPT //
        // ////// //

        Text(
          "Please wait for your order to be submitted",
          style: MyTextStyles.bodyText1,
          textAlign: TextAlign.center,
        ),

        SizedBox(height: MySizes.spacing * 3),

        // ////////////////// //
        // PROGRESS INDICATOR //
        // ////////////////// //

        SizedBox(
          height: 45,
          width: 45,
          child: CircularProgressIndicator(
            color: MyColors.blueAccent,
            strokeWidth: 5,
          ),
        ),
      ],
    );
  }

  /// Builds the container shown when the inspection has submitted.
  ///
  /// This container includes a title message, and a button to return to the
  /// [ProfilePage].
  Widget _buildSubmittedContainer(BuildContext context) {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        // ///// //
        // TITLE //
        // ///// //

        const Text(
          "Order Complete",
          style: MyTextStyles.headerText1,
          textAlign: TextAlign.center,
        ),

        const SizedBox(height: MySizes.spacing),

        // ////// //
        // PROMPT //
        // ////// //

        const Text(
          "Your order was successfully submitted",
          style: MyTextStyles.bodyText1,
          textAlign: TextAlign.center,
        ),

        const SizedBox(height: MySizes.spacing * 3),

        // ///////////// //
        // FINISH BUTTON //
        // ///////////// //

        MyTextButton.custom(
          text: "Finish",
          backgroundColor: MyColors.blue,
          borderColor: MyColors.blue,
          textColor: MyColors.antiPrimary,
          onPressed: () async {
            // getting the current state of the vehicle
            Vehicle vehicle =
                await VehicleController.instance.getVehicleAtInstant(vehicleID);

            print(vehicle.conformanceStatus);

            // redirecting based on conformance status of vehicle
            if (vehicle.conformanceStatus == ConformanceStatus.conforming) {
              GoRouter.of(context).goNamed(
                Routes.profile,
                params: {"vehicleID": vehicle.id},
              );
            } else {
              // navigating to remediate screen (by popping)
              Navigator.of(context).pop();
            }
          },
        ),
      ],
    );
  }
}
