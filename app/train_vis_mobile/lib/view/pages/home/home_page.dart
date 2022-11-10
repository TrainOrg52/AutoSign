import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'package:train_vis_mobile/view/routes/routes.dart';
import 'package:train_vis_mobile/view/theme/data/my_colors.dart';
import 'package:train_vis_mobile/view/theme/data/my_text_styles.dart';
import 'package:train_vis_mobile/view/theme/widgets/my_text_button.dart';
import 'package:train_vis_mobile/view/theme/widgets/my_text_field.dart';
import 'package:train_vis_mobile/view/widgets/colored_container.dart';
import 'package:train_vis_mobile/view/widgets/padded_custom_scroll_view.dart';

/// The home page of the application.
///
/// Displays a single container that allows the user to enter the ID of a train
/// vehicle and view the inforrmation for this train.
class HomePage extends StatefulWidget {
  // THEME-ING //
  // sizes
  final double containerHeight = 160;

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  const HomePage({super.key});

  // //////////// //
  // CREATE STATE //
  // //////////// //

  @override
  State<HomePage> createState() => _HomePageState();
}

/// State class for [HomePage].
class _HomePageState extends State<HomePage> {
  // MEMBER STATE //
  late final TextEditingController vehicleIDController;

  // //////////////////// //
  // INIT / DISPOSE STATE //
  // //////////////////// //

  @override
  void initState() {
    super.initState();

    // initialzing state
    vehicleIDController = TextEditingController();
  }

  @override
  void dispose() {
    super.dispose();

    // disposing of state
    vehicleIDController.dispose();
  }

  // //////////// //
  // BUILD METHOD //
  // //////////// //

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: PaddedCustomScrollView(
          slivers: [
            SliverFillRemaining(
              hasScrollBody: false,
              child: Center(
                child: SizedBox(
                  height: widget.containerHeight,
                  child: ColoredContainer(
                    color: MyColors.backgroundSecondary,
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        // ///// //
                        // TITLE //
                        // ///// //

                        const Text(
                          "TrainVis",
                          style: MyTextStyles.headerText2,
                        ),

                        // ////// //
                        // PROMPT //
                        // ////// //

                        const Text(
                          "Enter the ID of the vehicle",
                          style: MyTextStyles.bodyText1,
                        ),

                        // ////////// //
                        // TEXT FIELD //
                        // ////////// //

                        MyTextField.normal(
                          controller: vehicleIDController,
                          hintText: "e.g., 707-008",
                        ),

                        // ///////////// //
                        // SUBMIT BUTTON //
                        // ///////////// //

                        MyTextButton.primary(
                          text: "Submit",
                          onPressed: () {
                            // navigate to train profile page
                            context.pushNamed(
                              Routes.profile,
                              params: {"vehicleID": vehicleIDController.text},
                            );
                          },
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            )
          ],
        ),
      ),
    );
  }
}
