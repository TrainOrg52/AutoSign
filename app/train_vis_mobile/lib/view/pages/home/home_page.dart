import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'package:train_vis_mobile/view/theme/my_colors.dart';
import 'package:train_vis_mobile/view/theme/my_sizes.dart';
import 'package:train_vis_mobile/view/theme/my_text_button.dart';
import 'package:train_vis_mobile/view/theme/my_text_styles.dart';
import 'package:train_vis_mobile/view/widgets/colored_container.dart';

/// The home page of the application.
///
/// TODO
class HomePage extends StatefulWidget {
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
        child: Padding(
          padding: const EdgeInsets.symmetric(
            vertical: 0,
            horizontal: MySizes.paddingValue,
          ),
          child: Center(
            child: Row(
              children: [
                Expanded(
                  child: ColoredContainer(
                    color: MyColors.backgroundSecondary,
                    child: SizedBox(
                      height: 160,
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          // train vis header
                          const Text(
                            "TrainVis",
                            style: MyTextStyles.headerText2,
                          ),

                          // prompt
                          const Text(
                            "Enter the ID of the vehicle",
                            style: MyTextStyles.bodyText1,
                          ),

                          // text field
                          TextField(
                            controller: vehicleIDController,
                            decoration: const InputDecoration(
                              hintText: "e.g., 707-008",
                              isDense: true,
                              contentPadding: MySizes.padding,
                              border: OutlineInputBorder(
                                borderSide: BorderSide(
                                    color: MyColors.lineColor,
                                    width: MySizes.lineWidth),
                              ),
                              focusedBorder: OutlineInputBorder(
                                borderSide: BorderSide(
                                    color: MyColors.lineColor,
                                    width: MySizes.lineWidth),
                              ),
                            ),
                          ),

                          // submit button
                          MyTextButton.primary(
                            text: "Submit",
                            onPressed: () {
                              // navigate to train profile page
                              context.push("/${vehicleIDController.text}");
                            },
                          ),
                        ],
                      ),
                    ),
                  ),
                )
              ],
            ),
          ),
        ),
      ),
    );

    return Scaffold(
      body: Center(
        child: ColoredContainer(
          color: MyColors.backgroundSecondary,
          child: SizedBox(
            width: 380,
            height: 160,
            child: Padding(
              padding: MySizes.padding,
              child: Column(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  // train vis header
                  const Text(
                    "TrainVis",
                    style: MyTextStyles.headerText2,
                  ),

                  // prompt
                  const Text(
                    "Enter the ID of the vehicle",
                    style: MyTextStyles.bodyText1,
                  ),

                  // text field
                  TextField(
                    controller: vehicleIDController,
                    decoration: const InputDecoration(
                      hintText: "e.g., 707-008",
                      isDense: true,
                      contentPadding: MySizes.padding,
                      border: OutlineInputBorder(
                        borderSide: BorderSide(
                            color: MyColors.lineColor,
                            width: MySizes.lineWidth),
                      ),
                      focusedBorder: OutlineInputBorder(
                        borderSide: BorderSide(
                            color: MyColors.lineColor,
                            width: MySizes.lineWidth),
                      ),
                    ),
                  ),

                  // submit button
                  MyTextButton.primary(
                    text: "Submit",
                    onPressed: () {},
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}
