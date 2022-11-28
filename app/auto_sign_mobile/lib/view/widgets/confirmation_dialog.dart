import 'package:auto_sign_mobile/view/theme/data/my_text_styles.dart';
import 'package:auto_sign_mobile/view/theme/widgets/my_text_button.dart';
import 'package:flutter/material.dart';

/// TODO
class ConfirmationDialog extends StatelessWidget {
  // MEMBER VARIABLES //
  final String title; // title for the dialog
  final String message; // message for the dialog
  final String falseText; // text for false option
  final String trueText; // text for true option
  final Color trueBackgroundColor; // color for true option
  final Color trueTextColor; // color for text of true option

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  const ConfirmationDialog({
    super.key,
    required this.title,
    required this.message,
    required this.falseText,
    required this.trueText,
    required this.trueBackgroundColor,
    required this.trueTextColor,
  });

  // //////////// //
  // BUILD METHOD //
  // //////////// //

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      // ///////////// //
      // CONFIGURATION //
      // ////////////// /

      actionsAlignment: MainAxisAlignment.center,

      // ///// //
      // TITLE //
      // ///// //

      title: _buildTitle(),

      // /////// //
      // CONTENT //
      // /////// //

      content: _buildContent(),

      // /////// //
      // ACTIONS //
      // /////// //

      actions: _buildActions(context),
    );
  }

  // ////////////////////// //
  // HELPER BUILDER METHODS //
  // ////////////////////// //

  Widget _buildTitle() {
    return Text(
      title,
      style: MyTextStyles.headerText2,
      textAlign: TextAlign.center,
    );
  }

  Widget _buildContent() {
    return Text(
      message,
      style: MyTextStyles.bodyText1,
      textAlign: TextAlign.center,
    );
  }

  List<Widget> _buildActions(BuildContext context) {
    return [
      // ///// //
      // FALSE //
      // ///// //

      MyTextButton.secondary(
        text: falseText,
        onPressed: () {
          // popping the dialog with false
          Navigator.of(context).pop(false);
        },
      ),

      // //// //
      // TRUE //
      // //// //

      MyTextButton.custom(
        text: trueText,
        backgroundColor: trueBackgroundColor,
        borderColor: trueBackgroundColor,
        textColor: trueTextColor,
        onPressed: () {
          // popping the dialog with true
          Navigator.of(context).pop(true);
        },
      ),
    ];
  }
}
