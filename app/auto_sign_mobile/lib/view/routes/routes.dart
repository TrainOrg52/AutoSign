import 'package:auto_sign_mobile/view/pages/home/home_page.dart';
import 'package:auto_sign_mobile/view/pages/inspect/inspect_page.dart';
import 'package:auto_sign_mobile/view/pages/inspections/image_view.dart';
import 'package:auto_sign_mobile/view/pages/inspections/inspections.dart';
import 'package:auto_sign_mobile/view/pages/inspections/summary.dart';
import 'package:auto_sign_mobile/view/pages/profile/profile_page.dart';
import 'package:auto_sign_mobile/view/pages/remediate/checkout/checkout_page.dart';
import 'package:auto_sign_mobile/view/pages/remediate/remediate_page.dart';
import 'package:auto_sign_mobile/view/pages/remediate/sign_remediate/sign_remediate_page.dart';
import 'package:auto_sign_mobile/view/pages/remediations/remediation_fix.dart';
import 'package:auto_sign_mobile/view/pages/remediations/remediation_summary.dart';
import 'package:auto_sign_mobile/view/pages/remediations/remediations.dart';
import 'package:auto_sign_mobile/view/pages/status/status_page.dart';
import 'package:go_router/go_router.dart';

/// Contains definitions for the application's routing information.
///
/// - Identifiers/names for each of the routes.
/// - The [GoRouter] object for the application.
class Routes {
  // ///////////////// //
  // ROUTE IDENTIFIERS //
  // ///////////////// //

  static const home = "home";
  static const profile = "profile";
  static const status = "status";
  static const inspect = "inspect";
  static const remediate = "remediate";
  static const signRemediate = "signRemediate";
  static const checkout = "checkout";
  static const inspections = "inspections";
  static const vehicleInspection = "inspectionWalkthrough";
  static const checkpointInspection = "inspectionCheckpoint";
  static const remediations = "remediations";
  static const vehicleRemediation = "vehicleRemediation";
  static const signRemdiation = "signRemdiation";

  // ///////////////// //
  // ROUTER DEFINITION //
  // ///////////////// //

  static final GoRouter router = GoRouter(
    initialLocation: "/",
    routes: [
      // //// //
      // HOME //
      // //// //

      GoRoute(
        name: Routes.home,
        path: "/",
        builder: (context, state) {
          return const HomePage();
        },
      ),

      // ///////////////// //
      // VEHICLE (PROFILE) //
      // ///////////////// //

      GoRoute(
        name: Routes.profile,
        path: "/:vehicleID",
        builder: (context, state) {
          // getting params from state
          String vehicleID = state.params["vehicleID"]!;

          // displaying profile page
          return ProfilePage(vehicleID: vehicleID);
        },
        routes: [
          // ////// //
          // STATUS //
          // ////// //

          GoRoute(
            name: Routes.status,
            path: "status",
            builder: (context, state) {
              // getting params from state
              String vehicleID = state.params["vehicleID"]!;

              // displaying status page
              return StatusPage(vehicleID: vehicleID);
            },
          ),

          // /////// //
          // INSPECT //
          // /////// //

          GoRoute(
            name: Routes.inspect,
            path: "inspect",
            builder: (context, state) {
              // getting params from state
              String vehicleID = state.params["vehicleID"]!;

              // displaying inspect page
              return InspectPage(vehicleID: vehicleID);
            },
          ),

          // ///////// //
          // REMEDIATE //
          // ///////// //

          GoRoute(
            name: Routes.remediate,
            path: "remediate",
            builder: (context, state) {
              // getting params from state
              String vehicleID = state.params["vehicleID"]!;

              // displaying remediate page
              return RemediatePage(vehicleID: vehicleID);
            },
            routes: [
              // //////// //
              // CHECKOUT //
              // //////// //

              GoRoute(
                name: Routes.checkout,
                path: "checkout",
                builder: (context, state) {
                  // getting params from state
                  String vehicleID = state.params["vehicleID"]!;

                  // displaying pagee
                  return CheckoutPage(vehicleID: vehicleID);
                },
              ),

              // ////////////// //
              // SIGN REMEDIATE //
              // ////////////// //

              GoRoute(
                name: Routes.signRemediate,
                path: ":checkpointID/:signID",
                builder: (context, state) {
                  // getting params from the state
                  String vehicleID = state.params["vehicleID"]!;
                  String checkpointID = state.params["checkpointID"]!;
                  String signID = state.params["signID"]!;

                  // displaying page
                  return SignRemediatePage(
                    vehicleID: vehicleID,
                    checkpointID: checkpointID,
                    signID: signID,
                  );
                },
              ),
            ],
          ),

          // /////////// //
          // INSPECTIONS //
          // /////////// //

          GoRoute(
            name: Routes.inspections,
            path: "inspections",
            builder: (context, state) {
              String vehicleID = state.params["vehicleID"]!;
              return InspectionsPage(vehicleID);
            },
            routes: [
              // ////////////////// //
              // VEHICLE INSPECTION //
              // ////////////////// //

              GoRoute(
                name: Routes.vehicleInspection,
                path: ":vehicleInspectionID",
                builder: (context, state) {
                  String vehicleID = state.params["vehicleID"]!;
                  String vehicleInspectionID =
                      state.params["vehicleInspectionID"]!;
                  return ReportSummary(
                    vehicleID: vehicleID,
                    vehicleInspectionID: vehicleInspectionID,
                  );
                },
                routes: [
                  // ///////////////////// //
                  // CHECKPOINT INSPECTION //
                  // ///////////////////// //

                  GoRoute(
                    name: Routes.checkpointInspection,
                    path: ":checkpointInspectionID/:checkpointID",
                    builder: (context, state) {
                      String vehicleID = state.params["vehicleID"]!;
                      String vehicleInspectionID =
                          state.params["vehicleInspectionID"]!;
                      String checkpointInspectionID =
                          state.params["checkpointInspectionID"]!;
                      String checkpointID = state.params["checkpointID"]!;
                      return ImageView(vehicleID, vehicleInspectionID,
                          checkpointInspectionID, checkpointID);
                    },
                  )
                ],
              ),
            ],
          ),

          // //////////// //
          // REMEDIATIONS //
          // //////////// //

          GoRoute(
            name: Routes.remediations,
            path: "remediations",
            builder: (context, state) {
              // getting params from state
              String vehicleID = state.params["vehicleID"]!;

              // displaying remediate page
              return RemediationsList(vehicleID);
            },
            routes: [
              // /////////////////////// //
              // REMEDIATION WALKTHROUGH //
              // /////////////////////// //

              GoRoute(
                name: Routes.vehicleRemediation,
                path: ":vehicleRemediationID",
                builder: (context, state) {
                  // getting params from state
                  String vehicleRemediationID =
                      state.params["vehicleRemediationID"]!;
                  String vehicleID = state.params["vehicleID"]!;

                  // displaying remediate page
                  return RemediationSummary(vehicleID, vehicleRemediationID);
                },

                // ////////////////////// //
                // REMEDIATION CHECKPOINT //
                // ////////////////////// //

                routes: [
                  GoRoute(
                    name: Routes.signRemdiation,
                    path: ":signRemediationID",
                    builder: (context, state) {
                      // getting params from state
                      String vehicleID = state.params["vehicleID"]!;
                      String vehicleRemediationID =
                          state.params["vehicleRemediationID"]!;
                      String signRemdiationID =
                          state.params["signRemediationID"]!;

                      // displaying remediate page
                      return RemediationFix(
                          vehicleID, vehicleRemediationID, signRemdiationID);
                    },
                  )
                ],
              ),
            ],
          ),
        ],
      ),
    ],
  );
}
