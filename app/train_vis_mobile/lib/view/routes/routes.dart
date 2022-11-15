import 'package:go_router/go_router.dart';
import 'package:train_vis_mobile/view/pages/home/home_page.dart';
import 'package:train_vis_mobile/view/pages/inspect/inspect_page.dart';
import 'package:train_vis_mobile/view/pages/profile/profile_page.dart';
import 'package:train_vis_mobile/view/pages/remediate/remediate_page.dart';
import 'package:train_vis_mobile/view/pages/remediations/remediation_fix.dart';
import 'package:train_vis_mobile/view/pages/remediations/remediation_summary.dart';
import 'package:train_vis_mobile/view/pages/remediations/remediations.dart';
import 'package:train_vis_mobile/view/pages/reports/image_view.dart';
import 'package:train_vis_mobile/view/pages/reports/reports.dart';
import 'package:train_vis_mobile/view/pages/reports/summary.dart';
import 'package:train_vis_mobile/view/pages/status/status_page.dart';

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
  static const reports = "reports";
  static const vehicleInspection = "inspectionWalkthrough";
  static const checkpointInspection = "inspectionCheckpoint";
  static const remediations = "remediations";
  static const remediationWalkthrough = "remediationWalkthrough";
  static const remediationCheckpoint = "remediationCheckpoint";

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
          ),

          // /////// //
          // REPORTS //
          // /////// //

          GoRoute(
            name: Routes.reports,
            path: "reports",
            builder: (context, state) {
              String vehicleID = state.params["vehicleID"]!;
              return ReportsPage(vehicleID);
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
              return RemediationsList();
            },
            routes: [
              // /////////////////////// //
              // REMEDIATION WALKTHROUGH //
              // /////////////////////// //

              GoRoute(
                name: Routes.remediationWalkthrough,
                path: ":remediationWalkthroughID",
                builder: (context, state) {
                  // getting params from state
                  String remediationWalkthroughID =
                      state.params["remediationWalkthroughID"]!;

                  // displaying remediate page
                  return RemediationSummary();
                },

                // ////////////////////// //
                // REMEDIATION CHECKPOINT //
                // ////////////////////// //

                routes: [
                  GoRoute(
                    name: Routes.remediationCheckpoint,
                    path: ":remediationCheckpointID",
                    builder: (context, state) {
                      // getting params from state
                      String remediationCheckpointID =
                          state.params["remediationCheckpointID"]!;

                      // displaying remediate page
                      return RemediationFix();
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
