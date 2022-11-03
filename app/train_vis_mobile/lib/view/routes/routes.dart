// router
import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'package:train_vis_mobile/view/pages/home/home_page.dart';
import 'package:train_vis_mobile/view/pages/inspect/inspect_page.dart';
import 'package:train_vis_mobile/view/pages/profile/profile_page.dart';
import 'package:train_vis_mobile/view/pages/remediate/remediate_page.dart';
import 'package:train_vis_mobile/view/pages/remediations/remediation_checkpoint_page.dart';
import 'package:train_vis_mobile/view/pages/remediations/remediation_walkthrough_page.dart';
import 'package:train_vis_mobile/view/pages/remediations/remediations_page.dart';
import 'package:train_vis_mobile/view/pages/status/status_page.dart';

// ///////////////// //
// ROUTE INFORMATION //
// ///////////////// //

const home = "home";
const profile = "profile";
const status = "status";
const inspect = "inspect";
const remediate = "remediate";
const reports = "reports";
const inspectionWalkthrough = "inspectionWalkthrough";
const inspectionCheckpoint = "inspectionCheckpoint";
const remediations = "remediations";
const remediationWalkthrough = "remediationWalkthrough";
const remediationCheckpoint = "remediationCheckpoint";

// ///////////////// //
// ROUTER DEFINITION //
// ///////////////// //

// go router object
final GoRouter router = GoRouter(
  initialLocation: "/",
  routes: [
    // //// //
    // HOME //
    // //// //

    GoRoute(
      name: home,
      path: "/",
      builder: (context, state) {
        return const HomePage();
      },
    ),

    // ///////////////// //
    // VEHICLE (PROFILE) //
    // ///////////////// //

    GoRoute(
      name: profile,
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
          name: status,
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
          name: inspect,
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
          name: remediate,
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
          name: reports,
          path: "reports",
          builder: (context, state) {
            // TODO displaying filler page
            return Center(
              child: Text("Reports for ${state.params["vehicleID"]}."),
            );
          },
          routes: [
            // ////////////////////// //
            // INSPECTION WALKTHROUGH //
            // ////////////////////// //

            GoRoute(
              name: inspectionWalkthrough,
              path: ":inspectionWalkthroughID",
              builder: (context, state) {
                // TODO displaying filler page
                return Center(
                  child: Text(
                    "Inspection: ${state.params["inspectionWalkthroughID"]}.",
                  ),
                );
              },
              routes: [
                // ///////////////////// //
                // INSPECTION CHECKPOINT //
                // ///////////////////// //

                GoRoute(
                  name: inspectionCheckpoint,
                  path: ":inspectionCheckpointID",
                  builder: (context, state) {
                    // TODO displaying filler page
                    return Center(
                      child: Text(
                        "Inspection checkpoint: ${state.params["inspectionCheckpointID"]}.",
                      ),
                    );
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
          name: remediations,
          path: "remediations",
          builder: (context, state) {
            // getting params from state
            String vehicleID = state.params["vehicleID"]!;

            // displaying remediate page
            return RemediationsPage(vehicleID: vehicleID);
          },
          routes: [
            // /////////////////////// //
            // REMEDIATION WALKTHROUGH //
            // /////////////////////// //

            GoRoute(
              name: remediationWalkthrough,
              path: ":remediationWalkthroughID",
              builder: (context, state) {
                // getting params from state
                String remediationWalkthroughID =
                    state.params["remediationWalkthroughID"]!;

                // displaying remediate page
                return RemediationWalkthroughPage(
                    remediationWalkthroughID: remediationWalkthroughID);
              },

              // ////////////////////// //
              // REMEDIATION CHECKPOINT //
              // ////////////////////// //

              routes: [
                GoRoute(
                  name: remediationCheckpoint,
                  path: ":remediationCheckpointID",
                  builder: (context, state) {
                    // getting params from state
                    String remediationCheckpointID =
                        state.params["remediationCheckpointID"]!;

                    // displaying remediate page
                    return RemediationCheckpointPage(
                        remediationCheckpointID: remediationCheckpointID);
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
