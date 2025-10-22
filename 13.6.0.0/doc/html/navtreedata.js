/*
 @licstart  The following is the entire license notice for the JavaScript code in this file.

 The MIT License (MIT)

 Copyright (C) 1997-2020 by Dimitri van Heesch

 Permission is hereby granted, free of charge, to any person obtaining a copy of this software
 and associated documentation files (the "Software"), to deal in the Software without restriction,
 including without limitation the rights to use, copy, modify, merge, publish, distribute,
 sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all copies or
 substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

 @licend  The above is the entire license notice for the JavaScript code in this file
*/
var NAVTREE =
[
  [ "R | API+.NET", "index.html", [
    [ "Overview", "index.html", [
      [ "Copyright and License", "index.html#legalese", null ],
      [ "Introduction", "index.html#overview_intro", null ]
    ] ],
    [ "Quick Start", "quick_start.html", [
      [ "Table Of Contents", "quick_start.html#qs_toc", null ],
      [ "Rithmic Systems, Agreements, and Conformance", "quick_start.html#qs_systems_agreements_conformance", null ],
      [ "Verify Connectivity, Credentials, And Configuration Using R | Trader", "quick_start.html#qs_verify_with_rtrader", [
        [ "Download And Install R | Trader", "quick_start.html#qs_download_rtrader", null ],
        [ "Log In Using R | Trader", "quick_start.html#qs_login_rtrader", null ],
        [ "Unsigned Agreements", "quick_start.html#qs_rtrader_agreements", null ],
        [ "Market Data Permissions", "quick_start.html#qs_rtrader_market_data", null ],
        [ "Risk Management Settings", "quick_start.html#qs_rtrader_order", null ]
      ] ],
      [ "Download And Install", "quick_start.html#qs_download_rapi", null ],
      [ "Configure Sample Code", "quick_start.html#qs_configure_sample", [
        [ "Configure For A Rithmic Installation", "quick_start.html#qs_configure_sample_installation", null ]
      ] ],
      [ "Build Sample Code", "quick_start.html#qs_build_sample", null ],
      [ "Running Sample Applications", "quick_start.html#qs_running_sample", null ],
      [ "Verify Market Data Functionality With SampleMD", "quick_start.html#qs_verify_sample_md", null ],
      [ "Verify Order Functionality With SampleOrder", "quick_start.html#qs_verify_sample_order", null ],
      [ "Next Steps", "quick_start.html#qs_next_steps", null ]
    ] ],
    [ "Programmers' Guide", "programmers_guide.html", [
      [ "Document Information", "programmers_guide.html#pg_doc_info", null ],
      [ "Table Of Contents", "programmers_guide.html#pg_toc", null ],
      [ "Preface", "programmers_guide.html#pg_preface", null ],
      [ "What is R | Trade Execution Platform?", "programmers_guide.html#pg_what_is_rtrade", null ],
      [ "What is R | API+?", "programmers_guide.html#pg_what_is_rapi", [
        [ "C++ vs. .NET", "programmers_guide.html#pg_cpp_vs_dotnet", null ],
        [ "R | API+ and R | Diamond Cutter", "programmers_guide.html#pg_which_api_for_me", null ]
      ] ],
      [ "Operating Systems, Frameworks, Compilers and Target Platforms", "programmers_guide.html#pg_os_platforms", [
        [ "The C++ compilation of Rithmic APIs have been built as follows:", "programmers_guide.html#pg_cpp_compilation", null ],
        [ "The .NET compilation of Rithmic APIs have been built with a target frameworks of .NET 4.72 and 3.5 as follows:", "programmers_guide.html#pg_dotnet_compilation", null ]
      ] ],
      [ "Installation and File List", "programmers_guide.html#pg_file_list", [
        [ "Files in the C++ Compilation", "programmers_guide.html#pg_cpp_files", null ],
        [ "Files in the .NET Compilation", "programmers_guide.html#pg_dotnet_files", null ]
      ] ],
      [ "Building A C++ Rithmic API Application", "programmers_guide.html#pg_building_cpp", [
        [ "Building C++ On Windows Using Visual Studio", "programmers_guide.html#pg_building_cpp_windows", [
          [ "Compiling On Windows", "programmers_guide.html#pg_compiling_windows", null ],
          [ "Linking On Windows", "programmers_guide.html#pg_linking_windows", null ]
        ] ],
        [ "Building C++ On Linux Using GCC/g++", "programmers_guide.html#pg_building_cpp_linux", [
          [ "Compiling On Linux", "programmers_guide.html#pg_compiling_linux", null ],
          [ "Linking On Linux", "programmers_guide.html#pg_linking_linux", null ]
        ] ],
        [ "Building C++ On Darwin Using GCC/g++", "programmers_guide.html#pg_building_cpp_darwin", [
          [ "Compiling On Darwin", "programmers_guide.html#pg_compiling_darwin", null ],
          [ "Linking On Darwin", "programmers_guide.html#pg_linking_darwin", null ]
        ] ]
      ] ],
      [ "Building A .NET Rithmic API Application", "programmers_guide.html#pg_building_dotnet", null ],
      [ "Organization Of Rithmic API Classes", "programmers_guide.html#pg_classes", [
        [ "There are three main categories of classes in the Rithmic APIs :", "programmers_guide.html#pg_class_categories", null ],
        [ "The dump() Method", "programmers_guide.html#pg_class_dump_method", null ]
      ] ],
      [ "Configuring A Rithmic API Application", "programmers_guide.html#pg_configuring", [
        [ "Configuration Files", "programmers_guide.html#pg_configuring_configuration_files", null ],
        [ "Conformance and Fees", "programmers_guide.html#pg_configuring_conformance", null ]
      ] ],
      [ "Max Session Counts", "programmers_guide.html#pg_max_session_count", null ],
      [ "Using R | Trader Pro as a Plug-In Host", "programmers_guide.html#pg_plugins", null ],
      [ "Connecting to Multiple Rithmic Systems Simultaneously Using Environments", "programmers_guide.html#pg_environments", [
        [ "Background", "programmers_guide.html#pg_environments_background", null ],
        [ "Multiple Environments", "programmers_guide.html#pg_environments_multiple_environments", null ],
        [ "Prior Behavior With A Single Environment", "programmers_guide.html#pg_environments_single_environment", null ],
        [ "Comments", "programmers_guide.html#pg_environments_comments", null ],
        [ "Programming Interface", "programmers_guide.html#pg_environments_interface", null ]
      ] ],
      [ "Connections", "programmers_guide.html#pg_connections", [
        [ "Connect Points", "programmers_guide.html#pg_cnnct_pts", null ],
        [ "Aggregated Data Market Data", "programmers_guide.html#pg_connection_aggregated_data", null ],
        [ "Gateways and Points-Of-Presence", "programmers_guide.html#pg_connection_locations", null ],
        [ "Best Practices", "programmers_guide.html#pg_connection_best_practices", null ],
        [ "Connection Management", "programmers_guide.html#pg_connection_mgmt", null ]
      ] ],
      [ "Establishing And Maintaining State With The Snapshot-Update Pattern", "programmers_guide.html#pg_updates_and_snapshots", [
        [ "Establishing And Maintaining State", "programmers_guide.html#pg_updates_and_snapshots_establishing_state", null ],
        [ "Polling Snapshots", "programmers_guide.html#pg_updates_and_snapshots_polling_snapshots", null ],
        [ "Poll Once, Then Updates", "programmers_guide.html#pg_updates_and_snapshots_poll_once", null ],
        [ "An Example With Orders", "programmers_guide.html#pg_updates_and_snapshots_example", null ],
        [ "Some Subtleties To Be Aware Of", "programmers_guide.html#pg_updates_and_snapshots_subtleties", null ]
      ] ],
      [ "Strings And tsNCharcb (C++ Only)", "programmers_guide.html#pg_strings_cpp", [
        [ "Why use the tsNCharcb for representing strings?", "programmers_guide.html#pg_strings_why_tsncharcb", null ],
        [ "How does one convert a C++ string into a tsNCharcb?", "programmers_guide.html#pg_strings_cpp_to_tsnchar", null ],
        [ "How does one convert a tsNCharcb into a C++ string?", "programmers_guide.html#pg_strings_tsnchar_to_cpp", null ],
        [ "Do I need to initialize both the pData and the iDataLen of each tsNCharcb?", "programmers_guide.html#pg_strings_tsnchar_init", null ],
        [ "Who owns the memory pointed to by tsNCharcb::pData?", "programmers_guide.html#pg_strings_memory", null ]
      ] ],
      [ "On Timestamps (ssboe, usecs and nsecs)", "programmers_guide.html#pg_timestamps", null ],
      [ "On Threads", "programmers_guide.html#pg_threads", [
        [ "Threads In C++ API", "programmers_guide.html#pg_threads_cpp", null ],
        [ "Threads In .NET API", "programmers_guide.html#pg_threads_dotnet", null ]
      ] ],
      [ "Error Handling Conventions", "programmers_guide.html#pg_errors", [
        [ "Error Codes", "programmers_guide.html#pg_error_codes", null ],
        [ "Exception Classes", "programmers_guide.html#pg_exception_classes", null ],
        [ "The 'aiCode' Out Parameter (C++ only)", "programmers_guide.html#pg_aicode_out_param", null ],
        [ "Why And Where Errors Might Occur", "programmers_guide.html#pg_error_why", null ],
        [ "Examples Of Errors", "programmers_guide.html#pg_error_examples", null ],
        [ "Error Handling In User Implemented Callbacks", "programmers_guide.html#pg_errors_in_user_callbacks", null ]
      ] ],
      [ "Basic Steps of a Program that Incorporates R | API", "programmers_guide.html#pg_basic_steps", null ],
      [ "Feedback And Bug Reports", "programmers_guide.html#pg_bug_reports", null ],
      [ "Contact Info", "programmers_guide.html#pg_contact_info", null ]
    ] ],
    [ "F.A.Q.", "faq_root.html", "faq_root" ],
    [ "Namespaces", "namespaces.html", [
      [ "Namespace List", "namespaces.html", "namespaces_dup" ],
      [ "Namespace Members", "namespacemembers.html", [
        [ "All", "namespacemembers.html", null ],
        [ "Enumerations", "namespacemembers_enum.html", null ]
      ] ]
    ] ],
    [ "Classes", "annotated.html", [
      [ "Class List", "annotated.html", "annotated_dup" ],
      [ "Class Index", "classes.html", null ],
      [ "Class Hierarchy", "hierarchy.html", "hierarchy" ],
      [ "Class Members", "functions.html", [
        [ "All", "functions.html", "functions_dup" ],
        [ "Functions", "functions_func.html", "functions_func" ],
        [ "Variables", "functions_vars.html", null ],
        [ "Properties", "functions_prop.html", "functions_prop" ]
      ] ]
    ] ]
  ] ]
];

var NAVTREEINDEX =
[
"annotated.html",
"classcom_1_1omnesys_1_1rapi_1_1BarInfo.html#a59720dd1e2167815ddc715eda86b3eef",
"classcom_1_1omnesys_1_1rapi_1_1BracketParams.html#ac6da4847ffcdac57e7535cf42bf1f6cc",
"classcom_1_1omnesys_1_1rapi_1_1EquityOptionStrategyListInfo.html",
"classcom_1_1omnesys_1_1rapi_1_1LineInfo.html#a7771028ebb805556682b8745a3eabc37",
"classcom_1_1omnesys_1_1rapi_1_1ModifyLimitOrderParams.html#aca7cab88f4bd1723a1439cef22aec340",
"classcom_1_1omnesys_1_1rapi_1_1OrderFillReport.html",
"classcom_1_1omnesys_1_1rapi_1_1OrderReport.html#a7d1d825f8f4e25cfe7a8629f214015de",
"classcom_1_1omnesys_1_1rapi_1_1ProductRmsInfo.html#a71960325432634328118236e9276b215",
"classcom_1_1omnesys_1_1rapi_1_1RCallbacks.html#a3bf363ff74aea46b560a6570952181b3",
"classcom_1_1omnesys_1_1rapi_1_1REngine.html#a82dd000ee8390a530d0386d7dbc27116",
"classcom_1_1omnesys_1_1rapi_1_1RmsInfo.html#af78375d21ad6a950012daf0cf116ed7c",
"classcom_1_1omnesys_1_1rapi_1_1TradeRouteListInfo.html#a706b0d6c11388462b93856aae81a0360",
"faq_std.html#faq_rtrade_what_is_dbo",
"programmers_guide.html#pg_threads_dotnet"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';