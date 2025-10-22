/*   =====================================================================

Copyright (c) 2025 by Omnesys Technologies, Inc.  All rights reserved.

Warning :
        This Software Product is protected by copyright law and international
        treaties.  Unauthorized use, reproduction or distribution of this
        Software Product (including its documentation), or any portion of it,
        may result in severe civil and criminal penalties, and will be
        prosecuted to the maximum extent possible under the law.

        Omnesys Technologies, Inc. will compensate individuals providing
        admissible evidence of any unauthorized use, reproduction, distribution
        or redistribution of this Software Product by any person, company or 
        organization.

This Software Product is licensed strictly in accordance with a separate
Software System License Agreement, granted by Omnesys Technologies, Inc., which
contains restrictions on use, reverse engineering, disclosure, confidentiality 
and other matters.

     =====================================================================   */

using System;
using System.Collections.Generic;
using System.Text;
using com.omnesys.omne.om;
using com.omnesys.rapi;

/*   =====================================================================   */

namespace SampleMDNamespace
     {
     /*   ================================================================   */

     class MyAdmCallbacks : AdmCallbacks
          {
          public override void Alert(AlertInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);
               }
          }

     /*   ================================================================   */

     class MyCallbacks : RCallbacks
          {
	  public enum LoginStatus
	       {
	       NotLoggedIn,
	       LoginFailed,
	       LoggedIn
	       };

          public LoginStatus IhLoginStatus
               {
               get { return PRI_eIhLoginStatus; }
	       set { PRI_eIhLoginStatus = value; }
               }

          public LoginStatus MdLoginStatus
               {
               get { return PRI_eMdLoginStatus; }
	       set { PRI_eMdLoginStatus = value; }
               }

          public bool BarsAreDone
               {
               get { return PRI_bBarsAreDone; }
	       set { PRI_bBarsAreDone = value; }
               }

          /*   -----------------------------------------------------------   */

          public MyCallbacks()
               {
	       PRI_eIhLoginStatus = LoginStatus.NotLoggedIn;
	       PRI_eMdLoginStatus = LoginStatus.NotLoggedIn;
	       PRI_bBarsAreDone   = false;
               }

          /*   -----------------------------------------------------------   */

          public override void Alert(AlertInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);

               if (oInfo.ConnectionId == ConnectionId.MarketData)
		    {
		    if (oInfo.AlertType == AlertType.LoginComplete)
			 {
			 PRI_eMdLoginStatus = LoginStatus.LoggedIn;
			 }
		    else if (oInfo.AlertType == AlertType.LoginFailed)
			 {
			 PRI_eMdLoginStatus = LoginStatus.LoginFailed;
			 }
		    }

               if (oInfo.ConnectionId == ConnectionId.History)
		    {
		    if (oInfo.AlertType == AlertType.LoginComplete)
			 {
			 PRI_eIhLoginStatus = LoginStatus.LoggedIn;
			 }
		    else if (oInfo.AlertType == AlertType.LoginFailed)
			 {
			 PRI_eIhLoginStatus = LoginStatus.LoginFailed;
			 }
		    }
               }

          /*   -----------------------------------------------------------   */

	  public override void  Bar(BarInfo oInfo)
	       {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);
               Console.Out.WriteLine();
	       }

          /*   -----------------------------------------------------------   */

	  public override void  BarReplay(BarReplayInfo oInfo)
	       {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);
               Console.Out.WriteLine();

	       PRI_bBarsAreDone = true;
	       }

          /*   -----------------------------------------------------------   */

	  private LoginStatus PRI_eIhLoginStatus;
	  private LoginStatus PRI_eMdLoginStatus;
          private bool        PRI_bBarsAreDone;
          }

     /*   ================================================================   */

class Program
     {
     static void Main(string[] args)
          {
          string USAGE = "SampleBar user password exchange symbol " +
	       "bar_type[r|t|v|s|m|d|w] type_specifier"; 

          if (args.Length < 6)
               {
               System.Console.Out.WriteLine(USAGE);
               return;
               }

          /*   -----------------------------------------------------------   */
          /*   Remember to add the parameters in Visual Studio if debugging  */

          string sUser          = args[0];
          string sPassword      = args[1];
          string sExchange      = args[2];            /* CME, for example    */
          string sSymbol        = args[3];            /* ESU0, for example   */
          string sType          = args[4];
          string sTypeSpecifier = args[5];

          /*   -----------------------------------------------------------   */

          MyCallbacks   oCallbacks = new MyCallbacks();
          REngineParams oParams    = new REngineParams();
          REngine       oEngine    = null;

          /*   -----------------------------------------------------------   */

          oParams.AppName      = "SampleBar.NET";
          oParams.AppVersion   = "1.0.0.0";
          oParams.AdmCallbacks = new MyAdmCallbacks();
          oParams.DmnSrvrAddr  = "rituz00100.00.rithmic.com:65000~rituz00100.00.rithmic.net:65000~rituz00100.00.theomne.net:65000~rituz00100.00.theomne.com:65000";
          oParams.DomainName   = "rithmic_uat_dmz_domain";
          oParams.LicSrvrAddr  = "rituz00100.00.rithmic.com:56000~rituz00100.00.rithmic.net:56000~rituz00100.00.theomne.net:56000~rituz00100.00.theomne.com:56000";
          oParams.LocBrokAddr  = "rituz00100.00.rithmic.com:64100";
          oParams.LoggerAddr   = "rituz00100.00.rithmic.com:45454~rituz00100.00.rithmic.net:45454~rituz00100.00.theomne.com:45454~rituz00100.00.theomne.net:45454";
          oParams.LogFilePath  = "sb.log";

          /*   -----------------------------------------------------------   */

          try
               {
               /*   ------------------------------------------------------   */
               /*   Instantiate the REngine.                                 */
               /*   ------------------------------------------------------   */

               oEngine = new REngine(oParams);

               /*   ------------------------------------------------------   */
               /*   Initiate the login to the trading platform.              */
               /*   ------------------------------------------------------   */

               oEngine.login(oCallbacks, 
                    string.Empty,  // empty is same as default env key
                    sUser, 
                    sPassword,
                    "login_agent_tpc", 
                    Constants.DEFAULT_ENVIRONMENT_KEY,
                    string.Empty,            // sTsUser
                    string.Empty,            // sTsPassword
                    string.Empty,            // sTsCnnctPt
                    string.Empty,            // sPnlCnnctPt
		    string.Empty,            // sIhEnvKey
		    sUser,                   // sIhUser
		    sPassword,               // sIhPassword
                    "login_agent_historyc"); // sIhCnnctPt

               /*   ------------------------------------------------------   */
               /*   After calling REngine::login, RCallbacks::Alert will be  */
               /*   called a number of times.  Wait for when the login to    */
               /*   the MdCnnctPt and IhCnnctPt completes or fails.  (See    */
               /*   MyCallbacks::Alert() for details).                       */
               /*   ------------------------------------------------------   */

               while (oCallbacks.MdLoginStatus != MyCallbacks.LoginStatus.LoggedIn &&
		      oCallbacks.MdLoginStatus != MyCallbacks.LoginStatus.LoginFailed)
                    {
                    System.Threading.Thread.Sleep(1000);
                    }

	       if (oCallbacks.MdLoginStatus == MyCallbacks.LoginStatus.LoginFailed)
		    {
		    System.Console.Out.WriteLine("Please make sure you entered the username and password "
			 + "correctly.  Also, please make sure your credentials match "
			 + "the system you are trying to log in to.");

		    oEngine.shutdown();
		    return;
		    }

               while (oCallbacks.IhLoginStatus != MyCallbacks.LoginStatus.LoggedIn &&
		      oCallbacks.IhLoginStatus != MyCallbacks.LoginStatus.LoginFailed)
                    {
                    System.Threading.Thread.Sleep(1000);
                    }

	       if (oCallbacks.IhLoginStatus == MyCallbacks.LoginStatus.LoginFailed)
		    {
		    System.Console.Out.WriteLine("Please make sure you entered the username and password "
			 + "correctly.  Also, please make sure your credentials match "
			 + "the system you are trying to log in to.");

		    oEngine.shutdown();
		    return;
		    }

               /*   ------------------------------------------------------   */
               /*   Request bars by filling in ReplayBarParams.              */
               /*   ------------------------------------------------------   */

	       ReplayBarParams oRbParams = new ReplayBarParams();

	       oRbParams.Exchange = sExchange;
	       oRbParams.Symbol   = sSymbol;

               /*   ------------------------------------------------------   */
	       //   bar type and type specifier

	       if (sType == "r" || sType == "R")
		    {
		    oRbParams.Type = BarType.Range;
		    oRbParams.SpecifiedRange = Convert.ToDouble(sTypeSpecifier, 
			 System.Globalization.CultureInfo.InvariantCulture);
		    }
	       else if (sType == "t" || sType == "T")
		    {
		    oRbParams.Type = BarType.Tick;
		    oRbParams.SpecifiedTicks = Convert.ToInt32(sTypeSpecifier);
		    }
	       else if (sType == "v" || sType == "V")
		    {
		    oRbParams.Type            = BarType.Volume;
		    oRbParams.SpecifiedVolume = Convert.ToInt32(sTypeSpecifier);
		    }
	       else if (sType == "s" || sType == "S")
		    {
		    oRbParams.Type             = BarType.Second;
		    oRbParams.SpecifiedSeconds = Convert.ToInt32(sTypeSpecifier);
		    }
	       else if (sType == "m" || sType == "M")
		    {
		    oRbParams.Type             = BarType.Minute;
		    oRbParams.SpecifiedMinutes = Convert.ToInt32(sTypeSpecifier);
		    }
	       else if (sType == "d" || sType == "D")
		    {
		    // daily bars do not support <N> day bars
		    oRbParams.Type = BarType.Daily;
		    }
	       else if (sType == "w" || sType == "W")
		    {
		    // weekly bars do not support <N> week bars
		    oRbParams.Type = BarType.Weekly;
		    }
	       else
		    {
		    System.Console.Out.WriteLine(USAGE);
		    return;
		    }

	       /*   ------------------------------------------------------   */

	       //   oRbParams.CustomSession : indicates whether the actual trading
	       //   session or a custom session should be used to calculate bars.  A
	       //   custom session is specified in seconds-since-midnight (ssm).
	       //   For more information, see documentation on :
	       //      ReplayBarParams::CustomSession
	       //      ReplayBarParams::OpenSsm
	       //      ReplayBarParams::CloseSsm
	       
	       /*   ------------------------------------------------------   */

	       //   oRbParams.Profile : set to true if you want to retrieve volume
	       //   profile metrics with minute bars.

	       /*   ------------------------------------------------------   */

	       //   This section specifies the time period over which bars are being
	       //   requested.  Daily and weekly bars use date strings in the form 
	       //   CCYYMMDD to specify the date range over which bars are being 
	       //   requested.  (See ReplayBarParams::Start/EndDate).  Other bar types
	       //   use the unix time seconds-since-the-beginning-of-epoch 
	       //   format.  (See https://en.wikipedia.org/wiki/Unix_time for more info.)

	       if (oRbParams.Type == BarType.Daily||
		   oRbParams.Type == BarType.Weekly)
		    {
		    // retreive one year of daily or weekly bars
		    DateTime dtNow        = DateTime.Now; // current local time
		    oRbParams.EndCcyymmdd = dtNow.ToString("yyyyMMdd");

		    DateTime dtOneYearAgo   = dtNow.AddYears(-1);
		    oRbParams.StartCcyymmdd = dtOneYearAgo.ToString("yyyyMMdd");
		    }
	       else
		    {
		    // For the purposes of this sample, use the most recent 24 hours, 
		    // in whole seconds.

		    // Get current system time in unix/ssboe as end time
		    DateTime dtUtcNow = DateTime.UtcNow;
		    DateTime dtEpoch = new DateTime(1970,1,1);
		    TimeSpan tsUnixNow = dtUtcNow.Subtract(dtEpoch);
		    double dSsboeNow = tsUnixNow.TotalSeconds;

		    oRbParams.EndSsboe = (int)dSsboeNow;
		    oRbParams.EndUsecs = 0;

		    // subtract 24 hrs * 60 min * 60 sec for start time
		    oRbParams.StartSsboe = oRbParams.EndSsboe - (24 * 60 * 60);
		    oRbParams.StartUsecs = 0;
		    }

	       /*   ------------------------------------------------------   */
	       
	       oRbParams.Context = oRbParams;
	       
	       /*   ------------------------------------------------------   */
	       
	       oEngine.replayBars(oRbParams);

               /*   ------------------------------------------------------   */
               /*   At this point the bar callback routine will start firing */
               /*   on a different thread.  When the bars are done, the bar  */
	       /*   replay callback will fire, signalling the end of this    */
	       /*   request.  This main thread will wait until then before   */
               /*   continuing.                                              */
               /*   ------------------------------------------------------   */

               while (!oCallbacks.BarsAreDone)
                    {
                    System.Threading.Thread.Sleep(1000);
                    }

               /*   ------------------------------------------------------   */
               /*   We are done, so log out...                               */
               /*   ------------------------------------------------------   */

               oEngine.logout();

               /*   ------------------------------------------------------   */
               /*   and shutdown the REngine instance.                       */
               /*   ------------------------------------------------------   */

               oEngine.shutdown();
               }
          catch (OMException oEx)
               {
               System.Console.Out.WriteLine("error : {0}", oEx.Message);
               }
          catch (Exception e)
               {
               System.Console.Out.WriteLine("exception : {0}", e.Message);
               }

          /*   -----------------------------------------------------------   */

          return;
          }
     }
}
