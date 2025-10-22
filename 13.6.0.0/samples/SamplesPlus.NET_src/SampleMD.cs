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
	       LoginInProgress,
	       LoginFailed,
	       LoggedIn
	       };

          public LoginStatus RepositoryLoginStatus
               {
               get { return PRI_eRepositoryLoginStatus; }
	       set { PRI_eRepositoryLoginStatus = value; }
               }

	  public bool ReceivedAgreementList
               {
               get { return PRI_bReceivedAgreementList; }
               }

	  public int UnacceptedMandatoryAgreementCount
               {
               get { return PRI_iUnacceptedMandatoryAgreementCount; }
               }

          public bool LoggedIntoMd
               {
               get { return PRI_bLoggedIntoMd; }
               }

          public bool GotPriceIncrInfo
               {
               get { return PRI_bGotPriceIncrInfo; }
               }

          /*   -----------------------------------------------------------   */

          public MyCallbacks()
               {
	       PRI_eRepositoryLoginStatus = LoginStatus.NotLoggedIn;
	       PRI_bReceivedAgreementList = false;
	       PRI_iUnacceptedMandatoryAgreementCount = 0;

               PRI_bLoggedIntoMd = false;
               PRI_bGotPriceIncrInfo = false;
               }

          /*   -----------------------------------------------------------   */

          public override void Alert(AlertInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);

               if (oInfo.ConnectionId == ConnectionId.Repository)
		    {
		    if (oInfo.AlertType == AlertType.LoginComplete)
			 {
			 PRI_eRepositoryLoginStatus = LoginStatus.LoggedIn;
			 }
		    else if (oInfo.AlertType == AlertType.LoginFailed)
			 {
			 PRI_eRepositoryLoginStatus = LoginStatus.LoginFailed;
			 }
		    }

	       if (oInfo.AlertType == AlertType.LoginComplete &&
                   oInfo.ConnectionId == ConnectionId.MarketData)
                    {
                    PRI_bLoggedIntoMd = true;
                    }
               }

          /*   -----------------------------------------------------------   */

	  public override void AgreementList(AgreementListInfo oInfo)
	       {
	       StringBuilder sb = new StringBuilder();
	       oInfo.Dump(sb);
	       Console.Out.Write(sb);

	       foreach (AgreementInfo oAgreement in oInfo.Agreements)
		    {
 		    if (oAgreement.Mandatory &&
		        oAgreement.Status == "active")
			 {
			 PRI_iUnacceptedMandatoryAgreementCount++;
			 }
		    }

	       PRI_bReceivedAgreementList = true;
	       }

          /*   -----------------------------------------------------------   */

          public override void AskQuote(AskInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);
               }

          public override void BestAskQuote(AskInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);
               }

          public override void BestBidAskQuote(BidInfo oBid,
	                                       AskInfo oAsk)
               {
               StringBuilder sb = new StringBuilder();
	       sb.AppendLine("BestBidAskQuote :");
               oBid.Dump(sb);
               oAsk.Dump(sb);
               Console.Out.Write(sb);
               }

          public override void BestBidQuote(BidInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);
               }

          public override void BidQuote(BidInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);
               }

          public override void BinaryContractList(BinaryContractListInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);
               }

          public override void ClosePrice(ClosePriceInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);
               }

          public override void ClosingIndicator(ClosingIndicatorInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);
               }

          public override void EndQuote(EndQuoteInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);
               }

          public override void InstrumentByUnderlying(InstrumentByUnderlyingInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);
               }

          public override void InstrumentSearch(InstrumentSearchInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);
               }
          
          public override void EquityOptionStrategyList(EquityOptionStrategyListInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);
               }
          public override void ExchangeList(ExchangeListInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);
               }

          public override void HighPrice(HighPriceInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);
               }

          public override void LimitOrderBook(OrderBookInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);
               }

          public override void LowPrice(LowPriceInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);
               }

          public override void MarketMode(MarketModeInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);
               }

          public override void OpenInterest(OpenInterestInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);
               }

          public override void OpenPrice(OpenPriceInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);
               }

          public override void OptionList(OptionListInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);
               }

          public override void OpeningIndicator(OpeningIndicatorInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);
               }

          public override void PriceIncrUpdate(PriceIncrInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);

               if (oInfo.RpCode == 0)
                    {
                    PRI_bGotPriceIncrInfo = true;
                    }
               }

          public override void RefData(RefDataInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);
               }

          public override void SettlementPrice(SettlementPriceInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);
               }

          public override void Strategy(StrategyInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);
               }

          public override void StrategyList(StrategyListInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);
               }

          public override void TradeCondition(TradeInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);
               }

          public override void TradePrint(TradeInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);
               }

          public override void TradeVolume(TradeVolumeInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);
               }

          /*   -----------------------------------------------------------   */

          public override void Bar(BarInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);
               }

          public override void BarReplay(BarReplayInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);
               }

          /*   -----------------------------------------------------------   */

          public override void TradeReplay(TradeReplayInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);
               }

          /*   -----------------------------------------------------------   */

          public override void AccountList(AccountListInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               sb.AppendFormat("\n");
               System.Console.Out.Write(sb);
               }

          public override void PasswordChange(PasswordChangeInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               sb.AppendFormat("\n");
               System.Console.Out.Write(sb);
               }

          public override void ProductRmsList(ProductRmsListInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               sb.AppendFormat("\n");
               System.Console.Out.Write(sb);
               }

          public override void ExecutionReplay(ExecutionReplayInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               sb.AppendFormat("\n");
               System.Console.Out.Write(sb);
               }

          public override void LineUpdate(LineInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();

               oInfo.Dump(sb);
               sb.AppendFormat("\n");

               System.Console.Out.Write(sb);
               }
          public override void OpenOrderReplay(OrderReplayInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               sb.AppendFormat("\n");
               System.Console.Out.Write(sb);
               }

          public override void OrderReplay(OrderReplayInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               sb.AppendFormat("\n");
               System.Console.Out.Write(sb);
               }

          public override void SingleOrderReplay(SingleOrderReplayInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               sb.AppendFormat("\n");
               System.Console.Out.Write(sb);
               }

          public override void TradeRoute(TradeRouteInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               sb.AppendFormat("\n");
               System.Console.Out.Write(sb);
               }

          public override void TradeRouteList(TradeRouteListInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               sb.AppendFormat("\n");
               System.Console.Out.Write(sb);
               }

          public override void PnlReplay(PnlReplayInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               
               oInfo.Dump(sb);
               sb.AppendFormat("\n");

               System.Console.Out.Write(sb);
               }

          public override void PnlUpdate(PnlInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();

               oInfo.Dump(sb);
               sb.AppendFormat("\n");

               System.Console.Out.Write(sb);
               }

          public override void SodUpdate(SodReport oReport)
               {
               StringBuilder sb = new StringBuilder();

               oReport.Dump(sb);
               sb.AppendFormat("\n");

               System.Console.Out.Write(sb);
               }

          /*   -----------------------------------------------------------   */

          public override void BustReport(OrderBustReport oReport)
               {
               StringBuilder sb = new StringBuilder();

               oReport.Dump(sb);
               sb.AppendFormat("\n");

               System.Console.Out.Write(sb);
               }

          public override void CancelReport(OrderCancelReport oReport)
               {
               StringBuilder sb = new StringBuilder();

               oReport.Dump(sb);
               sb.AppendFormat("\n");

               System.Console.Out.Write(sb);
               }

          public override void FailureReport(OrderFailureReport oReport)
               {
               StringBuilder sb = new StringBuilder();

               oReport.Dump(sb);
               sb.AppendFormat("\n");

               System.Console.Out.Write(sb);
               }

          public override void FillReport(OrderFillReport oReport)
               {
               StringBuilder sb = new StringBuilder();

               oReport.Dump(sb);
               sb.AppendFormat("\n");

               System.Console.Out.Write(sb);
               }

          public override void ModifyReport(OrderModifyReport oReport)
               {
               StringBuilder sb = new StringBuilder();

               oReport.Dump(sb);
               sb.AppendFormat("\n");

               System.Console.Out.Write(sb);
               }

          public override void NotCancelledReport(OrderNotCancelledReport oReport)
               {
               StringBuilder sb = new StringBuilder();

               oReport.Dump(sb);
               sb.AppendFormat("\n");

               System.Console.Out.Write(sb);
               }

          public override void NotModifiedReport(OrderNotModifiedReport oReport)
               {
               StringBuilder sb = new StringBuilder();

               oReport.Dump(sb);
               sb.AppendFormat("\n");

               System.Console.Out.Write(sb);
               }

          public override void OtherReport(OrderReport oReport)
               {
               StringBuilder sb = new StringBuilder();

               oReport.Dump(sb);
               sb.AppendFormat("\n");

               System.Console.Out.Write(sb);
               }

          public override void RejectReport(OrderRejectReport oReport)
               {
               StringBuilder sb = new StringBuilder();

               oReport.Dump(sb);
               sb.AppendFormat("\n");

               System.Console.Out.Write(sb);
               }

          public override void StatusReport(OrderStatusReport oReport)
               {
               StringBuilder sb = new StringBuilder();

               oReport.Dump(sb);
               sb.AppendFormat("\n");

               System.Console.Out.Write(sb);
               }

          public override void TradeCorrectReport(OrderTradeCorrectReport oReport)
               {
               StringBuilder sb = new StringBuilder();

               oReport.Dump(sb);
               sb.AppendFormat("\n");

               System.Console.Out.Write(sb);
               }

          public override void TriggerPulledReport(OrderTriggerPulledReport oReport)
               {
               StringBuilder sb = new StringBuilder();

               oReport.Dump(sb);
               sb.AppendFormat("\n");

               System.Console.Out.Write(sb);
               }

          public override void TriggerReport(OrderTriggerReport oReport)
               {
               StringBuilder sb = new StringBuilder();

               oReport.Dump(sb);
               sb.AppendFormat("\n");

               System.Console.Out.Write(sb);
               }

          private bool PRI_bLoggedIntoMd;
          private bool PRI_bGotPriceIncrInfo;

	  private LoginStatus  PRI_eRepositoryLoginStatus;
	  private bool         PRI_bReceivedAgreementList;   
	  private int          PRI_iUnacceptedMandatoryAgreementCount;
          }

     /*   ================================================================   */

class Program
     {
     static void Main(string[] args)
          {
          string USAGE = "SampleMD user password exchange symbol";

          if (args.Length < 4)
               {
               System.Console.Out.WriteLine(USAGE);
               return;
               }

          /*   -----------------------------------------------------------   */
          /*   Remember to add the parameters in Visual Studio if debugging  */

          string sUser     = args[0];
          string sPassword = args[1];
          string sExchange = args[2];                 /* CME, for example    */
          string sSymbol   = args[3];                 /* ESM2, for example   */

          /*   -----------------------------------------------------------   */

          MyCallbacks   oCallbacks = new MyCallbacks();
          REngineParams oParams    = new REngineParams();
          REngine       oEngine    = null;

          /*   ----------------------------------------------------------   */

          oParams.AppName      = "SampleMD.NET";
          oParams.AppVersion   = "1.0.0.0";
          oParams.AdmCallbacks = new MyAdmCallbacks();
          oParams.DmnSrvrAddr  = "rituz00100.00.rithmic.com:65000~rituz00100.00.rithmic.net:65000~rituz00100.00.theomne.net:65000~rituz00100.00.theomne.com:65000";
          oParams.DomainName   = "rithmic_uat_dmz_domain";
          oParams.LicSrvrAddr  = "rituz00100.00.rithmic.com:56000~rituz00100.00.rithmic.net:56000~rituz00100.00.theomne.net:56000~rituz00100.00.theomne.com:56000";
          oParams.LocBrokAddr  = "rituz00100.00.rithmic.com:64100";
          oParams.LoggerAddr   = "rituz00100.00.rithmic.com:45454~rituz00100.00.rithmic.net:45454~rituz00100.00.theomne.com:45454~rituz00100.00.theomne.net:45454";
          oParams.LogFilePath  = "smd.log";

          /*   -----------------------------------------------------------   */

          try
               {
               /*   ------------------------------------------------------   */
               /*   Instantiate the REngine.                                 */
               /*   ------------------------------------------------------   */

               oEngine = new REngine(oParams);

               /*   ------------------------------------------------------   */
               /*   First, log in to the repository to check agreements.     */
	       /*   (See the FAQ for more details on agreements.)            */
               /*   ------------------------------------------------------   */

               oEngine.loginRepository(oCallbacks,
                    string.Empty,
                    sUser, 
                    sPassword,
                    "login_agent_repositoryc");

	       oCallbacks.RepositoryLoginStatus = MyCallbacks.LoginStatus.LoginInProgress;

               /*   ------------------------------------------------------   */
               /*   After calling REngine::loginRepository,                  */
	       /*   RCallbacks::Alert will be called a number of times.      */
               /*   Wait until the login succeeds or fails.                  */
               /*   ------------------------------------------------------   */

               while (oCallbacks.RepositoryLoginStatus != MyCallbacks.LoginStatus.LoggedIn &&
		      oCallbacks.RepositoryLoginStatus != MyCallbacks.LoginStatus.LoginFailed)
                    {
                    System.Threading.Thread.Sleep(1000);
                    }

	       if (oCallbacks.RepositoryLoginStatus == MyCallbacks.LoginStatus.LoginFailed)
		    {
		    System.Console.Out.WriteLine("Please make sure you entered the username and password "
			 + "correctly.  Also, please make sure your credentials match "
			 + "the system you are trying to log in to.");

		    oEngine.shutdown();
		    return;
		    }

               /*   ------------------------------------------------------   */
               /*   Once logged in to the repository, we can request a       */
	       /*   list of unaccepted agreements.                           */
               /*   ------------------------------------------------------   */

	       oEngine.listAgreements(false, null);

               /*   ------------------------------------------------------   */
               /*   Wait for the list to arrive from the infrastructure.     */
               /*   ------------------------------------------------------   */

	       while (!oCallbacks.ReceivedAgreementList)
                    {
                    System.Threading.Thread.Sleep(1000);
                    }

               /*   ------------------------------------------------------   */
               /*   If there are any unaccepted mandatory agreements, you    */
	       /*   must log in using R | Trader (Pro) and accept them       */
	       /*   before being able to log in to the trading platform.     */
               /*   ------------------------------------------------------   */

	       if (oCallbacks.UnacceptedMandatoryAgreementCount > 0)
		    {
		    System.Console.Out.WriteLine("Please log in using R | Trader or R | Trader Pro" 
			 + " and sign the agreements.");

		    oEngine.logoutRepository();
		    oEngine.shutdown();
		    return;
		    }

               /*   ------------------------------------------------------   */
               /*   Log out from the repository.  We are done with it ...    */
               /*   ------------------------------------------------------   */

	       oEngine.logoutRepository();

               /*   ------------------------------------------------------   */
               /*   Initiate the login to the trading platform.              */
               /*   ------------------------------------------------------   */

               oEngine.login(oCallbacks, 
                    string.Empty,  // empty is same as default env key
                    sUser, 
                    sPassword,
                    "login_agent_tpc", 
                    Constants.DEFAULT_ENVIRONMENT_KEY,
                    sUser, 
                    sPassword,
                    "login_agent_opc", 
                    string.Empty,  // sPnlCnnctPt
                    string.Empty,  // sIhEnvKey
		    string.Empty,  // sIhUser
		    string.Empty,  // sIhPassword
                    string.Empty); // sIhCnnctPt


               /*   ------------------------------------------------------   */
               /*   After calling REngine::login, RCallbacks::Alert will be  */
               /*   called a number of times.  Wait for when the login to    */
               /*   the MdCnnctPt and TsCnnctPt is complete.  (See           */
               /*   MyCallbacks::Alert() for details).                       */
               /*   ------------------------------------------------------   */

               while (!oCallbacks.LoggedIntoMd)
                    {
                    System.Threading.Thread.Sleep(1000);
                    }

               /*   ------------------------------------------------------   */
               /*   Subscribe to the instrument of interest.  To express     */
               /*   interest in different types of updates, see              */
               /*   SubscriptionFlags.                                       */
               /*   ------------------------------------------------------   */

               oEngine.subscribe(sExchange, 
                    sSymbol, 
                    (SubscriptionFlags.Prints | SubscriptionFlags.Best), 
                    null);

               /*   ------------------------------------------------------   */
               /*   At this point the callback routines will start firing    */
               /*   on a different thread.  This main thread will wait       */
               /*   until a key is pressed before continuing.                */
               /*   ------------------------------------------------------   */

               Console.Read();

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
