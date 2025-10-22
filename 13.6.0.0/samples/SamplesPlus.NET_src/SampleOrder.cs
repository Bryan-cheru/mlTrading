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

namespace SampleOrderNamespace
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
          public AccountInfo Account
               {
               get { return PRI_oAccount; }
               }

          /*   -----------------------------------------------------------   */

          public bool GotAccounts
               {
               get { return PRI_bGotAccounts; }
               }

          /*   -----------------------------------------------------------   */

          public bool GotTradeRoutes
               {
               get { return PRI_bGotTradeRoutes; }
               }

          /*   -----------------------------------------------------------   */

          public bool LoggedIntoMd
               {
               get { return PRI_bLoggedIntoMd; }
               }

          /*   -----------------------------------------------------------   */

          public bool LoggedIntoTs
               {
               get { return PRI_bLoggedIntoTs; }
               }

          /*   -----------------------------------------------------------   */

          public bool OrderComplete
               {
               get { return PRI_bOrderComplete; }
               }

          /*   -----------------------------------------------------------   */

          public string Exchange
               {
               get { return PRI_sExchange; }
               set { PRI_sExchange = value; }
               }

          /*   -----------------------------------------------------------   */

          public string TradeRouteToUse
               {
               get { return PRI_sTradeRoute; }
               set { PRI_sTradeRoute = value; }
               }

          /*   -----------------------------------------------------------   */

          public MyCallbacks()
               {
               PRI_oAccount = null;
               PRI_bGotAccounts = false;

               PRI_bGotTradeRoutes = false;

               PRI_bLoggedIntoMd = false;
               PRI_bLoggedIntoTs = false;

               PRI_bOrderComplete = false;

               PRI_sExchange = string.Empty;
               PRI_sTradeRoute = string.Empty;
               }

          /*   -----------------------------------------------------------   */

          public override void Alert(AlertInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);

               if (oInfo.AlertType == AlertType.LoginComplete &&
                   oInfo.ConnectionId == ConnectionId.MarketData)
                    {
                    PRI_bLoggedIntoMd = true;
                    }
               if (oInfo.AlertType == AlertType.LoginComplete &&
                   oInfo.ConnectionId == ConnectionId.TradingSystem)
                    {
                    PRI_bLoggedIntoTs = true;
                    }
               }

          /*   -----------------------------------------------------------   */

          public override void AskQuote(AskInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);
               }

          /*   -----------------------------------------------------------   */

          public override void BestAskQuote(AskInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);
               }

          /*   -----------------------------------------------------------   */

          public override void BestBidAskQuote(BidInfo oBid,
	                                       AskInfo oAsk)
               {
               StringBuilder sb = new StringBuilder();
	       sb.AppendLine("BestBidAskQuote :");
               oBid.Dump(sb);
               oAsk.Dump(sb);
               Console.Out.Write(sb);
               }

          /*   -----------------------------------------------------------   */

          public override void BestBidQuote(BidInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);
               }

          /*   -----------------------------------------------------------   */

          public override void BidQuote(BidInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);
               }

          /*   -----------------------------------------------------------   */

          public override void BinaryContractList(BinaryContractListInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);
               }

          /*   -----------------------------------------------------------   */

          public override void ClosePrice(ClosePriceInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);
               }

          /*   -----------------------------------------------------------   */

          public override void ClosingIndicator(ClosingIndicatorInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);
               }

          /*   -----------------------------------------------------------   */

          public override void EndQuote(EndQuoteInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);
               }

          /*   -----------------------------------------------------------   */

          public override void EquityOptionStrategyList(EquityOptionStrategyListInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);
               }

          /*   -----------------------------------------------------------   */

          public override void ExchangeList(ExchangeListInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);
               }

          /*   -----------------------------------------------------------   */

          public override void HighPrice(HighPriceInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);
               }

          /*   -----------------------------------------------------------   */

          public override void InstrumentByUnderlying(InstrumentByUnderlyingInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);
               }

          /*   -----------------------------------------------------------   */

          public override void InstrumentSearch(InstrumentSearchInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);
               }
          
          /*   -----------------------------------------------------------   */

          public override void LimitOrderBook(OrderBookInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);
               }

          /*   -----------------------------------------------------------   */

          public override void LowPrice(LowPriceInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);
               }

          /*   -----------------------------------------------------------   */

          public override void MarketMode(MarketModeInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);
               }

          /*   -----------------------------------------------------------   */

          public override void OpenInterest(OpenInterestInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);
               }

          /*   -----------------------------------------------------------   */

          public override void OpenPrice(OpenPriceInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);
               }

          /*   -----------------------------------------------------------   */

          public override void OptionList(OptionListInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);
               }

          /*   -----------------------------------------------------------   */

          public override void OpeningIndicator(OpeningIndicatorInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);
               }

          /*   -----------------------------------------------------------   */

          public override void PriceIncrUpdate(PriceIncrInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);
               }

          /*   -----------------------------------------------------------   */

          public override void RefData(RefDataInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);
               }

          /*   -----------------------------------------------------------   */

          public override void SettlementPrice(SettlementPriceInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);
               }

          /*   -----------------------------------------------------------   */

          public override void Strategy(StrategyInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);
               }

          /*   -----------------------------------------------------------   */

          public override void StrategyList(StrategyListInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);
               }

          /*   -----------------------------------------------------------   */

          public override void TradeCondition(TradeInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);
               }

          /*   -----------------------------------------------------------   */

          public override void TradePrint(TradeInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               Console.Out.Write(sb);
               }

          /*   -----------------------------------------------------------   */

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

               PRI_bGotAccounts = true;
               if (oInfo.Accounts.Count > 0)
                    {
                    PRI_oAccount = new AccountInfo(oInfo.Accounts[0].FcmId,
                                                   oInfo.Accounts[0].IbId,
                                                   oInfo.Accounts[0].AccountId);
                    }
               }

          /*   -----------------------------------------------------------   */

          public override void PasswordChange(PasswordChangeInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               sb.AppendFormat("\n");
               System.Console.Out.Write(sb);
               }

          /*   -----------------------------------------------------------   */

          public override void ProductRmsList(ProductRmsListInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               sb.AppendFormat("\n");
               System.Console.Out.Write(sb);
               }

          /*   -----------------------------------------------------------   */

          public override void ExecutionReplay(ExecutionReplayInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               sb.AppendFormat("\n");
               System.Console.Out.Write(sb);
               }

          /*   -----------------------------------------------------------   */

          public override void LineUpdate(LineInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();

               oInfo.Dump(sb);
               sb.AppendFormat("\n");

               System.Console.Out.Write(sb);

               if (!string.IsNullOrEmpty(oInfo.CompletionReason))
                    {
                    PRI_bOrderComplete = true;
                    }
               }

          /*   -----------------------------------------------------------   */

          public override void OpenOrderReplay(OrderReplayInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               sb.AppendFormat("\n");
               System.Console.Out.Write(sb);
               }

          /*   -----------------------------------------------------------   */

          public override void OrderReplay(OrderReplayInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               sb.AppendFormat("\n");
               System.Console.Out.Write(sb);
               }

          /*   -----------------------------------------------------------   */

          public override void SingleOrderReplay(SingleOrderReplayInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               sb.AppendFormat("\n");
               System.Console.Out.Write(sb);
               }
          
          /*   -----------------------------------------------------------   */

          public override void TradeRoute(TradeRouteInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               oInfo.Dump(sb);
               sb.AppendFormat("\n");
               System.Console.Out.Write(sb);
               }

          /*   -----------------------------------------------------------   */

          public override void TradeRouteList(TradeRouteListInfo oInfo)
               {
               for (int i = 0; i < oInfo.TradeRoutes.Count; i++)
                    {
                    if (oInfo.TradeRoutes[i].FcmId    == PRI_oAccount.FcmId &&
                        oInfo.TradeRoutes[i].IbId     == PRI_oAccount.IbId  &&
                        oInfo.TradeRoutes[i].Exchange == PRI_sExchange &&
                        oInfo.TradeRoutes[i].Status   == Constants.TRADE_ROUTE_STATUS_UP)
                         {
                         PRI_sTradeRoute = oInfo.TradeRoutes[i].TradeRoute;
                         break;
                         }
                    }

               PRI_bGotTradeRoutes = true;
               }

          /*   -----------------------------------------------------------   */

          public override void PnlReplay(PnlReplayInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();
               
               oInfo.Dump(sb);
               sb.AppendFormat("\n");

               System.Console.Out.Write(sb);
               }

          /*   -----------------------------------------------------------   */

          public override void PnlUpdate(PnlInfo oInfo)
               {
               StringBuilder sb = new StringBuilder();

               oInfo.Dump(sb);
               sb.AppendFormat("\n");

               System.Console.Out.Write(sb);
               }

          /*   -----------------------------------------------------------   */

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

          /*   -----------------------------------------------------------   */

          public override void CancelReport(OrderCancelReport oReport)
               {
               StringBuilder sb = new StringBuilder();

               oReport.Dump(sb);
               sb.AppendFormat("\n");

               System.Console.Out.Write(sb);
               }

          /*   -----------------------------------------------------------   */

          public override void FailureReport(OrderFailureReport oReport)
               {
               StringBuilder sb = new StringBuilder();

               oReport.Dump(sb);
               sb.AppendFormat("\n");

               System.Console.Out.Write(sb);
               }

          /*   -----------------------------------------------------------   */

          public override void FillReport(OrderFillReport oReport)
               {
               StringBuilder sb = new StringBuilder();

               oReport.Dump(sb);
               sb.AppendFormat("\n");

               System.Console.Out.Write(sb);
               }

          /*   -----------------------------------------------------------   */

          public override void ModifyReport(OrderModifyReport oReport)
               {
               StringBuilder sb = new StringBuilder();

               oReport.Dump(sb);
               sb.AppendFormat("\n");

               System.Console.Out.Write(sb);
               }

          /*   -----------------------------------------------------------   */

          public override void NotCancelledReport(OrderNotCancelledReport oReport)
               {
               StringBuilder sb = new StringBuilder();

               oReport.Dump(sb);
               sb.AppendFormat("\n");

               System.Console.Out.Write(sb);
               }

          /*   -----------------------------------------------------------   */

          public override void NotModifiedReport(OrderNotModifiedReport oReport)
               {
               StringBuilder sb = new StringBuilder();

               oReport.Dump(sb);
               sb.AppendFormat("\n");

               System.Console.Out.Write(sb);
               }

          /*   -----------------------------------------------------------   */

          public override void OtherReport(OrderReport oReport)
               {
               StringBuilder sb = new StringBuilder();

               oReport.Dump(sb);
               sb.AppendFormat("\n");

               System.Console.Out.Write(sb);
               }

          /*   -----------------------------------------------------------   */

          public override void RejectReport(OrderRejectReport oReport)
               {
               StringBuilder sb = new StringBuilder();

               oReport.Dump(sb);
               sb.AppendFormat("\n");

               System.Console.Out.Write(sb);
               }

          /*   -----------------------------------------------------------   */

          public override void StatusReport(OrderStatusReport oReport)
               {
               StringBuilder sb = new StringBuilder();

               oReport.Dump(sb);
               sb.AppendFormat("\n");

               System.Console.Out.Write(sb);
               }

          /*   -----------------------------------------------------------   */

          public override void TradeCorrectReport(OrderTradeCorrectReport oReport)
               {
               StringBuilder sb = new StringBuilder();

               oReport.Dump(sb);
               sb.AppendFormat("\n");

               System.Console.Out.Write(sb);
               }

          /*   -----------------------------------------------------------   */

          public override void TriggerPulledReport(OrderTriggerPulledReport oReport)
               {
               StringBuilder sb = new StringBuilder();

               oReport.Dump(sb);
               sb.AppendFormat("\n");

               System.Console.Out.Write(sb);
               }

          /*   -----------------------------------------------------------   */

          public override void TriggerReport(OrderTriggerReport oReport)
               {
               StringBuilder sb = new StringBuilder();

               oReport.Dump(sb);
               sb.AppendFormat("\n");

               System.Console.Out.Write(sb);
               }

          /*   -----------------------------------------------------------   */

          private AccountInfo PRI_oAccount;
          private string PRI_sExchange;
          private string PRI_sTradeRoute;
          private bool PRI_bGotAccounts;
          private bool PRI_bGotTradeRoutes;
          private bool PRI_bLoggedIntoMd;
          private bool PRI_bLoggedIntoTs;
          private bool PRI_bOrderComplete;
          }

     /*   ================================================================   */

class Program
     {
     static void Main(string[] args)
          {
          string USAGE = "SampleOrder user password exchange symbol [B|S]";

          if (args.Length < 5)
               {
               System.Console.Out.WriteLine(USAGE);
               return;
               }

          /*   -----------------------------------------------------------   */
          /*   See Constants.BUY_SELL_TYPE_BUY and related constants for     */ 
          /*   more information about specifying order side.                 */

          string sUser        = args[0];
          string sPassword    = args[1];
          string sExchange    = args[2];    /* CME, for example              */
          string sSymbol      = args[3];    /* ESM2, for example             */
          string sBuySellType = args[4];    /* "B" or "S", for example       */

          /*   -----------------------------------------------------------   */

          MyCallbacks   oCallbacks = new MyCallbacks();
          REngineParams oParams    = new REngineParams();
          REngine       oEngine;

          /*   ----------------------------------------------------------   */

          oParams.AppName      = "SampleOrder.NET";
          oParams.AppVersion   = "1.0.0.0";
          oParams.AdmCallbacks = new MyAdmCallbacks();
          oParams.DmnSrvrAddr  = "rituz00100.00.rithmic.com:65000~rituz00100.00.rithmic.net:65000~rituz00100.00.theomne.net:65000~rituz00100.00.theomne.com:65000";
          oParams.DomainName   = "rithmic_uat_dmz_domain";
          oParams.LicSrvrAddr  = "rituz00100.00.rithmic.com:56000~rituz00100.00.rithmic.net:56000~rituz00100.00.theomne.net:56000~rituz00100.00.theomne.com:56000";
          oParams.LocBrokAddr  = "rituz00100.00.rithmic.com:64100";
          oParams.LoggerAddr   = "rituz00100.00.rithmic.com:45454~rituz00100.00.rithmic.net:45454~rituz00100.00.theomne.com:45454~rituz00100.00.theomne.net:45454";
          oParams.LogFilePath  = "so.log";

          /*   ----------------------------------------------------------   */

          try
               {
               /*   ------------------------------------------------------   */
               /*   Instantiate the REngine.                                 */
               /*   ------------------------------------------------------   */

               oEngine = new REngine(oParams);

               /*   ------------------------------------------------------   */
               /*   Initiate the login.                                      */
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

               while (!oCallbacks.LoggedIntoMd || !oCallbacks.LoggedIntoTs)
                    {
                    System.Threading.Thread.Sleep(1000);
                    }

               /*   ------------------------------------------------------   */
               /*   Wait for the AccountList callback to fire, so we know    */
               /*   which accounts we are permissioned on.  The account on   */
               /*   which we place the order will be the first account in    */
               /*   the list.  See MyCallbacks::AccountList() for details.   */
               /*   ------------------------------------------------------   */

               while (!oCallbacks.GotAccounts)
                    {
                    System.Threading.Thread.Sleep(1000);
                    }

               /*   ------------------------------------------------------   */

               if (oCallbacks.Account == null)
                    {
                    System.Console.WriteLine("Error : didn't get an account");
                    return;
                    }

               /*   ------------------------------------------------------   */
               /*   Subscribe to order activity.  By doing so, we will       */
               /*   receive updates for orders placed on the account.        */
               /*   ------------------------------------------------------   */

               oEngine.subscribeOrder(oCallbacks.Account);
               
               /*   ------------------------------------------------------   */
               /*   Get the list of available trade routes.  See             */
               /*   MyCallbacks::TradeRouteList() for details.               */
               /*   ------------------------------------------------------   */

               oCallbacks.Exchange = sExchange;

               oEngine.listTradeRoutes(null);

               while (!oCallbacks.GotTradeRoutes)
                    {
                    System.Threading.Thread.Sleep(1000);
                    }

               if (string.IsNullOrEmpty(oCallbacks.TradeRouteToUse))
                    {
                    System.Console.WriteLine("Error : no available trade routes which are up");
                    return;
                    }

               /*   ------------------------------------------------------   */
               /*   Prepare the order params and then send it.               */
               /*   ------------------------------------------------------   */

               MarketOrderParams oOrderParams = new MarketOrderParams();
               oOrderParams.Account = oCallbacks.Account;
               oOrderParams.BuySellType = sBuySellType;
               oOrderParams.Context = null;
               oOrderParams.Duration = Constants.ORDER_DURATION_DAY;
               oOrderParams.EntryType = Constants.ORDER_ENTRY_TYPE_MANUAL;
               oOrderParams.Exchange = sExchange;
               oOrderParams.Qty = 1;
               oOrderParams.Symbol = sSymbol;
               oOrderParams.Tag = null;
               oOrderParams.TradeRoute = oCallbacks.TradeRouteToUse;
               oOrderParams.UserMsg = null;

               oEngine.sendOrder(oOrderParams);

               /*   ------------------------------------------------------   */
               /*   Wait for the order to complete.  A number of related     */
               /*   callbacks will fire, but the one controlling the status  */
               /*   of complete is done in MyCallbacks::LineUpdate().        */
               /*   ------------------------------------------------------   */

               while (!oCallbacks.OrderComplete)
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

          /*   ----------------------------------------------------------   */

          return;
          }
     }
}
