"""
NinjaTrader Integration for Institutional Strategies
Replaces single-instrument ML with pairs arbitrage execution
"""

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Xml.Serialization;
using NinjaTrader.Cbi;
using NinjaTrader.Gui;
using NinjaTrader.Gui.Chart;
using NinjaTrader.Gui.SuperDom;
using NinjaTrader.Gui.Tools;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.Core.FloatingPoint;
using NinjaTrader.NinjaScript.Indicators;
using NinjaTrader.NinjaScript.DrawingTools;
using System.IO;
using Newtonsoft.Json;

namespace NinjaTrader.NinjaScript.Strategies
{
    /// <summary>
    /// Institutional Statistical Arbitrage Strategy
    /// Replaces ML prediction with pairs trading, volatility arbitrage, and flow-based strategies
    /// Based on Renaissance Technologies / Two Sigma institutional approach
    /// </summary>
    public class InstitutionalStatArb : Strategy
    {
        #region Variables
        
        // === PAIRS TRADING COMPONENTS ===
        private Dictionary<string, List<double>> priceHistory;
        private Dictionary<string, double> currentPrices;
        private Dictionary<string, PairConfig> pairConfigs;
        private Dictionary<string, PairPosition> activePairs;
        
        // === STATISTICAL COMPONENTS ===
        private Dictionary<string, double> hedgeRatios;
        private Dictionary<string, RollingStats> spreadStats;
        private Dictionary<string, double> currentZScores;
        
        // === RISK MANAGEMENT ===
        private double dailyPnL;
        private double dailyHighWater;
        private double maxDailyLoss;
        private bool isRiskOff;
        private List<string> riskViolations;
        
        // === PERFORMANCE TRACKING ===
        private int signalsGenerated;
        private int tradesExecuted;
        private double totalPnL;
        private DateTime lastResetTime;
        
        // === NEWS & FLOW MANAGEMENT ===
        private List<DateTime> newsBlackoutTimes;
        private Dictionary<string, DateTime> rebalancingDates;
        
        #endregion
        
        #region Strategy Parameters
        
        [NinjaScriptProperty]
        [Display(Name = "ES-NQ Lookback", Description = "Rolling window for ES-NQ pair", Order = 1, GroupName = "Pairs Config")]
        public int ESNQLookback { get; set; }
        
        [NinjaScriptProperty]
        [Display(Name = "ES-NQ Entry Z-Score", Description = "Z-score threshold for ES-NQ entry", Order = 2, GroupName = "Pairs Config")]
        public double ESNQEntryZ { get; set; }
        
        [NinjaScriptProperty]
        [Display(Name = "ES-NQ Exit Z-Score", Description = "Z-score threshold for ES-NQ exit", Order = 3, GroupName = "Pairs Config")]
        public double ESNQExitZ { get; set; }
        
        [NinjaScriptProperty]
        [Display(Name = "ZN-ZB Lookback", Description = "Rolling window for ZN-ZB pair", Order = 4, GroupName = "Pairs Config")]
        public int ZNZBLookback { get; set; }
        
        [NinjaScriptProperty]
        [Display(Name = "ZN-ZB Entry Z-Score", Description = "Z-score threshold for ZN-ZB entry", Order = 5, GroupName = "Pairs Config")]
        public double ZNZBEntryZ { get; set; }
        
        [NinjaScriptProperty]
        [Display(Name = "ZN-ZB Exit Z-Score", Description = "Z-score threshold for ZN-ZB exit", Order = 6, GroupName = "Pairs Config")]
        public double ZNZBExitZ { get; set; }
        
        [NinjaScriptProperty]
        [Display(Name = "Max Daily Loss %", Description = "Maximum daily loss as percentage", Order = 7, GroupName = "Risk Management")]
        public double MaxDailyLossPercent { get; set; }
        
        [NinjaScriptProperty]
        [Display(Name = "Position Size", Description = "Base position size for pairs", Order = 8, GroupName = "Risk Management")]
        public int PositionSize { get; set; }
        
        [NinjaScriptProperty]
        [Display(Name = "Enable News Blackout", Description = "Avoid trading during news events", Order = 9, GroupName = "Risk Management")]
        public bool EnableNewsBlackout { get; set; }
        
        [NinjaScriptProperty]
        [Display(Name = "Trading Start Hour", Description = "Start trading hour (24h format)", Order = 10, GroupName = "Time Filters")]
        public int TradingStartHour { get; set; }
        
        [NinjaScriptProperty]
        [Display(Name = "Trading End Hour", Description = "End trading hour (24h format)", Order = 11, GroupName = "Time Filters")]
        public int TradingEndHour { get; set; }
        
        #endregion
        
        #region Strategy Lifecycle
        
        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "Institutional Statistical Arbitrage - Pairs Trading, Vol Arb, Flow-Based Strategies";
                Name = "InstitutionalStatArb";
                Calculate = Calculate.OnBarClose;
                EntriesPerDirection = 10;
                EntryHandling = EntryHandling.AllEntries;
                IsExitOnSessionCloseStrategy = false;
                ExitOnSessionCloseSeconds = 30;
                IsFillLimitOnTouch = false;
                MaximumBarsLookBack = MaximumBarsLookBack.TwoHundredFiftySix;
                OrderFillResolution = OrderFillResolution.Standard;
                Slippage = 0;
                StartBehavior = StartBehavior.WaitUntilFlat;
                TimeInForce = TimeInForce.Gtc;
                TraceOrders = false;
                RealtimeErrorHandling = RealtimeErrorHandling.StopCancelClose;
                StopTargetHandling = StopTargetHandling.PerEntryExecution;
                BarsRequiredToTrade = 300;
                IsInstantiatedOnEachOptimizationIteration = true;
                
                // === DEFAULT PARAMETERS ===
                ESNQLookback = 300;
                ESNQEntryZ = 2.0;
                ESNQExitZ = 0.3;
                ZNZBLookback = 240;
                ZNZBEntryZ = 1.8;
                ZNZBExitZ = 0.25;
                MaxDailyLossPercent = 2.0;
                PositionSize = 1;
                EnableNewsBlackout = true;
                TradingStartHour = 6;
                TradingEndHour = 21;
            }
            else if (State == State.Configure)
            {
                // === ADD DATA SERIES FOR PAIRS ===
                // ES-NQ Pair
                AddDataSeries("ES 12-25", Data.BarsPeriodType.Minute, 1);
                AddDataSeries("NQ 12-25", Data.BarsPeriodType.Minute, 1);
                
                // ZN-ZB Pair (Treasury futures)
                AddDataSeries("ZN 12-25", Data.BarsPeriodType.Minute, 1);
                AddDataSeries("ZB 12-25", Data.BarsPeriodType.Minute, 1);
                
                // Optional: VIX for volatility arbitrage
                // AddDataSeries("VIX 12-25", Data.BarsPeriodType.Minute, 5);
            }
            else if (State == State.DataLoaded)
            {
                InitializeInstitutionalComponents();
                LoadNewsAndRebalancingCalendar();
                
                Print($"📊 Institutional Statistical Arbitrage initialized");
                Print($"🎯 Monitoring: ES-NQ pairs, ZN-ZB spreads");
                Print($"⚠️ Daily loss limit: {MaxDailyLossPercent}%");
            }
        }
        
        #endregion
        
        #region Initialization
        
        private void InitializeInstitutionalComponents()
        {
            // === INITIALIZE DATA STRUCTURES ===
            priceHistory = new Dictionary<string, List<double>>();
            currentPrices = new Dictionary<string, double>();
            activePairs = new Dictionary<string, PairPosition>();
            hedgeRatios = new Dictionary<string, double>();
            spreadStats = new Dictionary<string, RollingStats>();
            currentZScores = new Dictionary<string, double>();
            riskViolations = new List<string>();
            
            // === CONFIGURE PAIRS ===
            pairConfigs = new Dictionary<string, PairConfig>
            {
                ["ES_NQ"] = new PairConfig
                {
                    SymbolA = "ES",
                    SymbolB = "NQ", 
                    BarsInProgressA = 1,
                    BarsInProgressB = 2,
                    Lookback = ESNQLookback,
                    EntryZScore = ESNQEntryZ,
                    ExitZScore = ESNQExitZ,
                    MaxHoldingBars = 480,
                    VolFilterThreshold = 0.02
                },
                ["ZN_ZB"] = new PairConfig
                {
                    SymbolA = "ZN",
                    SymbolB = "ZB",
                    BarsInProgressA = 3,
                    BarsInProgressB = 4,
                    Lookback = ZNZBLookback,
                    EntryZScore = ZNZBEntryZ,
                    ExitZScore = ZNZBExitZ,
                    MaxHoldingBars = 720,
                    VolFilterThreshold = 0.015
                }
            };
            
            // === INITIALIZE PRICE HISTORY ===
            foreach (var pair in pairConfigs.Keys)
            {
                var config = pairConfigs[pair];
                priceHistory[config.SymbolA] = new List<double>();
                priceHistory[config.SymbolB] = new List<double>();
                spreadStats[pair] = new RollingStats();
            }
            
            // === RISK MANAGEMENT ===
            maxDailyLoss = MaxDailyLossPercent / 100.0;
            dailyPnL = 0.0;
            dailyHighWater = Account.Get(AccountItem.NetLiquidation, Currency.UsDollar);
            isRiskOff = false;
            
            // === PERFORMANCE TRACKING ===
            signalsGenerated = 0;
            tradesExecuted = 0;
            totalPnL = 0.0;
            lastResetTime = DateTime.Now.Date;
        }
        
        private void LoadNewsAndRebalancingCalendar()
        {
            // === NEWS BLACKOUT TIMES ===
            newsBlackoutTimes = new List<DateTime>();
            
            if (EnableNewsBlackout)
            {
                // Major economic events (would load from external calendar)
                newsBlackoutTimes.AddRange(new[]
                {
                    new DateTime(2025, 10, 10, 12, 30, 0), // CPI 8:30 AM ET
                    new DateTime(2025, 10, 15, 18, 0, 0),  // FOMC 2:00 PM ET
                    new DateTime(2025, 11, 1, 12, 30, 0),  // NFP 8:30 AM ET
                });
            }
            
            // === INDEX REBALANCING DATES ===
            rebalancingDates = new Dictionary<string, DateTime>
            {
                ["Russell_Rebalance"] = new DateTime(2025, 6, 27),
                ["SP500_Quarterly"] = new DateTime(2025, 12, 19),
                ["MSCI_Quarterly"] = new DateTime(2025, 11, 28)
            };
        }
        
        #endregion
        
        #region Main Strategy Logic
        
        protected override void OnBarUpdate()
        {
            try
            {
                // === DAILY RESET CHECK ===
                CheckDailyReset();
                
                // === UPDATE PRICE DATA ===
                UpdatePriceData();
                
                // === RISK CHECKS ===
                if (isRiskOff || !IsWithinTradingHours() || IsNewsBlackout())
                {
                    return;
                }
                
                // === PROCESS EACH PAIR ===
                foreach (var pairName in pairConfigs.Keys)
                {
                    ProcessPairStrategy(pairName);
                }
                
                // === MONITOR EXISTING POSITIONS ===
                MonitorActivePairs();
                
            }
            catch (Exception ex)
            {
                Print($"❌ Error in OnBarUpdate: {ex.Message}");
            }
        }
        
        private void ProcessPairStrategy(string pairName)
        {
            var config = pairConfigs[pairName];
            
            // === CHECK DATA AVAILABILITY ===
            if (!HasSufficientData(config))
                return;
            
            // === CALCULATE STATISTICAL MEASURES ===
            var hedgeRatio = CalculateRollingHedgeRatio(config);
            var spreadValue = CalculateCurrentSpread(config, hedgeRatio);
            var zScore = CalculateZScore(pairName, spreadValue);
            
            // === STORE CURRENT VALUES ===
            hedgeRatios[pairName] = hedgeRatio;
            currentZScores[pairName] = zScore;
            
            // === GENERATE SIGNALS ===
            var signal = GeneratePairSignal(pairName, config, zScore, hedgeRatio);
            
            if (signal.SignalType != "HOLD")
            {
                signalsGenerated++;
                
                // === RISK CHECK ===
                if (CheckSignalRisk(signal))
                {
                    ExecutePairSignal(signal);
                }
            }
        }
        
        private PairSignal GeneratePairSignal(string pairName, PairConfig config, double zScore, double hedgeRatio)
        {
            var signal = new PairSignal
            {
                PairName = pairName,
                SignalType = "HOLD",
                ZScore = zScore,
                HedgeRatio = hedgeRatio,
                Confidence = 0.0,
                Timestamp = Time[0]
            };
            
            // === ENTRY SIGNALS ===
            if (Math.Abs(zScore) > config.EntryZScore)
            {
                if (zScore > 0)
                {
                    signal.SignalType = "SHORT_PAIR"; // Short B, Long A
                }
                else
                {
                    signal.SignalType = "LONG_PAIR";  // Long B, Short A
                }
                
                signal.Confidence = Math.Min(Math.Abs(zScore) / config.EntryZScore, 1.0);
            }
            
            // === EXIT SIGNALS FOR EXISTING POSITIONS ===
            if (activePairs.ContainsKey(pairName))
            {
                if (Math.Abs(zScore) < config.ExitZScore)
                {
                    signal.SignalType = "EXIT_PAIR";
                    signal.Confidence = 1.0 - Math.Abs(zScore) / config.ExitZScore;
                }
            }
            
            return signal;
        }
        
        #endregion
        
        #region Statistical Calculations
        
        private double CalculateRollingHedgeRatio(PairConfig config)
        {
            var pricesA = priceHistory[config.SymbolA];
            var pricesB = priceHistory[config.SymbolB];
            
            if (pricesA.Count < config.Lookback || pricesB.Count < config.Lookback)
                return 1.0; // Default hedge ratio
            
            // === OLS REGRESSION ===
            var recentA = pricesA.Skip(pricesA.Count - config.Lookback).Select(Math.Log).ToArray();
            var recentB = pricesB.Skip(pricesB.Count - config.Lookback).Select(Math.Log).ToArray();
            
            return CalculateOLSSlope(recentA, recentB);
        }
        
        private double CalculateOLSSlope(double[] x, double[] y)
        {
            var n = x.Length;
            var sumX = x.Sum();
            var sumY = y.Sum();
            var sumXY = x.Zip(y, (a, b) => a * b).Sum();
            var sumX2 = x.Sum(a => a * a);
            
            var slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
            
            // === VALIDATE HEDGE RATIO ===
            if (slope <= 0 || slope > 10)
                return 1.0; // Return safe default
            
            return slope;
        }
        
        private double CalculateCurrentSpread(PairConfig config, double hedgeRatio)
        {
            var priceA = currentPrices[config.SymbolA];
            var priceB = currentPrices[config.SymbolB];
            
            return Math.Log(priceB) - hedgeRatio * Math.Log(priceA);
        }
        
        private double CalculateZScore(string pairName, double currentSpread)
        {
            var config = pairConfigs[pairName];
            var stats = spreadStats[pairName];
            
            // === UPDATE ROLLING STATISTICS ===
            stats.AddValue(currentSpread);
            
            if (stats.Count < 50) // Need minimum data for stable stats
                return 0.0;
            
            var mean = stats.Mean;
            var stdDev = stats.StandardDeviation;
            
            if (stdDev <= 0)
                return 0.0;
            
            return (currentSpread - mean) / stdDev;
        }
        
        #endregion
        
        #region Risk Management
        
        private bool CheckSignalRisk(PairSignal signal)
        {
            riskViolations.Clear();
            
            // === DAILY LOSS LIMIT ===
            var currentCapital = Account.Get(AccountItem.NetLiquidation, Currency.UsDollar);
            var dailyDrawdown = (dailyHighWater - currentCapital) / dailyHighWater;
            
            if (dailyDrawdown > maxDailyLoss)
            {
                riskViolations.Add($"Daily loss limit exceeded: {dailyDrawdown:P2}");
                isRiskOff = true;
            }
            
            // === CONFIDENCE THRESHOLD ===
            if (signal.Confidence < 0.6)
            {
                riskViolations.Add($"Signal confidence too low: {signal.Confidence:F2}");
            }
            
            // === VOLATILITY FILTER ===
            var recentVol = CalculateRealizedVolatility(signal.PairName);
            var config = pairConfigs[signal.PairName];
            
            if (recentVol > config.VolFilterThreshold)
            {
                riskViolations.Add($"High volatility filter triggered: {recentVol:F4}");
            }
            
            // === MAX POSITIONS ===
            if (activePairs.Count >= 4) // Max 4 concurrent pairs
            {
                riskViolations.Add("Maximum concurrent positions reached");
            }
            
            bool approved = riskViolations.Count == 0 && !isRiskOff;
            
            if (!approved)
            {
                Print($"⚠️ Signal rejected for {signal.PairName}: {string.Join(", ", riskViolations)}");
            }
            
            return approved;
        }
        
        private double CalculateRealizedVolatility(string pairName)
        {
            var config = pairConfigs[pairName];
            var pricesA = priceHistory[config.SymbolA];
            
            if (pricesA.Count < 21)
                return 0.0;
            
            var recent = pricesA.Skip(pricesA.Count - 20).ToArray();
            var returns = new double[recent.Length - 1];
            
            for (int i = 1; i < recent.Length; i++)
            {
                returns[i - 1] = Math.Log(recent[i] / recent[i - 1]);
            }
            
            var mean = returns.Average();
            var variance = returns.Select(r => Math.Pow(r - mean, 2)).Average();
            
            return Math.Sqrt(variance * 1440); // Annualized (1440 minutes per day)
        }
        
        private bool IsWithinTradingHours()
        {
            var currentHour = DateTime.Now.Hour;
            return currentHour >= TradingStartHour && currentHour <= TradingEndHour;
        }
        
        private bool IsNewsBlackout()
        {
            if (!EnableNewsBlackout)
                return false;
            
            var now = DateTime.Now;
            
            foreach (var newsTime in newsBlackoutTimes)
            {
                var timeDiff = Math.Abs((now - newsTime).TotalMinutes);
                if (timeDiff <= 5) // 5-minute blackout window
                {
                    return true;
                }
            }
            
            return false;
        }
        
        #endregion
        
        #region Execution
        
        private void ExecutePairSignal(PairSignal signal)
        {
            try
            {
                var config = pairConfigs[signal.PairName];
                
                Print($"🎯 Executing {signal.SignalType} for {signal.PairName}");
                Print($"📊 Z-Score: {signal.ZScore:F2}, Confidence: {signal.Confidence:F2}");
                
                if (signal.SignalType == "LONG_PAIR")
                {
                    ExecuteLongPair(signal, config);
                }
                else if (signal.SignalType == "SHORT_PAIR")
                {
                    ExecuteShortPair(signal, config);
                }
                else if (signal.SignalType == "EXIT_PAIR")
                {
                    ExecuteExitPair(signal, config);
                }
                
                tradesExecuted++;
                
            }
            catch (Exception ex)
            {
                Print($"❌ Error executing signal for {signal.PairName}: {ex.Message}");
            }
        }
        
        private void ExecuteLongPair(PairSignal signal, PairConfig config)
        {
            // === LONG PAIR: Long B, Short A ===
            var qtyA = PositionSize;
            var qtyB = Math.Max(1, (int)Math.Round(PositionSize * Math.Abs(signal.HedgeRatio)));
            
            // Short A
            EnterShort(qtyA, $"{signal.PairName}_A");
            
            // Long B  
            EnterLong(qtyB, $"{signal.PairName}_B");
            
            // === TRACK POSITION ===
            activePairs[signal.PairName] = new PairPosition
            {
                PairName = signal.PairName,
                SignalType = signal.SignalType,
                EntryTime = Time[0],
                ZScoreAtEntry = signal.ZScore,
                QtyA = -qtyA, // Negative for short
                QtyB = qtyB,
                EntryPriceA = currentPrices[config.SymbolA],
                EntryPriceB = currentPrices[config.SymbolB]
            };
            
            Print($"✅ Long pair executed: Short {qtyA} {config.SymbolA}, Long {qtyB} {config.SymbolB}");
        }
        
        private void ExecuteShortPair(PairSignal signal, PairConfig config)
        {
            // === SHORT PAIR: Short B, Long A ===
            var qtyA = PositionSize;
            var qtyB = Math.Max(1, (int)Math.Round(PositionSize * Math.Abs(signal.HedgeRatio)));
            
            // Long A
            EnterLong(qtyA, $"{signal.PairName}_A");
            
            // Short B
            EnterShort(qtyB, $"{signal.PairName}_B");
            
            // === TRACK POSITION ===
            activePairs[signal.PairName] = new PairPosition
            {
                PairName = signal.PairName,
                SignalType = signal.SignalType,
                EntryTime = Time[0],
                ZScoreAtEntry = signal.ZScore,
                QtyA = qtyA,
                QtyB = -qtyB, // Negative for short
                EntryPriceA = currentPrices[config.SymbolA],
                EntryPriceB = currentPrices[config.SymbolB]
            };
            
            Print($"✅ Short pair executed: Long {qtyA} {config.SymbolA}, Short {qtyB} {config.SymbolB}");
        }
        
        private void ExecuteExitPair(PairSignal signal, PairConfig config)
        {
            if (!activePairs.ContainsKey(signal.PairName))
                return;
            
            // === FLATTEN ALL POSITIONS FOR THIS PAIR ===
            ExitLong($"{signal.PairName}_A");
            ExitShort($"{signal.PairName}_A");
            ExitLong($"{signal.PairName}_B");
            ExitShort($"{signal.PairName}_B");
            
            var position = activePairs[signal.PairName];
            var holdingTime = Time[0] - position.EntryTime;
            
            Print($"🔄 Pair position closed: {signal.PairName} after {holdingTime.TotalMinutes:F0} minutes");
            
            activePairs.Remove(signal.PairName);
        }
        
        #endregion
        
        #region Utility Methods
        
        private void UpdatePriceData()
        {
            // === UPDATE CURRENT PRICES ===
            if (BarsInProgress == 1) // ES
            {
                currentPrices["ES"] = Close[0];
                priceHistory["ES"].Add(Close[0]);
                TrimPriceHistory("ES");
            }
            else if (BarsInProgress == 2) // NQ
            {
                currentPrices["NQ"] = Close[0];
                priceHistory["NQ"].Add(Close[0]);
                TrimPriceHistory("NQ");
            }
            else if (BarsInProgress == 3) // ZN
            {
                currentPrices["ZN"] = Close[0];
                priceHistory["ZN"].Add(Close[0]);
                TrimPriceHistory("ZN");
            }
            else if (BarsInProgress == 4) // ZB
            {
                currentPrices["ZB"] = Close[0];
                priceHistory["ZB"].Add(Close[0]);
                TrimPriceHistory("ZB");
            }
        }
        
        private void TrimPriceHistory(string symbol)
        {
            var maxHistory = 2000; // Keep 2000 bars for memory management
            if (priceHistory[symbol].Count > maxHistory)
            {
                priceHistory[symbol] = priceHistory[symbol].Skip(priceHistory[symbol].Count - maxHistory).ToList();
            }
        }
        
        private bool HasSufficientData(PairConfig config)
        {
            return priceHistory.ContainsKey(config.SymbolA) &&
                   priceHistory.ContainsKey(config.SymbolB) &&
                   priceHistory[config.SymbolA].Count >= config.Lookback &&
                   priceHistory[config.SymbolB].Count >= config.Lookback;
        }
        
        private void CheckDailyReset()
        {
            var today = DateTime.Now.Date;
            if (today > lastResetTime)
            {
                // === RESET DAILY METRICS ===
                dailyPnL = 0.0;
                dailyHighWater = Account.Get(AccountItem.NetLiquidation, Currency.UsDollar);
                isRiskOff = false;
                riskViolations.Clear();
                lastResetTime = today;
                
                Print($"🔄 Daily metrics reset for {today:yyyy-MM-dd}");
            }
        }
        
        private void MonitorActivePairs()
        {
            var positionsToClose = new List<string>();
            
            foreach (var pair in activePairs.Values)
            {
                var config = pairConfigs[pair.PairName];
                var holdingTime = Time[0] - pair.EntryTime;
                
                // === TIME-BASED EXIT ===
                if (holdingTime.TotalMinutes > config.MaxHoldingBars)
                {
                    positionsToClose.Add(pair.PairName);
                    Print($"⏰ Time-based exit for {pair.PairName} after {holdingTime.TotalMinutes:F0} minutes");
                }
            }
            
            // === EXECUTE TIME-BASED EXITS ===
            foreach (var pairName in positionsToClose)
            {
                var signal = new PairSignal { PairName = pairName, SignalType = "EXIT_PAIR" };
                ExecutePairSignal(signal);
            }
        }
        
        #endregion
        
        #region Data Classes
        
        public class PairConfig
        {
            public string SymbolA { get; set; }
            public string SymbolB { get; set; }
            public int BarsInProgressA { get; set; }
            public int BarsInProgressB { get; set; }
            public int Lookback { get; set; }
            public double EntryZScore { get; set; }
            public double ExitZScore { get; set; }
            public int MaxHoldingBars { get; set; }
            public double VolFilterThreshold { get; set; }
        }
        
        public class PairSignal
        {
            public string PairName { get; set; }
            public string SignalType { get; set; }
            public double ZScore { get; set; }
            public double HedgeRatio { get; set; }
            public double Confidence { get; set; }
            public DateTime Timestamp { get; set; }
        }
        
        public class PairPosition
        {
            public string PairName { get; set; }
            public string SignalType { get; set; }
            public DateTime EntryTime { get; set; }
            public double ZScoreAtEntry { get; set; }
            public int QtyA { get; set; }
            public int QtyB { get; set; }
            public double EntryPriceA { get; set; }
            public double EntryPriceB { get; set; }
        }
        
        public class RollingStats
        {
            private Queue<double> values = new Queue<double>();
            private int maxSize = 300;
            
            public int Count => values.Count;
            
            public double Mean => values.Count > 0 ? values.Average() : 0.0;
            
            public double StandardDeviation
            {
                get
                {
                    if (values.Count < 2) return 0.0;
                    var mean = Mean;
                    var variance = values.Select(v => Math.Pow(v - mean, 2)).Average();
                    return Math.Sqrt(variance);
                }
            }
            
            public void AddValue(double value)
            {
                values.Enqueue(value);
                while (values.Count > maxSize)
                {
                    values.Dequeue();
                }
            }
        }
        
        #endregion
    }
}