#region Using declarations
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
using System.IO;
using System.Collections.ObjectModel;
using System.Net.Sockets;
using System.Net;
using System.Threading;
using System.Globalization;
using System.Diagnostics;
using Newtonsoft.Json;
#endregion

//This namespace holds Add ons in this folder and is required. Do not change it. 
namespace NinjaTrader.NinjaScript.AddOns
{
    /// <summary>
    /// ES ML Trading System - Professional institutional-grade trading AddOn
    /// Follows official NinjaTrader 8 documentation patterns and best practices
    /// Real-time market data integration with ML signal generation
    /// </summary>
    public class ESMLTradingSystem : AddOnBase
    {
        #region Variables
        // Core system components
        private ESMLTradingWindow tradingWindow;
        private Account tradingAccount;
        private Instrument tradingInstrument;
        private MasterInstrument esInstrument;
        private Bars esMarketData;
        
        // Data subscriptions - Following official NT8 patterns
        private bool isSubscribedToMarketData;
        private bool isSystemActive;
        
        // Risk management and trading state
        private int currentPosition;
        private double dailyPnL;
        private int dailyTradeCount;
        private List<ESTradeRecord> tradeHistory;
        private ESPerformanceMetrics performanceMetrics;
        
        // ML Integration - Python model integration
        private Process pythonMLProcess;
        private bool isMLModelAvailable;
        private string pythonExecutablePath;
        private string mlModelPath;
        
        // Threading and timers - Following official NT8 threading model
        private System.Windows.Threading.DispatcherTimer systemTimer;
        private readonly object tradingLock = new object();
        
        // Configuration and settings
        private const int MAX_POSITION_SIZE = 2;
        private const int MAX_DAILY_TRADES = 5;
        private const double MIN_SIGNAL_CONFIDENCE = 0.70;
        
        // Logging and error handling - Following official NT8 patterns
        private string logFilePath;
        private TextWriter logWriter;
        
        // Menu integration - Following official NT8 menu patterns
        private NTMenuItem esMLMenuItem;
        private NTMenuItem separatorMenuItem;
        #endregion

        #region OnStateChange - Following Official NT8 Documentation
        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                // Required AddOn properties - Following official NT8 AddOn framework
                Description = @"ES ML Trading System - Professional institutional-grade trading system with real-time ML signal generation and risk management";
                Name = "ESMLTradingSystem";
                
                // Initialize core components
                InitializeSystemDefaults();
                
                // Setup logging - Following official NT8 logging patterns
                SetupLogging();
            }
            else if (State == State.Active)
            {
                // Following official NT8 Active state pattern
                try
                {
                    // Initialize trading components
                    InitializeTradingComponents();
                    
                    // Setup market data subscriptions
                    SetupMarketDataSubscription();
                    
                    // Add menu items - Following official NT8 menu integration
                    AddMenuItems();
                    
                    // Start system timer
                    StartSystemTimer();
                    
                    LogMessage("ES ML Trading System activated successfully");
                }
                catch (Exception ex)
                {
                    // Following official NT8 error handling patterns
                    LogError("Error during system activation", ex);
                    Print($"ES ML Trading System activation failed: {ex.Message}");
                }
            }
            else if (State == State.Terminated)
            {
                // Following official NT8 cleanup patterns
                CleanupSystem();
                LogMessage("ES ML Trading System terminated");
            }
        }
        #endregion

        #region Initialization Methods - Following Official NT8 Patterns
        private void InitializeSystemDefaults()
        {
            // Initialize collections and state
            tradeHistory = new List<ESTradeRecord>();
            performanceMetrics = new ESPerformanceMetrics();
            currentPosition = 0;
            dailyPnL = 0.0;
            dailyTradeCount = 0;
            isSystemActive = false;
            isSubscribedToMarketData = false;
            
            // Initialize ML components
            InitializeMLComponents();
        }

        private void InitializeMLComponents()
        {
            try
            {
                // Set path to Python executable in virtual environment
                var desktopPath = Environment.GetFolderPath(Environment.SpecialFolder.Desktop);
                pythonExecutablePath = Path.Combine(desktopPath, "InstitutionalMLTrading", "venv", "Scripts", "python.exe");
                
                // Set path to ML model
                mlModelPath = Path.Combine(desktopPath, "InstitutionalMLTrading", "models", "es_ml_model.joblib");
                
                // Check if ML components are available
                isMLModelAvailable = File.Exists(pythonExecutablePath) && File.Exists(mlModelPath);
                
                if (isMLModelAvailable)
                {
                    LogMessage("✅ ML model found - Enhanced ML signals enabled");
                }
                else
                {
                    LogMessage("⚠️ ML model not found - Using technical analysis mode");
                    LogMessage($"Python path: {pythonExecutablePath}");
                    LogMessage($"Model path: {mlModelPath}");
                }
            }
            catch (Exception ex)
            {
                LogError("Error initializing ML components", ex);
                isMLModelAvailable = false;
            }
        }

        private void SetupLogging()
        {
            try
            {
                // Following official NT8 logging location patterns
                var logDirectory = Path.Combine(
                    Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments),
                    "NinjaTrader 8", "logs"
                );
                
                if (!Directory.Exists(logDirectory))
                    Directory.CreateDirectory(logDirectory);
                
                logFilePath = Path.Combine(logDirectory, $"ESMLTradingSystem_{DateTime.Now:yyyyMMdd}.log");
                logWriter = new StreamWriter(logFilePath, true);
                
                LogMessage("ES ML Trading System logging initialized");
            }
            catch (Exception ex)
            {
                Print($"Failed to initialize logging: {ex.Message}");
            }
        }

        private void InitializeTradingComponents()
        {
            // Find ES futures instrument - Following official NT8 instrument lookup
            esInstrument = MasterInstrument.All.FirstOrDefault(mi => mi.Name == "ES");
            if (esInstrument == null)
            {
                throw new InvalidOperationException("ES futures instrument not found. Please ensure ES is available in your instrument list.");
            }
            
            // Get current ES contract - Following official NT8 contract selection
            tradingInstrument = esInstrument.GetInstrument(Cbi.Instrument.GetClosestExpiryDate(esInstrument));
            if (tradingInstrument == null)
            {
                throw new InvalidOperationException("Unable to find current ES contract.");
            }
            
            // Find simulation account - Following official NT8 account selection
            tradingAccount = Account.All.FirstOrDefault(a => 
                a.Name.Contains("Sim") || a.Name.Contains("sim") || a.Name.Contains("Simulation"));
            
            if (tradingAccount == null)
            {
                LogMessage("Warning: No simulation account found. Using first available account.");
                tradingAccount = Account.All.FirstOrDefault();
            }
            
            if (tradingAccount == null)
            {
                throw new InvalidOperationException("No trading account available.");
            }
            
            LogMessage($"Initialized trading components - Instrument: {tradingInstrument.MasterInstrument.Name}, Account: {tradingAccount.Name}");
        }

        private void SetupMarketDataSubscription()
        {
            try
            {
                // Following official NT8 market data subscription patterns
                if (tradingInstrument != null && !isSubscribedToMarketData)
                {
                    // Subscribe to 1-minute bars for real-time data
                    esMarketData = Data.GetBars(tradingInstrument, new BarsPeriod 
                    { 
                        BarsPeriodType = BarsPeriodType.Minute, 
                        Value = 1 
                    }, 500); // Keep 500 bars in memory
                    
                    if (esMarketData != null)
                    {
                        // Following official NT8 event subscription patterns
                        esMarketData.BarsCallback = OnBarsUpdate;
                        isSubscribedToMarketData = true;
                        LogMessage($"Subscribed to market data for {tradingInstrument.FullName}");
                    }
                }
            }
            catch (Exception ex)
            {
                LogError("Failed to setup market data subscription", ex);
                throw;
            }
        }

        private void AddMenuItems()
        {
            try
            {
                // Following official NT8 menu integration patterns from documentation
                if (NTWindow.GetWindow(typeof(ControlCenter)) is ControlCenter controlCenter)
                {
                    // Add separator
                    separatorMenuItem = new NTMenuItem { Style = (Style)Application.Current.TryFindResource("MainMenuSeparator") };
                    controlCenter.mnuTools.Items.Add(separatorMenuItem);
                    
                    // Add main menu item
                    esMLMenuItem = new NTMenuItem 
                    { 
                        Header = "ES ML Trading System",
                        Style = (Style)Application.Current.TryFindResource("MainMenuItem")
                    };
                    esMLMenuItem.Click += OnMenuItemClick;
                    controlCenter.mnuTools.Items.Add(esMLMenuItem);
                    
                    LogMessage("Menu items added successfully");
                }
            }
            catch (Exception ex)
            {
                LogError("Failed to add menu items", ex);
            }
        }

        private void StartSystemTimer()
        {
            // Following official NT8 timer patterns
            systemTimer = new System.Windows.Threading.DispatcherTimer();
            systemTimer.Interval = TimeSpan.FromSeconds(1);
            systemTimer.Tick += OnSystemTimerTick;
            systemTimer.Start();
        }
        #endregion

        #region Market Data Event Handlers - Following Official NT8 Patterns
        private void OnBarsUpdate(Bars bars, BarUpdateEventArgs e)
        {
            try
            {
                // Following official NT8 bars update patterns
                if (bars == null || bars.Count < 2 || !isSystemActive)
                    return;
                
                // Thread-safe data access - Following official NT8 threading guidelines
                lock (tradingLock)
                {
                    // Update trading window with latest market data
                    if (tradingWindow != null)
                    {
                        // Following official NT8 dispatcher patterns for UI updates
                        Application.Current?.Dispatcher.InvokeAsync(() =>
                        {
                            tradingWindow.UpdateMarketData(bars);
                        });
                    }
                    
                    // Generate ML signals with latest data
                    ProcessMLSignals(bars);
                }
            }
            catch (Exception ex)
            {
                LogError("Error in OnBarsUpdate", ex);
            }
        }

        private void ProcessMLSignals(Bars bars)
        {
            try
            {
                if (bars.Count < 50) // Need sufficient data for ML signals
                    return;
                
                // Generate ML trading signals - Following institutional ML patterns
                var signalResult = GenerateMLSignals(bars);
                
                // Update UI with signals
                if (tradingWindow != null && signalResult != null)
                {
                    Application.Current?.Dispatcher.InvokeAsync(() =>
                    {
                        tradingWindow.UpdateSignals(signalResult);
                    });
                }
                
                // Execute trades based on signals if system is active
                if (isSystemActive && signalResult != null && 
                    signalResult.Confidence >= MIN_SIGNAL_CONFIDENCE &&
                    dailyTradeCount < MAX_DAILY_TRADES)
                {
                    ExecuteMLSignal(signalResult);
                }
            }
            catch (Exception ex)
            {
                LogError("Error processing ML signals", ex);
            }
        }

        private ESSignalResult GenerateMLSignals(Bars bars)
        {
            try
            {
                var signals = new Dictionary<string, string>();
                var confidence = 0.0;
                string consensusSignal = "HOLD";
                
                // If ML model is available, get ML prediction first
                if (isMLModelAvailable)
                {
                    var mlResult = GetMLPrediction(bars);
                    if (mlResult != null)
                    {
                        signals["ML"] = mlResult.Signal;
                        confidence = mlResult.Confidence;
                        consensusSignal = mlResult.Signal;
                        
                        LogMessage($"🤖 ML Signal: {mlResult.Signal} (Confidence: {mlResult.Confidence:P1})");
                    }
                }
                
                // Generate technical analysis signals as backup/confirmation
                var smaSignal = CalculateSMASignal(bars);
                signals["SMA"] = smaSignal;
                
                var rsiSignal = CalculateRSISignal(bars);
                signals["RSI"] = rsiSignal;
                
                var bbSignal = CalculateBollingerSignal(bars);
                signals["Bollinger"] = bbSignal;
                
                var momentumSignal = CalculateMomentumSignal(bars);
                signals["Momentum"] = momentumSignal;
                
                // If no ML model, use technical analysis consensus
                if (!isMLModelAvailable)
                {
                    var buySignals = signals.Values.Count(s => s == "BUY");
                    var sellSignals = signals.Values.Count(s => s == "SELL");
                    var totalSignals = signals.Count;
                    
                    if (buySignals > sellSignals)
                    {
                        consensusSignal = "BUY";
                        confidence = (double)buySignals / totalSignals;
                    }
                    else if (sellSignals > buySignals)
                    {
                        consensusSignal = "SELL";
                        confidence = (double)sellSignals / totalSignals;
                    }
                    else
                    {
                        consensusSignal = "HOLD";
                        confidence = 0.5;
                    }
                }
                
                return new ESSignalResult
                {
                    Signal = consensusSignal,
                    Confidence = confidence,
                    IndividualSignals = signals,
                    Timestamp = DateTime.Now,
                    IsMLGenerated = isMLModelAvailable
                };
            }
            catch (Exception ex)
            {
                LogError("Error generating ML signals", ex);
                return null;
            }
        }

        private ESSignalResult GetMLPrediction(Bars bars)
        {
            try
            {
                if (bars.Count < 50)
                    return null;
                
                // Prepare market data for ML model
                var marketData = new
                {
                    timestamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss"),
                    open = bars.GetOpen(bars.Count - 1),
                    high = bars.GetHigh(bars.Count - 1),
                    low = bars.GetLow(bars.Count - 1),
                    close = bars.GetClose(bars.Count - 1),
                    volume = bars.GetVolume(bars.Count - 1),
                    // Historical data for feature engineering
                    historical_closes = Enumerable.Range(0, Math.Min(50, bars.Count))
                        .Select(i => bars.GetClose(bars.Count - 1 - i))
                        .Reverse()
                        .ToArray(),
                    historical_volumes = Enumerable.Range(0, Math.Min(50, bars.Count))
                        .Select(i => bars.GetVolume(bars.Count - 1 - i))
                        .Reverse()
                        .ToArray()
                };
                
                // Serialize to JSON for Python script
                var jsonData = Newtonsoft.Json.JsonConvert.SerializeObject(marketData);
                var tempFile = Path.GetTempFileName();
                File.WriteAllText(tempFile, jsonData);
                
                // Call Python ML prediction script
                var pythonScript = Path.Combine(
                    Environment.GetFolderPath(Environment.SpecialFolder.Desktop),
                    "InstitutionalMLTrading", "ml-models", "integration", "predict_signal.py"
                );
                
                if (!File.Exists(pythonScript))
                {
                    LogMessage("⚠️ ML prediction script not found, falling back to technical analysis");
                    return null;
                }
                
                var startInfo = new ProcessStartInfo
                {
                    FileName = pythonExecutablePath,
                    Arguments = $"\"{pythonScript}\" \"{tempFile}\"",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true,
                    WindowStyle = ProcessWindowStyle.Hidden
                };
                
                using (var process = new Process { StartInfo = startInfo })
                {
                    process.Start();
                    var output = process.StandardOutput.ReadToEnd();
                    var error = process.StandardError.ReadToEnd();
                    process.WaitForExit(5000); // 5 second timeout
                    
                    // Clean up temp file
                    try { File.Delete(tempFile); } catch { }
                    
                    if (process.ExitCode == 0 && !string.IsNullOrEmpty(output))
                    {
                        // Parse ML prediction result
                        var result = Newtonsoft.Json.JsonConvert.DeserializeObject<dynamic>(output);
                        
                        return new ESSignalResult
                        {
                            Signal = result.signal.ToString().ToUpper(),
                            Confidence = (double)result.confidence,
                            IndividualSignals = new Dictionary<string, string> { { "ML", result.signal.ToString().ToUpper() } },
                            Timestamp = DateTime.Now,
                            IsMLGenerated = true
                        };
                    }
                    else
                    {
                        if (!string.IsNullOrEmpty(error))
                            LogMessage($"ML prediction error: {error}");
                        return null;
                    }
                }
            }
            catch (Exception ex)
            {
                LogError("Error getting ML prediction", ex);
                return null;
            }
        }
                
                // RSI Signal  
                var rsiSignal = CalculateRSISignal(bars);
                signals["RSI"] = rsiSignal;
                
                // Bollinger Bands Signal
                var bbSignal = CalculateBollingerSignal(bars);
                signals["Bollinger"] = bbSignal;
                
                // Price Momentum Signal
                var momentumSignal = CalculateMomentumSignal(bars);
                signals["Momentum"] = momentumSignal;
                
                // Calculate consensus and confidence
                var buySignals = signals.Values.Count(s => s == "BUY");
                var sellSignals = signals.Values.Count(s => s == "SELL");
                var totalSignals = signals.Count;
                
                string consensusSignal;
                if (buySignals > sellSignals)
                {
                    consensusSignal = "BUY";
                    confidence = (double)buySignals / totalSignals;
                }
                else if (sellSignals > buySignals)
                {
                    consensusSignal = "SELL";
                    confidence = (double)sellSignals / totalSignals;
                }
                else
                {
                    consensusSignal = "HOLD";
                    confidence = 0.5;
                }
                
                return new ESSignalResult
                {
                    Signal = consensusSignal,
                    Confidence = confidence,
                    IndividualSignals = signals,
                    Timestamp = DateTime.Now
                };
            }
            catch (Exception ex)
            {
                LogError("Error generating ML signals", ex);
                return null;
            }
        }

        // Technical Analysis Signal Calculations
        private string CalculateSMASignal(Bars bars)
        {
            try
            {
                if (bars.Count < 50) return "HOLD";
                
                // 20-period and 50-period SMAs
                var sma20 = CalculateSMA(bars, 20);
                var sma50 = CalculateSMA(bars, 50);
                
                if (sma20 > sma50 && bars.GetClose(bars.Count - 1) > sma20)
                    return "BUY";
                else if (sma20 < sma50 && bars.GetClose(bars.Count - 1) < sma20)
                    return "SELL";
                else
                    return "HOLD";
            }
            catch
            {
                return "HOLD";
            }
        }

        private string CalculateRSISignal(Bars bars)
        {
            try
            {
                if (bars.Count < 14) return "HOLD";
                
                var rsi = CalculateRSI(bars, 14);
                
                if (rsi < 30) return "BUY";   // Oversold
                else if (rsi > 70) return "SELL"; // Overbought
                else return "HOLD";
            }
            catch
            {
                return "HOLD";
            }
        }

        private string CalculateBollingerSignal(Bars bars)
        {
            try
            {
                if (bars.Count < 20) return "HOLD";
                
                var currentPrice = bars.GetClose(bars.Count - 1);
                var sma20 = CalculateSMA(bars, 20);
                var stdDev = CalculateStandardDeviation(bars, 20);
                
                var upperBand = sma20 + (2 * stdDev);
                var lowerBand = sma20 - (2 * stdDev);
                
                if (currentPrice <= lowerBand) return "BUY";   // Price at lower band
                else if (currentPrice >= upperBand) return "SELL"; // Price at upper band
                else return "HOLD";
            }
            catch
            {
                return "HOLD";
            }
        }

        private string CalculateMomentumSignal(Bars bars)
        {
            try
            {
                if (bars.Count < 10) return "HOLD";
                
                var currentPrice = bars.GetClose(bars.Count - 1);
                var priceNPeriodsAgo = bars.GetClose(bars.Count - 10);
                var momentum = (currentPrice - priceNPeriodsAgo) / priceNPeriodsAgo;
                
                if (momentum > 0.001) return "BUY";   // Positive momentum > 0.1%
                else if (momentum < -0.001) return "SELL"; // Negative momentum < -0.1%
                else return "HOLD";
            }
            catch
            {
                return "HOLD";
            }
        }

        // Technical Analysis Helper Methods
        private double CalculateSMA(Bars bars, int period)
        {
            if (bars.Count < period) return 0;
            
            double sum = 0;
            for (int i = 0; i < period; i++)
            {
                sum += bars.GetClose(bars.Count - 1 - i);
            }
            return sum / period;
        }

        private double CalculateRSI(Bars bars, int period)
        {
            if (bars.Count < period + 1) return 50;
            
            double gains = 0, losses = 0;
            
            for (int i = 1; i <= period; i++)
            {
                double change = bars.GetClose(bars.Count - i) - bars.GetClose(bars.Count - i - 1);
                if (change > 0) gains += change;
                else losses += Math.Abs(change);
            }
            
            if (losses == 0) return 100;
            
            double avgGain = gains / period;
            double avgLoss = losses / period;
            double rs = avgGain / avgLoss;
            
            return 100 - (100 / (1 + rs));
        }

        private double CalculateStandardDeviation(Bars bars, int period)
        {
            if (bars.Count < period) return 0;
            
            double sma = CalculateSMA(bars, period);
            double sumSquaredDiff = 0;
            
            for (int i = 0; i < period; i++)
            {
                double diff = bars.GetClose(bars.Count - 1 - i) - sma;
                sumSquaredDiff += diff * diff;
            }
            
            return Math.Sqrt(sumSquaredDiff / period);
        }
        #endregion

        #region Order Execution - Following Official NT8 Trading Patterns
        private void ExecuteMLSignal(ESSignalResult signal)
        {
            try
            {
                lock (tradingLock)
                {
                    // Risk management checks
                    if (!ValidateTradeRisk(signal))
                        return;
                    
                    // Determine order quantity and direction
                    int quantity = CalculateOrderQuantity();
                    OrderAction action = signal.Signal == "BUY" ? OrderAction.Buy : OrderAction.Sell;
                    
                    // Place market order - Following official NT8 order submission patterns
                    Order order = tradingAccount.CreateOrder(
                        tradingInstrument,
                        action,
                        OrderType.Market,
                        OrderEntry.Manual,
                        TimeInForce.Day,
                        quantity,
                        0, // Limit price (not used for market orders)
                        0, // Stop price (not used for market orders) 
                        "",
                        $"ES ML Signal - {signal.Signal}",
                        null
                    );
                    
                    if (order != null)
                    {
                        // Following official NT8 order submission
                        tradingAccount.Submit(new[] { order });
                        
                        // Update trade tracking
                        dailyTradeCount++;
                        
                        // Log trade execution
                        LogMessage($"Order submitted: {action} {quantity} {tradingInstrument.MasterInstrument.Name} @ Market (Confidence: {signal.Confidence:P1})");
                        
                        // Update UI
                        if (tradingWindow != null)
                        {
                            Application.Current?.Dispatcher.InvokeAsync(() =>
                            {
                                tradingWindow.LogMessage($"Order placed: {action} {quantity} contracts (Confidence: {signal.Confidence:P1})");
                            });
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                LogError("Error executing ML signal", ex);
            }
        }

        private bool ValidateTradeRisk(ESSignalResult signal)
        {
            // Check daily trade limit
            if (dailyTradeCount >= MAX_DAILY_TRADES)
            {
                LogMessage($"Daily trade limit reached ({MAX_DAILY_TRADES})");
                return false;
            }
            
            // Check signal confidence
            if (signal.Confidence < MIN_SIGNAL_CONFIDENCE)
            {
                LogMessage($"Signal confidence too low: {signal.Confidence:P1} < {MIN_SIGNAL_CONFIDENCE:P1}");
                return false;
            }
            
            // Check position size limits
            int proposedPosition = currentPosition + (signal.Signal == "BUY" ? 1 : signal.Signal == "SELL" ? -1 : 0);
            if (Math.Abs(proposedPosition) > MAX_POSITION_SIZE)
            {
                LogMessage($"Position size limit exceeded. Current: {currentPosition}, Proposed: {proposedPosition}");
                return false;
            }
            
            return true;
        }

        private int CalculateOrderQuantity()
        {
            // For ES futures, typically trade 1 contract at a time for risk management
            return 1;
        }
        #endregion

        #region Event Handlers - Following Official NT8 Patterns
        private void OnMenuItemClick(object sender, RoutedEventArgs e)
        {
            // Following official NT8 window management patterns
            if (tradingWindow == null)
            {
                tradingWindow = new ESMLTradingWindow(this);
                tradingWindow.Closed += OnTradingWindowClosed;
            }
            
            if (tradingWindow.WindowState == WindowState.Minimized)
                tradingWindow.WindowState = WindowState.Normal;
            
            tradingWindow.Show();
            tradingWindow.Activate();
        }

        private void OnTradingWindowClosed(object sender, EventArgs e)
        {
            tradingWindow = null;
        }

        private void OnSystemTimerTick(object sender, EventArgs e)
        {
            try
            {
                // Update performance metrics
                UpdatePerformanceMetrics();
                
                // Update account information
                if (tradingWindow != null && tradingAccount != null)
                {
                    Application.Current?.Dispatcher.InvokeAsync(() =>
                    {
                        tradingWindow.UpdateAccountInfo(tradingAccount);
                    });
                }
            }
            catch (Exception ex)
            {
                LogError("Error in system timer tick", ex);
            }
        }
        #endregion

        #region Public Methods - Interface for Trading Window
        public void StartSystem()
        {
            lock (tradingLock)
            {
                isSystemActive = true;
                LogMessage("Trading system started - Active signal processing enabled");
            }
        }

        public void StopSystem()
        {
            lock (tradingLock)
            {
                isSystemActive = false;
                LogMessage("Trading system stopped - Signal processing disabled");
            }
        }

        public bool IsSystemActive => isSystemActive;
        public Account TradingAccount => tradingAccount;
        public Instrument TradingInstrument => tradingInstrument;
        #endregion

        #region Performance and Logging - Following Official NT8 Patterns
        private void UpdatePerformanceMetrics()
        {
            try
            {
                if (tradingAccount == null) return;
                
                // Update performance metrics with current account data
                performanceMetrics.TotalPnL = tradingAccount.Get(AccountItem.RealizedProfitLoss, Currency.UsDollar);
                performanceMetrics.DailyPnL = dailyPnL; // Track separately for daily reset
                performanceMetrics.TotalTrades = tradeHistory.Count;
                
                // Calculate win rate
                if (tradeHistory.Count > 0)
                {
                    var winningTrades = tradeHistory.Count(t => t.RealizedPnL > 0);
                    performanceMetrics.WinRate = (double)winningTrades / tradeHistory.Count;
                }
                
                // Update trading window
                if (tradingWindow != null)
                {
                    Application.Current?.Dispatcher.InvokeAsync(() =>
                    {
                        tradingWindow.UpdatePerformanceMetrics(performanceMetrics);
                    });
                }
            }
            catch (Exception ex)
            {
                LogError("Error updating performance metrics", ex);
            }
        }

        private void LogMessage(string message)
        {
            try
            {
                var logEntry = $"[{DateTime.Now:yyyy-MM-dd HH:mm:ss}] {message}";
                
                // Console output
                Print(logEntry);
                
                // File logging
                logWriter?.WriteLine(logEntry);
                logWriter?.Flush();
            }
            catch (Exception ex)
            {
                Print($"Logging error: {ex.Message}");
            }
        }

        private void LogError(string message, Exception ex)
        {
            var errorMessage = $"ERROR: {message} - {ex.Message}";
            if (ex.InnerException != null)
                errorMessage += $" (Inner: {ex.InnerException.Message})";
            
            LogMessage(errorMessage);
        }
        #endregion

        #region Cleanup - Following Official NT8 Cleanup Patterns
        private void CleanupSystem()
        {
            try
            {
                // Stop system timer
                systemTimer?.Stop();
                systemTimer = null;
                
                // Close trading window
                tradingWindow?.Close();
                tradingWindow = null;
                
                // Cleanup market data subscription
                if (isSubscribedToMarketData && esMarketData != null)
                {
                    esMarketData.BarsCallback = null;
                    isSubscribedToMarketData = false;
                }
                
                // Remove menu items
                if (esMLMenuItem != null && separatorMenuItem != null)
                {
                    if (NTWindow.GetWindow(typeof(ControlCenter)) is ControlCenter controlCenter)
                    {
                        controlCenter.mnuTools.Items.Remove(esMLMenuItem);
                        controlCenter.mnuTools.Items.Remove(separatorMenuItem);
                    }
                }
                
                // Close logging
                logWriter?.Close();
                logWriter?.Dispose();
                logWriter = null;
                
                LogMessage("System cleanup completed");
            }
            catch (Exception ex)
            {
                Print($"Error during cleanup: {ex.Message}");
            }
        }
        #endregion
    }

    #region Supporting Data Classes - Following Official NT8 Patterns
    /// <summary>
    /// ML Signal Result - Contains signal information and confidence
    /// </summary>
    public class ESSignalResult
    {
        public string Signal { get; set; }
        public double Confidence { get; set; }
        public Dictionary<string, string> IndividualSignals { get; set; }
        public DateTime Timestamp { get; set; }
        public bool IsMLGenerated { get; set; } = false;
    }

    /// <summary>
    /// Performance Metrics - Trading system performance tracking
    /// </summary>
    public class ESPerformanceMetrics
    {
        public double TotalPnL { get; set; }
        public double DailyPnL { get; set; }
        public int TotalTrades { get; set; }
        public double WinRate { get; set; }
        public double ProfitFactor { get; set; }
    }

    /// <summary>
    /// Trade Record - Individual trade tracking
    /// </summary>
    public class ESTradeRecord
    {
        public DateTime Timestamp { get; set; }
        public string Action { get; set; }
        public int Quantity { get; set; }
        public double Price { get; set; }
        public double RealizedPnL { get; set; }
        public string Signal { get; set; }
        public double Confidence { get; set; }
    }
    #endregion
}