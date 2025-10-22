///<summary>
/// Modern Institutional Trading AddOn - NinjaTrader 8 
/// Professional-grade institutional trading with WebSocket Rithmic + Python ML integration
/// Enhanced with circuit breakers, advanced risk management, and real-time monitoring
/// Author: Brian Cheruiyot
/// Date: October 2025
///</summary>

#region Using declarations
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using NinjaTrader.Cbi;
using NinjaTrader.Core.FloatingPoint;
using NinjaTrader.Data;
using NinjaTrader.Gui;
using NinjaTrader.Gui.Chart;
using NinjaTrader.Gui.SuperDom;
using NinjaTrader.Gui.Tools;
using NinjaTrader.NinjaScript;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Net.WebSockets;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Threading;
using System.Xml.Serialization;
using System.Xml.Linq;
#endregion

namespace NinjaTrader.NinjaScript.AddOns
{
    /// <summary>
    /// Modern Institutional Trading AddOn 
    /// </summary>
    public class ModernInstitutionalAddOn : AddOnBase
    {
        private NTMenuItem modernTradingMenuItem;
        private NTMenuItem existingMenuItemInControlCenter;

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "Enhanced institutional trading AddOn with ML, risk management, and real-time monitoring";
                Name = "Modern Institutional Trading";
            }
        }

        protected override void OnWindowCreated(Window window)
        {
            // We want to place our AddOn in the Control Center's menus
            ControlCenter cc = window as ControlCenter;
            if (cc == null)
                return;

            // Find the "New" menu in Control Center
            existingMenuItemInControlCenter = cc.FindFirst("ControlCenterMenuItemNew") as NTMenuItem;
            if (existingMenuItemInControlCenter == null)
                return;

            // Create our menu item
            modernTradingMenuItem = new NTMenuItem
            {
                Header = "Modern Institutional Trading",
                Style = Application.Current.TryFindResource("MainMenuItem") as Style
            };

            // Add our AddOn into the "New" menu
            existingMenuItemInControlCenter.Items.Add(modernTradingMenuItem);

            // Subscribe to the event for when the user presses our AddOn's menu item
            modernTradingMenuItem.Click += OnMenuItemClick;
        }

        protected override void OnWindowDestroyed(Window window)
        {
            if (modernTradingMenuItem != null && window is ControlCenter)
            {
                if (existingMenuItemInControlCenter != null && existingMenuItemInControlCenter.Items.Contains(modernTradingMenuItem))
                    existingMenuItemInControlCenter.Items.Remove(modernTradingMenuItem);

                modernTradingMenuItem.Click -= OnMenuItemClick;
                modernTradingMenuItem = null;
            }
        }

        // Open our AddOn's window when the menu item is clicked on
        private void OnMenuItemClick(object sender, RoutedEventArgs e)
        {
            Core.Globals.RandomDispatcher.BeginInvoke(new Action(() => new ModernInstitutionalWindow().Show()));
        }
    }

    /// <summary>
    /// Factory class for creating tabs and windows
    /// </summary>
    public class ModernInstitutionalWindowFactory : INTTabFactory
    {
        public NTWindow CreateParentWindow()
        {
            return new ModernInstitutionalWindow();
        }

        public NTTabPage CreateTabPage(string typeName, bool isTrue)
        {
            return new ModernInstitutionalTab();
        }
    }

    /// <summary>
    /// Main window for the Modern Institutional Trading AddOn
    /// </summary>
    public class ModernInstitutionalWindow : NTWindow, IWorkspacePersistence
    {
        public ModernInstitutionalWindow()
        {
            Caption = "Modern Institutional Trading";
            Width = 1200;
            Height = 800;

            // Create TabControl for the window
            TabControl tc = new TabControl();

            // Set tab control properties
            TabControlManager.SetIsMovable(tc, true);
            TabControlManager.SetCanAddTabs(tc, true);
            TabControlManager.SetCanRemoveTabs(tc, true);
            TabControlManager.SetFactory(tc, new ModernInstitutionalWindowFactory());

            Content = tc;

            // Add our main tab
            tc.AddNTTabPage(new ModernInstitutionalTab());

            // Set workspace options
            Loaded += (o, e) =>
            {
                if (WorkspaceOptions == null)
                    WorkspaceOptions = new WorkspaceOptions("ModernInstitutional-" + Guid.NewGuid().ToString("N"), this);
            };
        }

        public void Restore(XDocument document, XElement element)
        {
            if (MainTabControl != null)
                MainTabControl.RestoreFromXElement(element);
        }

        public void Save(XDocument document, XElement element)
        {
            if (MainTabControl != null)
                MainTabControl.SaveToXElement(element);
        }

        public WorkspaceOptions WorkspaceOptions { get; set; }
    }

    /// <summary>
    /// Main tab content for the Modern Institutional Trading AddOn
    /// </summary>
    public class ModernInstitutionalTab : NTTabPage, NinjaTrader.Gui.Tools.IInstrumentProvider
    {
        #region Core Trading System Components
        private ModernInstitutionalTradingEngine tradingEngine;
        private Grid mainGrid;
        private TextBox logOutput;
        private Button startButton;
        private Button stopButton;
        private Button emergencyStopButton;
        private Button resumeButton;
        private TextBlock statusText;
        private TextBlock pnlText;
        private TextBlock tradesText;
        private TextBlock signalsText;
        private TextBlock riskText;
        private AccountSelector accountSelector;
        private InstrumentSelector instrumentSelector;
        private Cbi.Instrument instrument;
        #endregion

        public ModernInstitutionalTab()
        {
            // Set tab name
            TabName = "Modern Institutional Trading";

            // Initialize the trading engine
            tradingEngine = new ModernInstitutionalTradingEngine();

            // Create the UI
            CreateUserInterface();

            // Wire up events
            WireUpEvents();
        }

        private void CreateUserInterface()
        {
            // Main grid layout
            mainGrid = new Grid();
            Content = mainGrid;

            // Define rows and columns
            mainGrid.RowDefinitions.Add(new RowDefinition { Height = new GridLength(1, GridUnitType.Auto) }); // Controls
            mainGrid.RowDefinitions.Add(new RowDefinition { Height = new GridLength(1, GridUnitType.Auto) }); // Status
            mainGrid.RowDefinitions.Add(new RowDefinition { Height = new GridLength(1, GridUnitType.Star) }); // Log output

            mainGrid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });

            // Create control panel
            CreateControlPanel();

            // Create status panel
            CreateStatusPanel();

            // Create log output
            CreateLogPanel();
        }

        private void CreateControlPanel()
        {
            var controlPanel = new WrapPanel();
            controlPanel.Orientation = Orientation.Horizontal;
            controlPanel.Margin = new Thickness(10, 10, 10, 10);
            Grid.SetRow(controlPanel, 0);
            mainGrid.Children.Add(controlPanel);

            // Account selector
            var accountLabel = new Label { Content = "Account:", VerticalAlignment = VerticalAlignment.Center };
            accountSelector = new AccountSelector { Width = 150, Margin = new Thickness(5, 5, 5, 5) };
            controlPanel.Children.Add(accountLabel);
            controlPanel.Children.Add(accountSelector);

            // Instrument selector
            var instrumentLabel = new Label { Content = "Instrument:", VerticalAlignment = VerticalAlignment.Center };
            instrumentSelector = new InstrumentSelector { Width = 150, Margin = new Thickness(5, 5, 5, 5) };
            controlPanel.Children.Add(instrumentLabel);
            controlPanel.Children.Add(instrumentSelector);

            // Control buttons
            startButton = new Button { Content = "Start Trading", Margin = new Thickness(5, 5, 5, 5), Padding = new Thickness(10, 5, 10, 5), Background = Brushes.LightGreen };
            stopButton = new Button { Content = "Stop Trading", Margin = new Thickness(5, 5, 5, 5), Padding = new Thickness(10, 5, 10, 5), IsEnabled = false, Background = Brushes.LightYellow };
            emergencyStopButton = new Button { Content = "EMERGENCY STOP", Margin = new Thickness(5, 5, 5, 5), Padding = new Thickness(10, 5, 10, 5), Background = Brushes.Red, Foreground = Brushes.White };
            resumeButton = new Button { Content = "Resume Trading", Margin = new Thickness(5, 5, 5, 5), Padding = new Thickness(10, 5, 10, 5), IsEnabled = false, Background = Brushes.LightBlue };

            controlPanel.Children.Add(startButton);
            controlPanel.Children.Add(stopButton);
            controlPanel.Children.Add(emergencyStopButton);
            controlPanel.Children.Add(resumeButton);
        }

        private void CreateStatusPanel()
        {
            var statusPanel = new WrapPanel();
            statusPanel.Orientation = Orientation.Horizontal;
            statusPanel.Margin = new Thickness(10, 0, 10, 10);
            statusPanel.Background = new SolidColorBrush(Color.FromRgb(240, 240, 240));
            Grid.SetRow(statusPanel, 1);
            mainGrid.Children.Add(statusPanel);

            // Status indicators
            statusText = new TextBlock { Text = "Status: Disconnected", Margin = new Thickness(10, 10, 10, 10), FontWeight = FontWeights.Bold };
            pnlText = new TextBlock { Text = "P&L: $0.00", Margin = new Thickness(10, 10, 10, 10) };
            tradesText = new TextBlock { Text = "Trades: 0", Margin = new Thickness(10, 10, 10, 10) };
            signalsText = new TextBlock { Text = "Signals: 0", Margin = new Thickness(10, 10, 10, 10) };
            riskText = new TextBlock { Text = "Risk: Normal", Margin = new Thickness(10, 10, 10, 10), Foreground = Brushes.Green };

            statusPanel.Children.Add(statusText);
            statusPanel.Children.Add(pnlText);
            statusPanel.Children.Add(tradesText);
            statusPanel.Children.Add(signalsText);
            statusPanel.Children.Add(riskText);
        }

        private void CreateLogPanel()
        {
            var logLabel = new Label { Content = "System Log:", Margin = new Thickness(10, 0, 10, 5) };
            Grid.SetRow(logLabel, 2);
            logLabel.VerticalAlignment = VerticalAlignment.Top;
            mainGrid.Children.Add(logLabel);

            logOutput = new TextBox
            {
                Margin = new Thickness(10, 25, 10, 10),
                IsReadOnly = true,
                VerticalScrollBarVisibility = ScrollBarVisibility.Auto,
                TextWrapping = TextWrapping.Wrap,
                FontFamily = new FontFamily("Consolas"),
                FontSize = 11
            };
            Grid.SetRow(logOutput, 2);
            mainGrid.Children.Add(logOutput);
        }

        private void WireUpEvents()
        {
            // Button events
            startButton.Click += StartButton_Click;
            stopButton.Click += StopButton_Click;
            emergencyStopButton.Click += EmergencyStopButton_Click;
            resumeButton.Click += ResumeButton_Click;

            // Selector events
            accountSelector.SelectionChanged += AccountSelector_SelectionChanged;
            instrumentSelector.InstrumentChanged += InstrumentSelector_InstrumentChanged;

            // Trading engine events
            tradingEngine.StatusChanged += TradingEngine_StatusChanged;
            tradingEngine.LogMessage += TradingEngine_LogMessage;
            tradingEngine.PerformanceUpdated += TradingEngine_PerformanceUpdated;
            tradingEngine.RiskStatusChanged += TradingEngine_RiskStatusChanged;
        }

        #region Event Handlers
        private void StartButton_Click(object sender, RoutedEventArgs e)
        {
            if (accountSelector.SelectedAccount == null)
            {
                LogMessage("Please select an account first.");
                return;
            }

            tradingEngine.StartTrading(accountSelector.SelectedAccount);
            startButton.IsEnabled = false;
            stopButton.IsEnabled = true;
            emergencyStopButton.IsEnabled = true;
        }

        private void StopButton_Click(object sender, RoutedEventArgs e)
        {
            tradingEngine.StopTrading();
            startButton.IsEnabled = true;
            stopButton.IsEnabled = false;
            emergencyStopButton.IsEnabled = false;
            resumeButton.IsEnabled = false;
        }

        private void EmergencyStopButton_Click(object sender, RoutedEventArgs e)
        {
            tradingEngine.EmergencyStopTrading();
            emergencyStopButton.IsEnabled = false;
            resumeButton.IsEnabled = true;
        }

        private void ResumeButton_Click(object sender, RoutedEventArgs e)
        {
            tradingEngine.ResumeTrading();
            emergencyStopButton.IsEnabled = true;
            resumeButton.IsEnabled = false;
        }

        private void AccountSelector_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (accountSelector.SelectedAccount != null)
            {
                LogMessage($"Account selected: {accountSelector.SelectedAccount.DisplayName}");
            }
        }

        private void InstrumentSelector_InstrumentChanged(object sender, EventArgs e)
        {
            Instrument = sender as Cbi.Instrument;
            if (Instrument != null)
            {
                LogMessage($"Instrument selected: {Instrument.MasterInstrument.Name}");
            }
        }

        private void TradingEngine_StatusChanged(object sender, string status)
        {
            Dispatcher.BeginInvoke(new Action(() =>
            {
                statusText.Text = $"Status: {status}";

                // Update colors based on status
                if (status.Contains("EMERGENCY") || status.Contains("Stop"))
                    statusText.Foreground = Brushes.Red;
                else if (status.Contains("Trading"))
                    statusText.Foreground = Brushes.Green;
                else
                    statusText.Foreground = Brushes.Black;
            }));
        }

        private void TradingEngine_LogMessage(object sender, string message)
        {
            Dispatcher.BeginInvoke(new Action(() =>
            {
                LogMessage(message);
            }));
        }

        private void TradingEngine_PerformanceUpdated(object sender, PerformanceData performance)
        {
            Dispatcher.BeginInvoke(new Action(() =>
            {
                pnlText.Text = $"P&L: ${performance.DailyPnL:F2}";
                tradesText.Text = $"Trades: {performance.TradesToday}";
                signalsText.Text = $"Signals: {performance.SignalsProcessed}";

                // Update P&L color
                pnlText.Foreground = performance.DailyPnL >= 0 ? Brushes.Green : Brushes.Red;
            }));
        }

        private void TradingEngine_RiskStatusChanged(object sender, string riskStatus)
        {
            Dispatcher.BeginInvoke(new Action(() =>
            {
                riskText.Text = $"Risk: {riskStatus}";

                // Update risk color
                if (riskStatus.Contains("High") || riskStatus.Contains("Breaker"))
                    riskText.Foreground = Brushes.Red;
                else if (riskStatus.Contains("Warning"))
                    riskText.Foreground = Brushes.Orange;
                else
                    riskText.Foreground = Brushes.Green;
            }));
        }
        #endregion

        private void LogMessage(string message)
        {
            string timeStamp = DateTime.Now.ToString("HH:mm:ss.fff");
            logOutput.AppendText($"[{timeStamp}] {message}\r\n");
            logOutput.ScrollToEnd();
        }

        // IInstrumentProvider implementation
        public Cbi.Instrument Instrument
        {
            get => instrument;
            set
            {
                instrument = value;
                if (instrumentSelector != null)
                    instrumentSelector.Instrument = value;
                PropagateInstrumentChange(value);
            }
        }

        // Required NTTabPage implementation
        protected override string GetHeaderPart(string variable)
        {
            switch (variable)
            {
                case "@INSTRUMENT":
                    return Instrument == null ? "Modern Trading" : Instrument.MasterInstrument.Name;
                case "@INSTRUMENT_FULL":
                    return Instrument == null ? "Modern Trading" : Instrument.FullName;
                default:
                    return variable;
            }
        }

        public override void Cleanup()
        {
            tradingEngine?.Shutdown();
            base.Cleanup();
        }

        protected override void Restore(XElement element)
        {
            // Restore settings from workspace
        }

        protected override void Save(XElement element)
        {
            // Save settings to workspace
        }
    }

    /// <summary>
    /// Core trading engine - COMPLETE implementation with all original logic
    /// </summary>
    public class ModernInstitutionalTradingEngine : INotifyPropertyChanged
    {
        #region Core System Components
        private Account tradingAccount;
        private List<Instrument> tradingInstruments = new List<Instrument>();
        private Dictionary<string, MasterInstrument> instrumentMapping = new Dictionary<string, MasterInstrument>();

        // WebSocket Connections
        private ClientWebSocket pythonWebSocket;
        private ClientWebSocket rithmicWebSocket;
        private readonly HttpClient httpClient = new HttpClient();
        private CancellationTokenSource webSocketCancellation;
        private CancellationTokenSource rithmicCancellation;

        // Connection Status
        private bool isPythonConnected = false;
        private bool isRithmicConnected = false;
        private string connectionStatus = "Disconnected";

        // Real-time Data Management
        public readonly ConcurrentDictionary<string, MarketDataSnapshot> marketDataCache = new ConcurrentDictionary<string, MarketDataSnapshot>();
        private readonly ConcurrentQueue<InstitutionalSignal> signalQueue = new ConcurrentQueue<InstitutionalSignal>();
        private readonly ConcurrentDictionary<string, Order> activeOrders = new ConcurrentDictionary<string, Order>();
        private readonly Dictionary<string, TradeRecord> pendingTrades = new Dictionary<string, TradeRecord>();

        // Timers
        private Timer signalProcessingTimer;
        private Timer performanceUpdateTimer;
        private Timer connectionMonitorTimer;
        private readonly object tradingLock = new object();
        private int totalTrades = 0;

        // Risk Management
        private EnhancedRiskManager riskManager;
        public CircuitBreaker tradingCircuitBreaker = new CircuitBreaker();
        public bool emergencyStop = false;
        private DateTime lastRiskCheck = DateTime.MinValue;

        // Machine Learning
        private InstitutionalSignal currentSignal;
        private MarketRegime currentRegime = MarketRegime.Normal;
        private double regimeConfidence = 0.0;

        // Performance Tracking
        public PerformanceMetrics dailyMetrics = new PerformanceMetrics();
        private List<TradeRecord> tradeHistory = new List<TradeRecord>();
        private AdvancedAnalytics analytics = new AdvancedAnalytics();

        // Configuration
        public ModernTradingConfig config;
        private string configFilePath;
        private StreamWriter performanceLogger;
        private readonly object logLock = new object();
        #endregion

        #region Events
        public event EventHandler<string> StatusChanged;
        public event EventHandler<string> LogMessage;
        public event EventHandler<PerformanceData> PerformanceUpdated;
        public event EventHandler<string> RiskStatusChanged;
        public event PropertyChangedEventHandler PropertyChanged;
        #endregion

        public ModernInstitutionalTradingEngine()
        {
            riskManager = new EnhancedRiskManager(this);
            configFilePath = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments),
                "NinjaTrader 8",
                "config",
                "modern_trading_config.json"
            );
            LoadConfigurationWithFallback();
            InitializeLogging();
        }

        #region Public Interface Methods
        public async void StartTrading(Account account)
        {
            try
            {
                tradingAccount = account;
                CurrentStatus = SystemStatus.Initializing;
                StatusChanged?.Invoke(this, "Starting Modern Institutional Trading System...");
                LogMessage?.Invoke(this, "Starting Modern Institutional Trading System...");

                await InitializeWebSocketConnections();
                StartMonitoringTimers();

                CurrentStatus = SystemStatus.Trading;
                StatusChanged?.Invoke(this, "Trading Active");
                LogMessage?.Invoke(this, "Trading system started successfully");
                RiskStatusChanged?.Invoke(this, "Normal");
            }
            catch (Exception ex)
            {
                LogMessage?.Invoke(this, $"System startup failed: {ex.Message}");
                CurrentStatus = SystemStatus.EmergencyStop;
                StatusChanged?.Invoke(this, "Startup Failed");
                RiskStatusChanged?.Invoke(this, "Startup Error");
                emergencyStop = true;
            }
        }

        public void StopTrading()
        {
            try
            {
                CurrentStatus = SystemStatus.Disconnected;
                StatusChanged?.Invoke(this, "Stopping Trading System...");
                LogMessage?.Invoke(this, "Stopping Modern Institutional Trading System...");

                ShutdownSystem();

                StatusChanged?.Invoke(this, "System Stopped");
                LogMessage?.Invoke(this, "Trading system stopped successfully");
                RiskStatusChanged?.Invoke(this, "System Offline");
            }
            catch (Exception ex)
            {
                LogMessage?.Invoke(this, $"Stop error: {ex.Message}");
            }
        }

        public void EmergencyStopTrading()
        {
            emergencyStop = true;
            CurrentStatus = SystemStatus.EmergencyStop;
            StatusChanged?.Invoke(this, "EMERGENCY STOP ACTIVATED");
            LogMessage?.Invoke(this, "EMERGENCY STOP ACTIVATED - All trading halted");
            RiskStatusChanged?.Invoke(this, "EMERGENCY STOP");
        }

        public void ResumeTrading()
        {
            emergencyStop = false;
            tradingCircuitBreaker.Reset();
            CurrentStatus = SystemStatus.Trading;
            StatusChanged?.Invoke(this, "Trading Resumed");
            LogMessage?.Invoke(this, "Trading resumed after emergency stop");
            RiskStatusChanged?.Invoke(this, "Normal");
        }

        public void Shutdown()
        {
            ShutdownSystem();
        }
        #endregion

        #region Core Trading System Implementation
        private SystemStatus _currentStatus = SystemStatus.Initializing;
        public SystemStatus CurrentStatus
        {
            get => _currentStatus;
            set { _currentStatus = value; OnPropertyChanged(); }
        }

        private async Task InitializeWebSocketConnections()
        {
            await InitializePythonWebSocket();
            if (config.UseRithmicData)
                await InitializeRithmicWebSocket();
        }

        private async Task InitializePythonWebSocket()
        {
            try
            {
                pythonWebSocket = new ClientWebSocket();
                webSocketCancellation = new CancellationTokenSource();

                await pythonWebSocket.ConnectAsync(new Uri(config.PythonWebSocketUrl), webSocketCancellation.Token);

                if (pythonWebSocket.State == WebSocketState.Open)
                {
                    isPythonConnected = true;
                    LogMessage?.Invoke(this, "Python ML WebSocket connected successfully");
                    _ = Task.Run(() => ListenForPythonSignals(webSocketCancellation.Token));
                }
            }
            catch (Exception ex)
            {
                isPythonConnected = false;
                LogMessage?.Invoke(this, $"Python WebSocket connection failed: {ex.Message}");
            }
        }

        private async Task InitializeRithmicWebSocket()
        {
            try
            {
                rithmicWebSocket = new ClientWebSocket();
                rithmicCancellation = new CancellationTokenSource();

                await rithmicWebSocket.ConnectAsync(new Uri(config.RithmicWebSocketUrl), rithmicCancellation.Token);

                if (rithmicWebSocket.State == WebSocketState.Open)
                {
                    isRithmicConnected = true;
                    LogMessage?.Invoke(this, "Rithmic WebSocket connected successfully");
                    _ = Task.Run(() => ListenForRithmicData(rithmicCancellation.Token));
                    await SubscribeToMarketData();
                }
            }
            catch (Exception ex)
            {
                isRithmicConnected = false;
                LogMessage?.Invoke(this, $"Rithmic WebSocket connection failed: {ex.Message}");
            }
        }

        private void StartMonitoringTimers()
        {
            // Signal processing (every 100ms for low latency)
            signalProcessingTimer = new Timer(ProcessSignalQueue, null, TimeSpan.Zero, TimeSpan.FromMilliseconds(100));

            // Performance monitoring (every 5 seconds)
            performanceUpdateTimer = new Timer(UpdatePerformanceMetrics, null, TimeSpan.Zero, TimeSpan.FromSeconds(5));

            // Connection monitoring (every 30 seconds)
            connectionMonitorTimer = new Timer(MonitorConnections, null, TimeSpan.Zero, TimeSpan.FromSeconds(30));

            LogMessage?.Invoke(this, "All monitoring timers started");
        }

        private void ShutdownSystem()
        {
            try
            {
                // Dispose timers
                signalProcessingTimer?.Dispose();
                performanceUpdateTimer?.Dispose();
                connectionMonitorTimer?.Dispose();

                // Cancel WebSocket connections
                webSocketCancellation?.Cancel();
                rithmicCancellation?.Cancel();

                // Dispose WebSockets
                pythonWebSocket?.Dispose();
                rithmicWebSocket?.Dispose();

                // Close logger
                performanceLogger?.Close();
                performanceLogger?.Dispose();

                LogMessage?.Invoke(this, "System shutdown completed successfully");
            }
            catch (Exception ex)
            {
                LogMessage?.Invoke(this, $"Shutdown error: {ex.Message}");
            }
        }
        #endregion

        #region WebSocket Message Handling
        private async Task ListenForPythonSignals(CancellationToken cancellationToken)
        {
            var buffer = new byte[8192];

            while (!cancellationToken.IsCancellationRequested && pythonWebSocket.State == WebSocketState.Open)
            {
                try
                {
                    var result = await pythonWebSocket.ReceiveAsync(new ArraySegment<byte>(buffer), cancellationToken);

                    if (result.MessageType == WebSocketMessageType.Text)
                    {
                        string message = Encoding.UTF8.GetString(buffer, 0, result.Count);
                        ProcessPythonSignal(message);
                    }
                    else if (result.MessageType == WebSocketMessageType.Close)
                    {
                        LogMessage?.Invoke(this, "Python WebSocket closed by server");
                        break;
                    }
                }
                catch (OperationCanceledException) { break; }
                catch (Exception ex)
                {
                    LogMessage?.Invoke(this, $"Python WebSocket error: {ex.Message}");
                    tradingCircuitBreaker.RecordFailure();
                    RiskStatusChanged?.Invoke(this, $"Circuit Breaker: {tradingCircuitBreaker.GetStatus()}");
                    await Task.Delay(5000, cancellationToken);
                }
            }
        }

        private async Task ListenForRithmicData(CancellationToken cancellationToken)
        {
            var buffer = new byte[8192];

            while (!cancellationToken.IsCancellationRequested && rithmicWebSocket.State == WebSocketState.Open)
            {
                try
                {
                    var result = await rithmicWebSocket.ReceiveAsync(new ArraySegment<byte>(buffer), cancellationToken);

                    if (result.MessageType == WebSocketMessageType.Text)
                    {
                        string message = Encoding.UTF8.GetString(buffer, 0, result.Count);
                        ProcessRithmicData(message);
                    }
                    else if (result.MessageType == WebSocketMessageType.Close)
                    {
                        LogMessage?.Invoke(this, "Rithmic WebSocket closed by server");
                        break;
                    }
                }
                catch (OperationCanceledException) { break; }
                catch (Exception ex)
                {
                    LogMessage?.Invoke(this, $"Rithmic WebSocket error: {ex.Message}");
                    await Task.Delay(5000, cancellationToken);
                }
            }
        }

        private void ProcessPythonSignal(string message)
        {
            try
            {
                // First parse as generic JSON to check message type
                var jsonObject = JObject.Parse(message);
                string messageType = jsonObject["type"]?.ToString();

                if (messageType == "WELCOME")
                {
                    LogMessage?.Invoke(this, "Python ML system connected and ready");
                    return;
                }
                else if (messageType == "TRADING_SIGNAL")
                {
                    // Convert TRADING_SIGNAL to InstitutionalSignal
                    var signal = new InstitutionalSignal
                    {
                        Type = SignalType.Entry, // Default to Entry for trading signals
                        Symbol = jsonObject["symbol"]?.ToString() ?? "ES",
                        Action = jsonObject["action"]?.ToString() ?? "HOLD",
                        Confidence = jsonObject["confidence"]?.ToObject<double>() ?? 0.0,
                        TargetPrice = jsonObject["target_price"]?.ToObject<double>() ?? 0.0,
                        StopLoss = jsonObject["stop_loss"]?.ToObject<double>() ?? 0.0,
                        TakeProfit = jsonObject["take_profit"]?.ToObject<double>() ?? 0.0,
                        Quantity = jsonObject["quantity"]?.ToObject<int>() ?? 1,
                        Regime = MarketRegime.Normal,
                        Timestamp = DateTime.Now,
                        ExpiryTime = DateTime.Now.AddSeconds(config.SignalTimeoutSeconds),
                        ModelVersion = jsonObject["model_version"]?.ToString() ?? "v1.0"
                    };

                    signalQueue.Enqueue(signal);
                    SignalsProcessed++;

                    LogMessage?.Invoke(this, $"📡 Signal received: {signal.Symbol} {signal.Action} Confidence: {signal.Confidence:P2} Qty: {signal.Quantity}");
                    tradingCircuitBreaker.RecordSuccess();
                    RiskStatusChanged?.Invoke(this, $"Normal - Circuit Breaker: {tradingCircuitBreaker.GetStatus()}");
                }
                else
                {
                    // Try to deserialize as InstitutionalSignal directly
                    var signal = JsonConvert.DeserializeObject<InstitutionalSignal>(message);
                    if (signal != null)
                    {
                        signal.Timestamp = DateTime.Now;
                        signal.ExpiryTime = DateTime.Now.AddSeconds(config.SignalTimeoutSeconds);

                        signalQueue.Enqueue(signal);
                        SignalsProcessed++;

                        LogMessage?.Invoke(this, $"Signal received: {signal.Symbol} {signal.Action} Confidence: {signal.Confidence:P2}");
                        tradingCircuitBreaker.RecordSuccess();
                    }
                }
            }
            catch (Exception ex)
            {
                LogMessage?.Invoke(this, $"Signal parsing error: {ex.Message}");
                tradingCircuitBreaker.RecordFailure();
                RiskStatusChanged?.Invoke(this, $"Warning - Circuit Breaker: {tradingCircuitBreaker.GetStatus()}");
            }
        }

        private void ProcessRithmicData(string message)
        {
            try
            {
                var marketData = JsonConvert.DeserializeObject<MarketDataSnapshot>(message);
                if (marketData != null)
                {
                    marketData.Timestamp = DateTime.Now;
                    marketDataCache.AddOrUpdate(marketData.Symbol, marketData, (key, oldValue) => marketData);
                }
            }
            catch (Exception ex)
            {
                LogMessage?.Invoke(this, $"Rithmic data parsing error: {ex.Message}");
            }
        }
        #endregion

        #region Signal Processing and Execution
        private void ProcessSignalQueue(object state)
        {
            if (emergencyStop || tradingAccount == null || tradingCircuitBreaker.IsOpen)
                return;

            while (signalQueue.TryDequeue(out InstitutionalSignal signal))
            {
                ProcessInstitutionalSignal(signal);
            }
        }

        private void ProcessInstitutionalSignal(InstitutionalSignal signal)
        {
            lock (tradingLock)
            {
                try
                {
                    // Signal validation
                    if (signal.ExpiryTime < DateTime.Now)
                    {
                        LogMessage?.Invoke(this, $"Signal expired: {signal.Symbol} {signal.SignalId}");
                        return;
                    }

                    // Enhanced risk management
                    if (!riskManager.CheckSignalRisk(signal))
                    {
                        var violations = riskManager.GetRiskViolations();
                        LogMessage?.Invoke(this, $"Risk check failed for {signal.Symbol}: {string.Join(", ", violations)}");
                        RiskStatusChanged?.Invoke(this, $"Risk Violation: {violations.FirstOrDefault()}");
                        return;
                    }

                    // Instrument validation
                    var instrument = tradingInstruments.FirstOrDefault(i =>
                        i.MasterInstrument.Name.Equals(signal.Symbol, StringComparison.OrdinalIgnoreCase));

                    if (instrument == null)
                    {
                        instrument = Instrument.GetInstrument(signal.Symbol);
                        if (instrument != null)
                            tradingInstruments.Add(instrument);
                        else
                        {
                            LogMessage?.Invoke(this, $"Instrument not found: {signal.Symbol}");
                            return;
                        }
                    }

                    // Execute trade
                    ExecuteInstitutionalTrade(signal, instrument);
                    currentSignal = signal;

                    riskManager.RecordTrade(signal.Symbol);
                    ActivePositions++;
                }
                catch (Exception ex)
                {
                    LogMessage?.Invoke(this, $"Signal processing error: {ex.Message}");
                    tradingCircuitBreaker.RecordFailure();
                    RiskStatusChanged?.Invoke(this, $"Error - Circuit Breaker: {tradingCircuitBreaker.GetStatus()}");
                }
            }
        }

        private void ExecuteInstitutionalTrade(InstitutionalSignal signal, Instrument instrument)
        {
            try
            {
                // Market data validation
                if (!marketDataCache.TryGetValue(signal.Symbol, out var marketData))
                {
                    LogMessage?.Invoke(this, $"No market data available for {signal.Symbol}");
                    return;
                }

                // Price validation
                double currentPrice = marketData.LastPrice;
                double priceDeviation = Math.Abs(signal.TargetPrice - currentPrice) / currentPrice;

                if (priceDeviation > config.PriceDeviationThreshold)
                {
                    LogMessage?.Invoke(this, $"Price deviation too large for {signal.Symbol}: {priceDeviation:P2}");
                    return;
                }

                // Determine order parameters
                OrderAction orderAction = signal.Action.ToUpper() == "BUY" ? OrderAction.Buy : OrderAction.Sell;
                OrderType orderType = config.UseLimitOrders ? OrderType.Limit : OrderType.Market;
                string orderName = $"ML_{signal.Symbol}_{DateTime.Now:HHmmssfff}";

                // Create and submit order
                Order order = tradingAccount.CreateOrder(
                    instrument,
                    orderAction,
                    orderType,
                    OrderEntry.Manual,
                    TimeInForce.Day,
                    signal.Quantity,
                    config.UseLimitOrders ? signal.TargetPrice : 0,
                    0,
                    string.Empty,
                    orderName,
                    Core.Globals.MaxDate,
                    null
                );

                // Submit the order
                tradingAccount.Submit(new[] { order });

                string orderTypeText = config.UseLimitOrders ? "Limit" : "Market";
                LogMessage?.Invoke(this, $"{orderTypeText} order submitted: {signal.Action} {signal.Quantity} {signal.Symbol} @ {signal.TargetPrice:F2}");

                // Update metrics
                dailyMetrics.TradesExecuted++;
                dailyMetrics.TradesToday++;
                totalTrades++;

                // Track order execution
                TrackOrderExecution(order, signal, orderName);
            }
            catch (Exception ex)
            {
                LogMessage?.Invoke(this, $"Trade execution failed: {ex.Message}");
                tradingCircuitBreaker.RecordFailure();
                RiskStatusChanged?.Invoke(this, $"Execution Error - Circuit Breaker: {tradingCircuitBreaker.GetStatus()}");
            }
        }

        private void TrackOrderExecution(Order order, InstitutionalSignal signal, string orderName)
        {
            try
            {
                string orderKey = $"{signal.Symbol}_{order.OrderId}";
                activeOrders[orderKey] = order;

                // Create trade record
                var tradeRecord = new TradeRecord
                {
                    EntryTime = DateTime.Now,
                    Symbol = signal.Symbol,
                    Direction = signal.Action,
                    Quantity = signal.Quantity,
                    EntryPrice = signal.TargetPrice,
                    SignalSource = "PythonML",
                    Confidence = signal.Confidence,
                    Regime = signal.Regime,
                    OrderId = order.OrderId ?? orderName
                };

                // Store in pending trades
                pendingTrades[orderName] = tradeRecord;

                // Start monitoring
                _ = Task.Run(async () => await MonitorOrderExecution(order, signal, orderName));

                LogMessage?.Invoke(this, $"Order tracking started: {orderName}");
            }
            catch (Exception ex)
            {
                LogMessage?.Invoke(this, $"Order tracking setup failed: {ex.Message}");
            }
        }

        private async Task MonitorOrderExecution(Order order, InstitutionalSignal signal, string orderName)
        {
            var timeout = TimeSpan.FromSeconds(10);
            var startTime = DateTime.Now;

            while (DateTime.Now - startTime < timeout)
            {
                try
                {
                    if (order.OrderState == OrderState.Filled)
                    {
                        LogMessage?.Invoke(this, $"Order filled: {order.OrderId} for {signal.Symbol}");

                        // Move from pending to trade history
                        if (pendingTrades.ContainsKey(orderName))
                        {
                            var tradeRecord = pendingTrades[orderName];
                            tradeRecord.EntryPrice = order.AverageFillPrice;
                            tradeHistory.Add(tradeRecord);
                            pendingTrades.Remove(orderName);
                        }

                        LogMessage?.Invoke(this, $"Trade completed: {signal.Symbol} {signal.Action} {signal.Quantity} @ {order.AverageFillPrice:F2}");
                        break;
                    }
                    else if (order.OrderState == OrderState.Cancelled || order.OrderState == OrderState.Rejected)
                    {
                        LogMessage?.Invoke(this, $"Order failed: {order.OrderId} - {order.OrderState}");
                        riskManager.RecordExit(signal.Symbol);
                        ActivePositions--;

                        // Clean up pending trade
                        if (pendingTrades.ContainsKey(orderName))
                            pendingTrades.Remove(orderName);
                        break;
                    }

                    await Task.Delay(100);
                }
                catch (Exception ex)
                {
                    LogMessage?.Invoke(this, $"Order monitoring error: {ex.Message}");
                    break;
                }
            }

            // Clean up active orders tracking
            activeOrders.TryRemove($"{signal.Symbol}_{order.OrderId}", out _);
        }
        #endregion

        #region Connection Management
        private async void MonitorConnections(object state)
        {
            try
            {
                await EnsurePythonConnection();
                if (config.UseRithmicData)
                    await EnsureRithmicConnection();

                UpdateConnectionStatus();
            }
            catch (Exception ex)
            {
                LogMessage?.Invoke(this, $"Connection monitoring error: {ex.Message}");
            }
        }

        private async Task EnsurePythonConnection()
        {
            if (pythonWebSocket?.State != WebSocketState.Open && !webSocketCancellation.Token.IsCancellationRequested)
            {
                LogMessage?.Invoke(this, "Python WebSocket disconnected, attempting reconnect...");
                await InitializePythonWebSocket();
            }
        }

        private async Task EnsureRithmicConnection()
        {
            if (rithmicWebSocket?.State != WebSocketState.Open && !rithmicCancellation.Token.IsCancellationRequested)
            {
                LogMessage?.Invoke(this, "Rithmic WebSocket disconnected, attempting reconnect...");
                await InitializeRithmicWebSocket();
            }
        }

        private void UpdateConnectionStatus()
        {
            string status = "Connected: ";
            if (isPythonConnected) status += "Python ";
            if (isRithmicConnected) status += "Rithmic ";

            if (!isPythonConnected && !isRithmicConnected)
                status = "Disconnected";

            connectionStatus = status;
            OnPropertyChanged(nameof(connectionStatus));
        }

        private async Task SubscribeToMarketData()
        {
            foreach (string symbol in config.TradingSymbols)
            {
                try
                {
                    var subscribeMessage = new
                    {
                        action = "subscribe",
                        symbol = symbol,
                        dataType = "level1",
                        timestamp = DateTime.Now
                    };

                    string message = JsonConvert.SerializeObject(subscribeMessage);
                    byte[] bytes = Encoding.UTF8.GetBytes(message);

                    await rithmicWebSocket.SendAsync(
                        new ArraySegment<byte>(bytes),
                        WebSocketMessageType.Text,
                        true,
                        rithmicCancellation.Token
                    );

                    LogMessage?.Invoke(this, $"Subscribed to {symbol} market data");
                }
                catch (Exception ex)
                {
                    LogMessage?.Invoke(this, $"Failed to subscribe to {symbol}: {ex.Message}");
                }
            }
        }
        #endregion

        #region Performance Monitoring and Analytics
        private void UpdatePerformanceMetrics(object state = null)
        {
            try
            {
                if (tradingAccount != null)
                {
                    // Use available AccountItem properties
                    dailyMetrics.DailyPnL = tradingAccount.Get(AccountItem.RealizedProfitLoss, Currency.UsDollar);
                    dailyMetrics.TotalPnL = tradingAccount.Get(AccountItem.CashValue, Currency.UsDollar);

                    // Calculate advanced metrics
                    CalculateAdvancedMetrics();

                    dailyMetrics.LastUpdate = DateTime.Now;
                    LogPerformance();

                    // Update UI
                    PerformanceUpdated?.Invoke(this, new PerformanceData
                    {
                        DailyPnL = dailyMetrics.DailyPnL,
                        TradesToday = dailyMetrics.TradesToday,
                        SignalsProcessed = SignalsProcessed
                    });

                    // Reset daily trades if new day
                    if (DateTime.Now.Date != dailyMetrics.LastUpdate.Date)
                    {
                        dailyMetrics.TradesToday = 0;
                    }
                }
            }
            catch (Exception ex)
            {
                LogMessage?.Invoke(this, $"Performance update error: {ex.Message}");
            }
        }

        private void CalculateAdvancedMetrics()
        {
            if (tradeHistory.Count > 0)
            {
                // Win rate
                int winningTrades = tradeHistory.Count(t => t.PnL > 0);
                dailyMetrics.WinRate = (double)winningTrades / tradeHistory.Count;

                // Sharpe ratio
                var returns = tradeHistory.Where(t => t.PnL != 0).Select(t => t.PnL).ToList();
                if (returns.Count > 1)
                {
                    dailyMetrics.SharpeRatio = analytics.CalculateSharpeRatio(returns);
                    dailyMetrics.VaR = analytics.CalculateVaR(returns);
                }

                // Profit factor
                double grossProfit = tradeHistory.Where(t => t.PnL > 0).Sum(t => t.PnL);
                double grossLoss = Math.Abs(tradeHistory.Where(t => t.PnL < 0).Sum(t => t.PnL));
                dailyMetrics.ProfitFactor = grossLoss > 0 ? grossProfit / grossLoss : double.PositiveInfinity;

                // Average win/loss
                var winningTradePnL = tradeHistory.Where(t => t.PnL > 0).Select(t => t.PnL).ToList();
                var losingTradePnL = tradeHistory.Where(t => t.PnL < 0).Select(t => t.PnL).ToList();

                dailyMetrics.AverageWin = winningTradePnL.Any() ? winningTradePnL.Average() : 0;
                dailyMetrics.AverageLoss = losingTradePnL.Any() ? losingTradePnL.Average() : 0;

                // Max drawdown
                var equityCurve = CalculateEquityCurve();
                dailyMetrics.MaxDrawdown = analytics.CalculateMaxDrawdown(equityCurve);
            }
        }

        private List<double> CalculateEquityCurve()
        {
            var equityCurve = new List<double>();
            double runningTotal = 0;

            foreach (var trade in tradeHistory.OrderBy(t => t.EntryTime))
            {
                runningTotal += trade.PnL;
                equityCurve.Add(runningTotal);
            }

            return equityCurve;
        }

        private void LogPerformance()
        {
            lock (logLock)
            {
                try
                {
                    if (performanceLogger != null && performanceLogger.BaseStream != null)
                    {
                        var logEntry = new
                        {
                            Timestamp = DateTime.Now,
                            DailyPnL = dailyMetrics.DailyPnL,
                            TotalPnL = dailyMetrics.TotalPnL,
                            TradesExecuted = dailyMetrics.TradesExecuted,
                            TradesToday = dailyMetrics.TradesToday,
                            WinRate = dailyMetrics.WinRate,
                            SharpeRatio = dailyMetrics.SharpeRatio,
                            MaxDrawdown = dailyMetrics.MaxDrawdown,
                            VaR = dailyMetrics.VaR,
                            PythonConnected = isPythonConnected,
                            RithmicConnected = isRithmicConnected,
                            EmergencyStop = emergencyStop,
                            CircuitBreaker = tradingCircuitBreaker.GetStatus(),
                            ActivePositions = ActivePositions,
                            SignalsProcessed = SignalsProcessed
                        };

                        performanceLogger.WriteLine(JsonConvert.SerializeObject(logEntry));
                        performanceLogger.Flush();
                    }
                }
                catch (Exception ex)
                {
                    LogMessage?.Invoke(this, $"Performance logging error: {ex.Message}");
                }
            }
        }
        #endregion

        #region Configuration and Logging
        private void LoadConfigurationWithFallback()
        {
            try
            {
                if (File.Exists(configFilePath))
                {
                    string configJson = File.ReadAllText(configFilePath);
                    config = JsonConvert.DeserializeObject<ModernTradingConfig>(configJson);

                    // Validate critical settings
                    if (config.MaxDailyLoss > 0) config.MaxDailyLoss = -2000.0;
                    if (config.MinSignalConfidence < 0.5) config.MinSignalConfidence = 0.75;
                    if (config.MaxPositionSize <= 0) config.MaxPositionSize = 10;

                    LogMessage?.Invoke(this, "Configuration loaded and validated successfully");
                }
                else
                {
                    CreateDefaultConfiguration();
                }
            }
            catch (Exception ex)
            {
                LogMessage?.Invoke(this, $"Configuration load failed: {ex.Message}");
                CreateDefaultConfiguration();
            }
        }

        private void CreateDefaultConfiguration()
        {
            config = new ModernTradingConfig();
            SaveConfiguration();
            LogMessage?.Invoke(this, "Default configuration created");
        }

        private void SaveConfiguration()
        {
            try
            {
                string configDir = Path.GetDirectoryName(configFilePath);
                if (!Directory.Exists(configDir))
                    Directory.CreateDirectory(configDir);

                string configJson = JsonConvert.SerializeObject(config, Formatting.Indented);
                File.WriteAllText(configFilePath, configJson);
            }
            catch (Exception ex)
            {
                LogMessage?.Invoke(this, $"Failed to save configuration: {ex.Message}");
            }
        }

        private void InitializeLogging()
        {
            try
            {
                string logDir = Path.Combine(
                    Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments),
                    "NinjaTrader 8",
                    "log",
                    "ModernInstitutional"
                );

                if (!Directory.Exists(logDir))
                    Directory.CreateDirectory(logDir);

                string logFile = Path.Combine(logDir, $"performance_{DateTime.Now:yyyyMMdd}.log");
                performanceLogger = new StreamWriter(logFile, true);
            }
            catch (Exception ex)
            {
                LogMessage?.Invoke(this, $"Logging initialization failed: {ex.Message}");
            }
        }
        #endregion

        #region INotifyPropertyChanged Implementation
        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        private int _signalsProcessed = 0;
        public int SignalsProcessed
        {
            get => _signalsProcessed;
            set { _signalsProcessed = value; OnPropertyChanged(); }
        }

        private int _activePositions = 0;
        public int ActivePositions
        {
            get => _activePositions;
            set { _activePositions = value; OnPropertyChanged(); }
        }
        #endregion
    }

    #region Data Structures and Supporting Classes

    public class PerformanceData
    {
        public double DailyPnL { get; set; }
        public int TradesToday { get; set; }
        public int SignalsProcessed { get; set; }
    }

    public enum MarketRegime
    {
        LowVolatility,
        Normal,
        HighVolatility,
        Crisis,
        TrendingUp,
        TrendingDown
    }

    public enum SignalType
    {
        Entry,
        Exit,
        ScaleIn,
        ScaleOut,
        EmergencyExit
    }

    public enum SystemStatus
    {
        Initializing,
        Connected,
        Trading,
        Paused,
        EmergencyStop,
        Disconnected
    }

    [Serializable]
    public class InstitutionalSignal
    {
        public SignalType Type { get; set; }
        public string Symbol { get; set; }
        public string Action { get; set; } // BUY, SELL, HOLD
        public double Confidence { get; set; }
        public double TargetPrice { get; set; }
        public double StopLoss { get; set; }
        public double TakeProfit { get; set; }
        public int Quantity { get; set; }
        public MarketRegime Regime { get; set; }
        public double KellyFraction { get; set; }
        public double ExpectedSharpe { get; set; }
        public DateTime Timestamp { get; set; }
        public DateTime ExpiryTime { get; set; }
        public Dictionary<string, double> Features { get; set; } = new Dictionary<string, double>();
        public string ModelVersion { get; set; }
        public double ModelAccuracy { get; set; }
        public string SignalId { get; set; } = Guid.NewGuid().ToString();
    }

    [Serializable]
    public class MarketDataSnapshot
    {
        public string Symbol { get; set; }
        public double LastPrice { get; set; }
        public int LastSize { get; set; }
        public double BidPrice { get; set; }
        public double AskPrice { get; set; }
        public int BidSize { get; set; }
        public int AskSize { get; set; }
        public long Volume { get; set; }
        public long OpenInterest { get; set; }
        public DateTime Timestamp { get; set; }
        public double DayHigh { get; set; }
        public double DayLow { get; set; }
        public double DayOpen { get; set; }
        public double PreviousClose { get; set; }
        public double VWAP { get; set; }
    }

    [Serializable]
    public class PerformanceMetrics
    {
        public double DailyPnL { get; set; }
        public double TotalPnL { get; set; }
        public int TradesExecuted { get; set; }
        public int TradesToday { get; set; }
        public double WinRate { get; set; }
        public double SharpeRatio { get; set; }
        public double MaxDrawdown { get; set; }
        public double AverageWin { get; set; }
        public double AverageLoss { get; set; }
        public double ProfitFactor { get; set; }
        public double VaR { get; set; }
        public DateTime LastUpdate { get; set; }
    }

    [Serializable]
    public class TradeRecord
    {
        public string TradeId { get; set; } = Guid.NewGuid().ToString();
        public DateTime EntryTime { get; set; }
        public DateTime ExitTime { get; set; }
        public string Symbol { get; set; }
        public string Direction { get; set; }
        public int Quantity { get; set; }
        public double EntryPrice { get; set; }
        public double ExitPrice { get; set; }
        public double PnL { get; set; }
        public double Commission { get; set; }
        public string SignalSource { get; set; }
        public double Confidence { get; set; }
        public MarketRegime Regime { get; set; }
        public string OrderId { get; set; }
        public TimeSpan Duration { get; set; }
    }

    [Serializable]
    public class ModernTradingConfig
    {
        public string SystemName { get; set; } = "Modern Institutional Trading";
        public string PythonWebSocketUrl { get; set; } = "ws://localhost:8000/ws";
        public string RithmicWebSocketUrl { get; set; } = "wss://rituz02100.rithmic.com:443";
        public double MaxDailyLoss { get; set; } = -2000.0;
        public double MaxPositionSize { get; set; } = 10;
        public int MaxDailyTrades { get; set; } = 50;
        public double MinSignalConfidence { get; set; } = 0.75;
        public int SignalTimeoutSeconds { get; set; } = 30;
        public bool UseRithmicData { get; set; } = true;
        public bool EnableRiskManagement { get; set; } = true;
        public bool LogAllTrades { get; set; } = true;
        public bool UseLimitOrders { get; set; } = true;
        public double PriceDeviationThreshold { get; set; } = 0.02; // 2%
        public int MaxPositionsPerSymbol { get; set; } = 3;
        public List<string> TradingSymbols { get; set; } = new List<string> { "ES", "NQ", "YM", "RTY" };
    }

    public class EnhancedRiskManager
    {
        private readonly ModernInstitutionalTradingEngine parent;
        private readonly List<string> riskViolations = new List<string>();
        private readonly Dictionary<string, int> symbolPositionCounts = new Dictionary<string, int>();
        private readonly Dictionary<string, DateTime> lastTradeTime = new Dictionary<string, DateTime>();

        public EnhancedRiskManager(ModernInstitutionalTradingEngine parentSystem)
        {
            parent = parentSystem;
        }

        public bool CheckSignalRisk(InstitutionalSignal signal)
        {
            riskViolations.Clear();

            // Circuit breaker check
            if (parent.tradingCircuitBreaker.IsOpen)
            {
                riskViolations.Add("Trading circuit breaker is open");
                return false;
            }

            // Emergency stop check
            if (parent.emergencyStop)
            {
                riskViolations.Add("Emergency stop activated");
                return false;
            }

            // Signal age validation
            if (DateTime.Now - signal.Timestamp > TimeSpan.FromSeconds(parent.config.SignalTimeoutSeconds))
            {
                riskViolations.Add($"Signal too old: {DateTime.Now - signal.Timestamp}");
                return false;
            }

            // Confidence threshold
            if (signal.Confidence < parent.config.MinSignalConfidence)
            {
                riskViolations.Add($"Signal confidence too low: {signal.Confidence:P2}");
                return false;
            }

            // Daily trade limit
            if (parent.dailyMetrics.TradesToday >= parent.config.MaxDailyTrades)
            {
                riskViolations.Add($"Daily trade limit reached: {parent.dailyMetrics.TradesToday}");
                return false;
            }

            // Daily loss limit
            if (parent.dailyMetrics.DailyPnL <= parent.config.MaxDailyLoss)
            {
                riskViolations.Add($"Daily loss limit exceeded: ${parent.dailyMetrics.DailyPnL:F2}");
                return false;
            }

            // Position size limits
            if (signal.Quantity > parent.config.MaxPositionSize)
            {
                riskViolations.Add($"Position size too large: {signal.Quantity} > {parent.config.MaxPositionSize}");
                return false;
            }

            // Symbol-specific position limits
            if (symbolPositionCounts.ContainsKey(signal.Symbol) &&
                symbolPositionCounts[signal.Symbol] >= parent.config.MaxPositionsPerSymbol)
            {
                riskViolations.Add($"Symbol position limit reached: {signal.Symbol}");
                return false;
            }

            // Price validation against market data
            if (parent.marketDataCache.TryGetValue(signal.Symbol, out var marketData))
            {
                double currentPrice = marketData.LastPrice;
                double priceDeviation = Math.Abs(signal.TargetPrice - currentPrice) / currentPrice;

                if (priceDeviation > parent.config.PriceDeviationThreshold)
                {
                    riskViolations.Add($"Price deviation too large: {priceDeviation:P2}");
                    return false;
                }

                // Volatility check
                double dailyRange = marketData.DayHigh - marketData.DayLow;
                double rangePercentage = dailyRange / marketData.PreviousClose;

                if (rangePercentage > 0.05) // 5% daily range threshold
                {
                    riskViolations.Add($"High volatility detected: {rangePercentage:P2}");
                    return false;
                }
            }

            // Correlation risk
            if (HasCorrelatedPositions(signal.Symbol))
            {
                riskViolations.Add("Correlated positions detected");
                return false;
            }

            return true;
        }

        private bool HasCorrelatedPositions(string symbol)
        {
            var correlatedGroups = new Dictionary<string, List<string>>
            {
                { "ES", new List<string> { "NQ", "YM" } },
                { "NQ", new List<string> { "ES", "YM" } },
                { "RTY", new List<string> { "ES", "NQ" } }
            };

            if (correlatedGroups.TryGetValue(symbol, out var correlated))
            {
                return correlated.Any(c => symbolPositionCounts.ContainsKey(c) && symbolPositionCounts[c] > 0);
            }

            return false;
        }

        public void RecordTrade(string symbol)
        {
            if (!symbolPositionCounts.ContainsKey(symbol))
                symbolPositionCounts[symbol] = 0;

            symbolPositionCounts[symbol]++;
            lastTradeTime[symbol] = DateTime.Now;
        }

        public void RecordExit(string symbol)
        {
            if (symbolPositionCounts.ContainsKey(symbol) && symbolPositionCounts[symbol] > 0)
                symbolPositionCounts[symbol]--;
        }

        public List<string> GetRiskViolations() => new List<string>(riskViolations);
    }

    public class CircuitBreaker
    {
        private int failureCount = 0;
        private DateTime? lastFailureTime = null;
        private readonly TimeSpan timeout = TimeSpan.FromMinutes(1);
        private readonly int threshold = 5;

        public bool IsOpen => failureCount >= threshold && DateTime.Now - lastFailureTime < timeout;

        public void RecordFailure()
        {
            failureCount++;
            lastFailureTime = DateTime.Now;
        }

        public void RecordSuccess()
        {
            failureCount = 0;
            lastFailureTime = null;
        }

        public void Reset()
        {
            failureCount = 0;
            lastFailureTime = null;
        }

        public string GetStatus()
        {
            if (IsOpen) return $"OPEN (Failures: {failureCount})";
            if (failureCount > 0) return $"HALF-OPEN (Failures: {failureCount})";
            return "CLOSED";
        }
    }

    public class AdvancedAnalytics
    {
        public double CalculateVaR(List<double> returns, double confidenceLevel = 0.95)
        {
            if (returns == null || returns.Count == 0) return 0;

            var sortedReturns = returns.OrderBy(r => r).ToList();
            int varIndex = (int)((1 - confidenceLevel) * sortedReturns.Count);
            return varIndex < sortedReturns.Count ? sortedReturns[varIndex] : sortedReturns.Last();
        }

        public double CalculateMaxDrawdown(List<double> equityCurve)
        {
            if (equityCurve == null || equityCurve.Count == 0) return 0;

            double peak = equityCurve[0];
            double maxDrawdown = 0;

            foreach (var value in equityCurve)
            {
                if (value > peak) peak = value;
                double drawdown = (peak - value) / peak;
                if (drawdown > maxDrawdown) maxDrawdown = drawdown;
            }

            return maxDrawdown;
        }

        public double CalculateSharpeRatio(List<double> returns, double riskFreeRate = 0.02)
        {
            if (returns == null || returns.Count < 2) return 0;

            double avgReturn = returns.Average();
            double stdDev = Math.Sqrt(returns.Select(r => Math.Pow(r - avgReturn, 2)).Average());

            return stdDev > 0 ? (avgReturn - riskFreeRate / 252) / stdDev * Math.Sqrt(252) : 0;
        }
    }
    #endregion
}

// Add this namespace for GUI components
namespace NinjaTrader.Gui.NinjaScript
{
    // This ensures the AddOn is properly registered in the GUI namespace
}