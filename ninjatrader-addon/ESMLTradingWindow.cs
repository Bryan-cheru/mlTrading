// ES ML Trading System - Professional WPF Interface
// Beautiful, modern trading interface with real-time data visualization

#region Using declarations
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Globalization;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Threading;
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.Gui.Tools;
using NinjaTrader.NinjaScript.AddOns;
using System.Windows.Shapes;
#endregion

namespace NinjaTrader.NinjaScript.AddOns
{
    /// <summary>
    /// Professional trading interface window
    /// </summary>
    public partial class ESMLTradingWindow : Window, INotifyPropertyChanged
    {
        #region Private Variables
        private ESMLTradingSystem parentAddOn;
        private ObservableCollection<ESTradeRecord> tradeHistory;
        private ObservableCollection<WindowLogEntry> activityLog;
        private ESSignalResult currentSignals;
        private ESPerformanceMetrics currentMetrics;
        private DispatcherTimer uiUpdateTimer;
        private readonly object collectionLock = new object(); // Thread safety lock

        // UI Properties for binding
        private string systemStatus = "STOPPED";
        private string currentPrice = "$0.00";
        private string priceChange = "$0.00";
        private string priceChangePercent = "0.00%";
        private string volume = "0";
        private Brush priceChangeBrush = Brushes.Gray;

        // Signal properties
        private string consensusSignal = "HOLD";
        private string signalConfidence = "0%";
        private Brush consensusBrush = Brushes.Gray;

        // Position properties
        private string currentPosition = "0 contracts";
        private string unrealizedPnL = "$0.00";
        private string dailyTrades = "0/5";

        // Performance properties
        private string totalPnL = "$0.00";
        private string dailyPnL = "$0.00";
        private string winRate = "0%";
        private string totalTrades = "0";

        // Account properties
        private string accountBalance = "$0.00";
        private string buyingPower = "$0.00";
        #endregion

        #region Constructor
        public ESMLTradingWindow(ESMLTradingSystem addOn)
        {
            parentAddOn = addOn;
            
            // Initialize collections directly - we're already on the UI thread when window is created
            tradeHistory = new ObservableCollection<ESTradeRecord>();
            activityLog = new ObservableCollection<WindowLogEntry>();

            // Enable collection synchronization for thread safety - this prevents the threading error
            BindingOperations.EnableCollectionSynchronization(tradeHistory, collectionLock);
            BindingOperations.EnableCollectionSynchronization(activityLog, collectionLock);

            InitializeComponent();
            SetupDataBinding();
            StartUIUpdateTimer();

            LogMessage("ES ML Trading System interface initialized");
        }
        #endregion

        #region UI Initialization
        private void InitializeComponent()
        {
            // Window setup
            Title = "ES ML Trading System - Professional Interface";
            Width = 1400;
            Height = 900;
            MinWidth = 1200;
            MinHeight = 700;
            WindowStartupLocation = WindowStartupLocation.CenterScreen;
            Background = new SolidColorBrush(Color.FromRgb(45, 45, 48));

            // Create main grid
            var mainGrid = new Grid();
            mainGrid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto }); // Header
            mainGrid.RowDefinitions.Add(new RowDefinition { Height = new GridLength(1, GridUnitType.Star) }); // Main content
            mainGrid.RowDefinitions.Add(new RowDefinition { Height = new GridLength(200) }); // Bottom panel

            // Header panel
            CreateHeaderPanel(mainGrid);

            // Main content area
            CreateMainContentArea(mainGrid);

            // Bottom panel
            CreateBottomPanel(mainGrid);

            Content = mainGrid;
        }

        private void CreateHeaderPanel(Grid parent)
        {
            var headerPanel = new Grid
            {
                Background = new SolidColorBrush(Color.FromRgb(37, 37, 38)),
                Margin = new Thickness(5)
            };

            headerPanel.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
            headerPanel.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
            headerPanel.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });

            // System status section
            var statusPanel = CreateStyledPanel("System Status");
            var statusStack = new StackPanel { Orientation = Orientation.Horizontal, Margin = new Thickness(10) };

            statusStack.Children.Add(CreateLabel("Status:", FontWeights.Bold));
            var statusLabel = CreateLabel(systemStatus, FontWeights.Bold);
            statusLabel.Name = "StatusLabel";
            statusStack.Children.Add(statusLabel);

            var timeMarginLabel = CreateLabel("Time:", FontWeights.Normal);
            timeMarginLabel.Margin = new Thickness(20, 0, 5, 0);
            statusStack.Children.Add(timeMarginLabel);
            
            var timeLabel = CreateLabel(DateTime.Now.ToString("HH:mm:ss"));
            timeLabel.Name = "TimeLabel";
            statusStack.Children.Add(timeLabel);

            statusPanel.Content = statusStack;
            headerPanel.Children.Add(statusPanel);
            Grid.SetColumn(statusPanel, 0);

            // Control buttons
            var buttonPanel = new StackPanel
            {
                Orientation = Orientation.Horizontal,
                Margin = new Thickness(10),
                HorizontalAlignment = HorizontalAlignment.Right
            };

            var startButton = CreateStyledButton("Start System", Colors.Green);
            startButton.Click += StartButton_Click;
            buttonPanel.Children.Add(startButton);

            var stopButton = CreateStyledButton("Stop System", Colors.Red);
            stopButton.Margin = new Thickness(5, 0, 0, 0);
            stopButton.Click += StopButton_Click;
            buttonPanel.Children.Add(stopButton);

            var resetButton = CreateStyledButton("Reset Data", Colors.Orange);
            resetButton.Margin = new Thickness(5, 0, 0, 0);
            resetButton.Click += ResetButton_Click;
            buttonPanel.Children.Add(resetButton);

            headerPanel.Children.Add(buttonPanel);
            Grid.SetColumn(buttonPanel, 2);

            parent.Children.Add(headerPanel);
            Grid.SetRow(headerPanel, 0);
        }

        private void CreateMainContentArea(Grid parent)
        {
            var mainContent = new Grid { Margin = new Thickness(5) };
            mainContent.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(300) }); // Left panel
            mainContent.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) }); // Center chart
            mainContent.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(300) }); // Right panel

            // Left panel - Market data and signals
            CreateLeftPanel(mainContent);

            // Center panel - Price chart
            CreateCenterPanel(mainContent);

            // Right panel - Performance and account
            CreateRightPanel(mainContent);

            parent.Children.Add(mainContent);
            Grid.SetRow(mainContent, 1);
        }

        private void CreateLeftPanel(Grid parent)
        {
            var leftPanel = new StackPanel { Margin = new Thickness(0, 0, 5, 0) };

            // Market Data section
            var marketDataPanel = CreateStyledPanel("ES Futures Market Data");
            var marketGrid = CreateDataGrid(new Tuple<string, string>[]
            {
                new("Current Price:", "PriceLabel"),
                new("Change:", "ChangeLabel"),
                new("Change %:", "ChangePercentLabel"),
                new("Volume:", "VolumeLabel"),
                new("Open:", "OpenLabel"),
                new("High:", "HighLabel"),
                new("Low:", "LowLabel")
            });
            marketDataPanel.Content = marketGrid;
            leftPanel.Children.Add(marketDataPanel);

            // Signals section
            var signalsPanel = CreateStyledPanel("Current Signals");
            var signalsGrid = CreateDataGrid(new Tuple<string, string>[]
            {
                new("SMA Signal:", "SMASignalLabel"),
                new("RSI Signal:", "RSISignalLabel"),
                new("Bollinger Signal:", "BBSignalLabel"),
                new("Momentum Signal:", "MomentumSignalLabel"),
                new("Consensus:", "ConsensusLabel"),
                new("Confidence:", "ConfidenceLabel")
            });
            signalsPanel.Content = signalsGrid;
            leftPanel.Children.Add(signalsPanel);

            // Position section
            var positionPanel = CreateStyledPanel("Current Position");
            var positionGrid = CreateDataGrid(new Tuple<string, string>[]
            {
                new("Position:", "PositionLabel"),
                new("Unrealized P&L:", "UnrealizedPnLLabel"),
                new("Daily Trades:", "DailyTradesLabel")
            });
            positionPanel.Content = positionGrid;
            leftPanel.Children.Add(positionPanel);

            parent.Children.Add(leftPanel);
            Grid.SetColumn(leftPanel, 0);
        }

        private void CreateCenterPanel(Grid parent)
        {
            var centerPanel = CreateStyledPanel("ES Futures Price Chart");

            // Chart placeholder - In real implementation, integrate with NinjaTrader charts
            var chartArea = new Border
            {
                Background = Brushes.Black,
                CornerRadius = new CornerRadius(5),
                Margin = new Thickness(10)
            };

            var chartLabel = new TextBlock
            {
                Text = "Real-time ES Futures Chart\n(Integrated with NinjaTrader)",
                Foreground = Brushes.White,
                HorizontalAlignment = HorizontalAlignment.Center,
                VerticalAlignment = VerticalAlignment.Center,
                TextAlignment = TextAlignment.Center,
                FontSize = 16
            };

            chartArea.Child = chartLabel;
            centerPanel.Content = chartArea;

            parent.Children.Add(centerPanel);
            Grid.SetColumn(centerPanel, 1);
        }

        private void CreateRightPanel(Grid parent)
        {
            var rightPanel = new StackPanel { Margin = new Thickness(5, 0, 0, 0) };

            // Performance section
            var perfPanel = CreateStyledPanel("Performance Metrics");
            var perfGrid = CreateDataGrid(new Tuple<string, string>[]
            {
                new("Total P&L:", "TotalPnLLabel"),
                new("Today's P&L:", "DailyPnLLabel"),
                new("Total Trades:", "TotalTradesLabel"),
                new("Win Rate:", "WinRateLabel"),
                new("Profit Factor:", "ProfitFactorLabel")
            });
            perfPanel.Content = perfGrid;
            rightPanel.Children.Add(perfPanel);

            // Account section
            var accountPanel = CreateStyledPanel("Account Information");
            var accountGrid = CreateDataGrid(new Tuple<string, string>[]
            {
                new("Account:", "AccountNameLabel"),
                new("Balance:", "BalanceLabel"),
                new("Buying Power:", "BuyingPowerLabel"),
                new("Margin Used:", "MarginUsedLabel")
            });
            accountPanel.Content = accountGrid;
            rightPanel.Children.Add(accountPanel);

            // Risk section
            var riskPanel = CreateStyledPanel("Risk Management");
            var riskGrid = CreateDataGrid(new Tuple<string, string>[]
            {
                new("Max Position:", "MaxPositionLabel"),
                new("Max Daily Trades:", "MaxDailyTradesLabel"),
                new("Min Confidence:", "MinConfidenceLabel"),
                new("Risk Per Trade:", "RiskPerTradeLabel")
            });
            riskPanel.Content = riskGrid;
            rightPanel.Children.Add(riskPanel);

            parent.Children.Add(rightPanel);
            Grid.SetColumn(rightPanel, 2);
        }

        private void CreateBottomPanel(Grid parent)
        {
            var bottomPanel = CreateStyledPanel("System Activity");

            var tabControl = new TabControl
            {
                Background = new SolidColorBrush(Color.FromRgb(45, 45, 48)),
                Margin = new Thickness(5)
            };

            // Activity log tab
            var logTab = new TabItem { Header = "Activity Log" };
            var logListBox = new ListBox
            {
                Name = "ActivityLogList",
                Background = Brushes.Black,
                Foreground = Brushes.LightGreen,
                FontFamily = new FontFamily("Consolas"),
                FontSize = 12,
                ItemsSource = activityLog
            };
            logTab.Content = logListBox;
            tabControl.Items.Add(logTab);

            // Trade history tab
            var tradeTab = new TabItem { Header = "Trade History" };
            var tradeDataGrid = new DataGrid
            {
                Name = "TradeHistoryGrid",
                AutoGenerateColumns = false,
                ItemsSource = tradeHistory,
                Background = new SolidColorBrush(Color.FromRgb(45, 45, 48)),
                Foreground = Brushes.White
            };

            // Define columns
            tradeDataGrid.Columns.Add(new DataGridTextColumn
            {
                Header = "Time",
                Binding = new Binding("Timestamp") { StringFormat = "HH:mm:ss" },
                Width = 80
            });
            tradeDataGrid.Columns.Add(new DataGridTextColumn
            {
                Header = "Action",
                Binding = new Binding("Action"),
                Width = 60
            });
            tradeDataGrid.Columns.Add(new DataGridTextColumn
            {
                Header = "Quantity",
                Binding = new Binding("Quantity"),
                Width = 70
            });
            tradeDataGrid.Columns.Add(new DataGridTextColumn
            {
                Header = "Price",
                Binding = new Binding("Price") { StringFormat = "F2" },
                Width = 80
            });
            tradeDataGrid.Columns.Add(new DataGridTextColumn
            {
                Header = "P&L",
                Binding = new Binding("RealizedPnL") { StringFormat = "C" },
                Width = 80
            });

            tradeTab.Content = tradeDataGrid;
            tabControl.Items.Add(tradeTab);

            bottomPanel.Content = tabControl;
            parent.Children.Add(bottomPanel);
            Grid.SetRow(bottomPanel, 2);
        }
        #endregion

        #region Helper Methods
        private GroupBox CreateStyledPanel(string header)
        {
            return new GroupBox
            {
                Header = header,
                Foreground = Brushes.White,
                BorderBrush = new SolidColorBrush(Color.FromRgb(100, 100, 100)),
                Margin = new Thickness(0, 0, 0, 10),
                Padding = new Thickness(5),
                FontWeight = FontWeights.Bold
            };
        }

        private Grid CreateDataGrid(Tuple<string, string>[] items)
        {
            var grid = new Grid { Margin = new Thickness(5) };

            for (int i = 0; i < items.Length; i++)
            {
                grid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });

                var label = CreateLabel(items[i].Item1, FontWeights.Bold);
                grid.Children.Add(label);
                Grid.SetRow(label, i);
                Grid.SetColumn(label, 0);

                var valueLabel = CreateLabel("$0.00");
                valueLabel.Name = items[i].Item2;
                valueLabel.HorizontalAlignment = HorizontalAlignment.Right;
                grid.Children.Add(valueLabel);
                Grid.SetRow(valueLabel, i);
                Grid.SetColumn(valueLabel, 1);
            }

            grid.ColumnDefinitions.Add(new ColumnDefinition());
            grid.ColumnDefinitions.Add(new ColumnDefinition());

            return grid;
        }

        private Label CreateLabel(string text, FontWeight weight = default)
        {
            return new Label
            {
                Content = text,
                Foreground = Brushes.White,
                FontWeight = weight,
                Padding = new Thickness(0, 2, 0, 2)
            };
        }

        private Button CreateStyledButton(string text, Color color)
        {
            return new Button
            {
                Content = text,
                Background = new SolidColorBrush(color),
                Foreground = Brushes.White,
                BorderThickness = new Thickness(0),
                Padding = new Thickness(15, 5, 15, 5),
                FontWeight = FontWeights.Bold,
                Cursor = Cursors.Hand
            };
        }

        private void UpdateLabel(string name, string text, Brush brush = null)
        {
            var label = FindName(name) as Label;
            if (label != null)
            {
                label.Content = text;
                if (brush != null)
                    label.Foreground = brush;
            }
        }

        private void UpdateSignalLabel(string name, string signal)
        {
            var brush = signal == "BUY" ? Brushes.LightGreen :
                       signal == "SELL" ? Brushes.LightCoral : Brushes.Orange;
            UpdateLabel(name, signal, brush);
        }

        private Brush GetConfidenceColor(double confidence)
        {
            if (confidence >= 0.7) return Brushes.LightGreen;
            if (confidence >= 0.5) return Brushes.Orange;
            return Brushes.LightCoral;
        }
        #endregion

        #region UI Updates
        private void SetupDataBinding()
        {
            DataContext = this;

            // Start UI update timer
            uiUpdateTimer = new DispatcherTimer
            {
                Interval = TimeSpan.FromSeconds(1)
            };
            uiUpdateTimer.Tick += UiUpdateTimer_Tick;
            uiUpdateTimer.Start();
        }

        private void StartUIUpdateTimer()
        {
            var timer = new DispatcherTimer
            {
                Interval = TimeSpan.FromSeconds(1)
            };
            timer.Tick += (s, e) =>
            {
                // Update time display
                var timeLabel = FindName("TimeLabel") as Label;
                if (timeLabel != null)
                    timeLabel.Content = DateTime.Now.ToString("HH:mm:ss");
            };
            timer.Start();
        }

        private void UiUpdateTimer_Tick(object sender, EventArgs e)
        {
            // Update time and other real-time elements
            OnPropertyChanged(nameof(CurrentTime));
        }

        public void UpdateMarketData(Bars bars)
        {
            try
            {
                if (bars?.Count > 0)
                {
                    // OFFICIAL PATTERN: Following AddOnFramework.cs dispatcher usage
                    if (!Dispatcher.CheckAccess())
                    {
                        Dispatcher.InvokeAsync(() => UpdateMarketData(bars));
                        return;
                    }

                    var latest = bars.GetClose(bars.Count - 1);
                    var previous = bars.Count > 1 ? bars.GetClose(bars.Count - 2) : latest;

                    // Validate prices to prevent display errors
                    if (double.IsNaN(latest) || double.IsInfinity(latest) || latest < 0)
                        latest = 0;
                    if (double.IsNaN(previous) || double.IsInfinity(previous) || previous < 0)
                        previous = latest;

                    CurrentPrice = $"${latest:F2}";
                    var change = latest - previous;
                    PriceChange = $"${change:+0.00;-0.00}";

                    // Prevent division by zero
                    if (previous != 0)
                        PriceChangePercent = $"{(change / previous * 100):+0.00;-0.00}%";
                    else
                        PriceChangePercent = "0.00%";

                    PriceChangeBrush = change >= 0 ? Brushes.LightGreen : Brushes.LightCoral;

                    var volumeValue = bars.GetVolume(bars.Count - 1);
                    Volume = volumeValue >= 0 ? $"{volumeValue:N0}" : "0";

                    // Update UI labels with validation
                    UpdateLabel("PriceLabel", CurrentPrice);
                    UpdateLabel("ChangeLabel", PriceChange, PriceChangeBrush);
                    UpdateLabel("ChangePercentLabel", PriceChangePercent, PriceChangeBrush);
                    UpdateLabel("VolumeLabel", Volume);

                    // Safely get OHLC values
                    var openValue = bars.GetOpen(bars.Count - 1);
                    var highValue = bars.GetHigh(bars.Count - 1);
                    var lowValue = bars.GetLow(bars.Count - 1);

                    UpdateLabel("OpenLabel", $"${(double.IsNaN(openValue) ? 0 : openValue):F2}");
                    UpdateLabel("HighLabel", $"${(double.IsNaN(highValue) ? 0 : highValue):F2}");
                    UpdateLabel("LowLabel", $"${(double.IsNaN(lowValue) ? 0 : lowValue):F2}");
                }
            }
            catch (Exception ex)
            {
                // OFFICIAL PATTERN: Following AddOnFramework.cs error handling
                Dispatcher.InvokeAsync(() =>
                {
                    // It is important to protect NinjaTrader from any unhandled exceptions
                    System.Diagnostics.Debug.WriteLine($"ES ML Trading System - UpdateMarketData Exception: {ex.Message}");
                });
                
                // Reset to safe values
                CurrentPrice = "$0.00";
                PriceChange = "$0.00";
                PriceChangePercent = "0.00%";
                Volume = "0";
            }
        }

        public void UpdateSignals(ESSignalResult signals)
        {
            // OFFICIAL PATTERN: Following AddOnFramework.cs dispatcher usage
            if (!Dispatcher.CheckAccess())
            {
                Dispatcher.InvokeAsync(() => UpdateSignals(signals));
                return;
            }

            currentSignals = signals;

            if (signals?.IndividualSignals != null)
            {
                // Use TryGetValue instead of GetValueOrDefault for .NET Framework 4.8 compatibility
                signals.IndividualSignals.TryGetValue("SMA", out string smaSignal);
                UpdateSignalLabel("SMASignalLabel", smaSignal ?? "HOLD");

                signals.IndividualSignals.TryGetValue("RSI", out string rsiSignal);
                UpdateSignalLabel("RSISignalLabel", rsiSignal ?? "HOLD");

                signals.IndividualSignals.TryGetValue("Bollinger", out string bbSignal);
                UpdateSignalLabel("BBSignalLabel", bbSignal ?? "HOLD");

                signals.IndividualSignals.TryGetValue("Momentum", out string momentumSignal);
                UpdateSignalLabel("MomentumSignalLabel", momentumSignal ?? "HOLD");

                UpdateSignalLabel("ConsensusLabel", signals.Signal);
                UpdateLabel("ConfidenceLabel", $"{signals.Confidence:P1}", GetConfidenceColor(signals.Confidence));
            }
        }

        public void UpdatePerformanceMetrics(ESPerformanceMetrics metrics)
        {
            // OFFICIAL PATTERN: Following AddOnFramework.cs dispatcher usage
            if (!Dispatcher.CheckAccess())
            {
                Dispatcher.InvokeAsync(() => UpdatePerformanceMetrics(metrics));
                return;
            }

            currentMetrics = metrics;

            UpdateLabel("TotalPnLLabel", $"${metrics.TotalPnL:F2}", metrics.TotalPnL >= 0 ? Brushes.LightGreen : Brushes.LightCoral);
            UpdateLabel("DailyPnLLabel", $"${metrics.DailyPnL:F2}", metrics.DailyPnL >= 0 ? Brushes.LightGreen : Brushes.LightCoral);
            UpdateLabel("TotalTradesLabel", metrics.TotalTrades.ToString());
            UpdateLabel("WinRateLabel", $"{metrics.WinRate:P1}");
            UpdateLabel("ProfitFactorLabel", $"{metrics.ProfitFactor:F2}");
        }

        public void UpdatePosition(Position position)
        {
            // OFFICIAL PATTERN: Following AddOnFramework.cs dispatcher usage
            if (!Dispatcher.CheckAccess())
            {
                Dispatcher.InvokeAsync(() => UpdatePosition(position));
                return;
            }

            if (position != null)
            {
                CurrentPosition = $"{position.Quantity} contracts";
                UnrealizedPnL = $"${position.GetUnrealizedProfitLoss(PerformanceUnit.Currency):F2}";

                UpdateLabel("PositionLabel", CurrentPosition);
                UpdateLabel("UnrealizedPnLLabel", UnrealizedPnL,
                    position.GetUnrealizedProfitLoss(PerformanceUnit.Currency) >= 0 ? Brushes.LightGreen : Brushes.LightCoral);
            }
        }

        public void UpdateAccountInfo(Account account)
        {
            // OFFICIAL PATTERN: Following AddOnFramework.cs dispatcher usage
            if (!Dispatcher.CheckAccess())
            {
                Dispatcher.InvokeAsync(() => UpdateAccountInfo(account));
                return;
            }

            if (account != null)
            {
                AccountBalance = $"${account.Get(AccountItem.CashValue, Currency.UsDollar):F2}";
                BuyingPower = $"${account.Get(AccountItem.BuyingPower, Currency.UsDollar):F2}";

                UpdateLabel("AccountNameLabel", account.DisplayName);
                UpdateLabel("BalanceLabel", AccountBalance);
                UpdateLabel("BuyingPowerLabel", BuyingPower);
            }
        }

        public void UpdateTradeHistory(List<ESTradeRecord> trades)
        {
            // Use synchronous invoke with proper locking for collection updates
            if (Application.Current?.Dispatcher != null)
            {
                Application.Current.Dispatcher.Invoke(() =>
                {
                    lock (collectionLock)
                    {
                        try
                        {
                            if (tradeHistory != null)
                            {
                                tradeHistory.Clear();
                                // Use Skip and Take instead of TakeLast for .NET Framework 4.8 compatibility
                                var recentTrades = trades.Count > 50 ? trades.Skip(trades.Count - 50) : trades;
                                foreach (var trade in recentTrades)
                                {
                                    tradeHistory.Add(trade);
                                }
                            }
                        }
                        catch (Exception ex)
                        {
                            // Log error but don't crash the system
                            System.Diagnostics.Debug.WriteLine($"Error updating trade history: {ex.Message}");
                        }
                    }
                });
            }
        }

        public void LogMessage(string message)
        {
            // Use synchronous invoke with proper locking for collection updates
            if (Application.Current?.Dispatcher != null)
            {
                Application.Current.Dispatcher.Invoke(() =>
                {
                    lock (collectionLock)
                    {
                        try
                        {
                            if (activityLog != null)
                            {
                                var logEntry = new WindowLogEntry
                                {
                                    Timestamp = DateTime.Now,
                                    Message = message
                                };

                                activityLog.Insert(0, logEntry);

                                // Keep only last 100 entries
                                while (activityLog.Count > 100)
                                {
                                    activityLog.RemoveAt(activityLog.Count - 1);
                                }
                            }
                        }
                        catch (Exception ex)
                        {
                            // Log error but don't crash the system
                            System.Diagnostics.Debug.WriteLine($"Error logging message: {ex.Message}");
                        }
                    }
                });
            }
        }
        #endregion

        #region Event Handlers
        private void StartButton_Click(object sender, RoutedEventArgs e)
        {
            parentAddOn.StartSystem();
            SystemStatus = "RUNNING";
            UpdateLabel("StatusLabel", SystemStatus, Brushes.LightGreen);
            LogMessage("Trading system started");
        }

        private void StopButton_Click(object sender, RoutedEventArgs e)
        {
            parentAddOn.StopSystem();
            SystemStatus = "STOPPED";
            UpdateLabel("StatusLabel", SystemStatus, Brushes.LightCoral);
            LogMessage("Trading system stopped");
        }

        private void ResetButton_Click(object sender, RoutedEventArgs e)
        {
            var result = MessageBox.Show("Are you sure you want to reset all data?",
                "Confirm Reset", MessageBoxButton.YesNo, MessageBoxImage.Question);

            if (result == MessageBoxResult.Yes)
            {
                lock (collectionLock)
                {
                    tradeHistory.Clear();
                    activityLog.Clear();
                }
                LogMessage("All data has been reset");
            }
        }
        #endregion

        #region Properties
        public string SystemStatus
        {
            get => systemStatus;
            set { systemStatus = value; OnPropertyChanged(); }
        }

        public string CurrentPrice
        {
            get => currentPrice;
            set { currentPrice = value; OnPropertyChanged(); }
        }

        public string PriceChange
        {
            get => priceChange;
            set { priceChange = value; OnPropertyChanged(); }
        }

        public string PriceChangePercent
        {
            get => priceChangePercent;
            set { priceChangePercent = value; OnPropertyChanged(); }
        }

        public string Volume
        {
            get => volume;
            set { volume = value; OnPropertyChanged(); }
        }

        public Brush PriceChangeBrush
        {
            get => priceChangeBrush;
            set { priceChangeBrush = value; OnPropertyChanged(); }
        }

        public string CurrentPosition
        {
            get => currentPosition;
            set { currentPosition = value; OnPropertyChanged(); }
        }

        public string UnrealizedPnL
        {
            get => unrealizedPnL;
            set { unrealizedPnL = value; OnPropertyChanged(); }
        }

        public string AccountBalance
        {
            get => accountBalance;
            set { accountBalance = value; OnPropertyChanged(); }
        }

        public string BuyingPower
        {
            get => buyingPower;
            set { buyingPower = value; OnPropertyChanged(); }
        }

        public string CurrentTime => DateTime.Now.ToString("HH:mm:ss");
        #endregion

        #region INotifyPropertyChanged
        public event PropertyChangedEventHandler PropertyChanged;

        protected void OnPropertyChanged([System.Runtime.CompilerServices.CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
        #endregion
    }

    #region Supporting Classes
    public class WindowLogEntry
    {
        public DateTime Timestamp { get; set; }
        public string Message { get; set; }
        public override string ToString() => $"[{Timestamp:HH:mm:ss}] {Message}";
    }
    #endregion
}