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
using System.Threading;
using System.Net;
using System.Net.Sockets;
#endregion

//This namespace holds Add ons in this folder and is required. Do not change it. 
namespace NinjaTrader.NinjaScript.AddOns
{
    /// <summary>
    /// ES Trading Order Executor AddOn
    /// Receives order commands from Python and executes them in NinjaTrader
    /// </summary>
    public class ESOrderExecutor : AddOnBase
    {
        #region Variables
        private Thread orderMonitorThread;
        private bool isRunning;
        private TcpListener tcpListener;
        private const int LISTEN_PORT = 36974; // Different from ATI port
        private string logFilePath;
        private Account tradingAccount;
        #endregion

        #region OnStateChange
        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = @"ES Trading Order Executor - Receives orders from Python and executes them";
                Name = "ESOrderExecutor";
                
                // Initialize log file path
                logFilePath = Path.Combine(
                    Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments),
                    "NinjaTrader 8", "logs", "ESOrderExecutor.log"
                );
            }
            else if (State == State.Active)
            {
                StartOrderExecutor();
                LogMessage("ES Order Executor started successfully");
            }
            else if (State == State.Terminated)
            {
                StopOrderExecutor();
                LogMessage("ES Order Executor stopped");
            }
        }
        #endregion

        #region Order Executor Methods
        private void StartOrderExecutor()
        {
            try
            {
                isRunning = true;
                
                // Find simulation account
                tradingAccount = Account.All.FirstOrDefault(a => a.Name.Contains("Sim"));
                if (tradingAccount == null)
                {
                    LogMessage("ERROR: No simulation account found");
                    return;
                }
                
                LogMessage($"Using account: {tradingAccount.Name}");
                
                // Start TCP listener for Python commands
                tcpListener = new TcpListener(IPAddress.Any, LISTEN_PORT);
                tcpListener.Start();
                
                // Start monitoring thread
                orderMonitorThread = new Thread(MonitorForOrders);
                orderMonitorThread.IsBackground = true;
                orderMonitorThread.Start();
                
                LogMessage($"Listening on port {LISTEN_PORT} for order commands");
            }
            catch (Exception ex)
            {
                LogMessage($"ERROR starting executor: {ex.Message}");
            }
        }

        private void StopOrderExecutor()
        {
            try
            {
                isRunning = false;
                
                if (tcpListener != null)
                {
                    tcpListener.Stop();
                }
                
                if (orderMonitorThread != null && orderMonitorThread.IsAlive)
                {
                    orderMonitorThread.Join(5000); // Wait 5 seconds max
                }
                
                LogMessage("Order executor stopped successfully");
            }
            catch (Exception ex)
            {
                LogMessage($"ERROR stopping executor: {ex.Message}");
            }
        }

        private void MonitorForOrders()
        {
            while (isRunning)
            {
                try
                {
                    if (tcpListener.Pending())
                    {
                        using (TcpClient client = tcpListener.AcceptTcpClient())
                        using (NetworkStream stream = client.GetStream())
                        using (StreamReader reader = new StreamReader(stream))
                        using (StreamWriter writer = new StreamWriter(stream))
                        {
                            string orderCommand = reader.ReadLine();
                            LogMessage($"Received order command: {orderCommand}");
                            
                            string response = ProcessOrderCommand(orderCommand);
                            
                            writer.WriteLine(response);
                            writer.Flush();
                            
                            LogMessage($"Sent response: {response}");
                        }
                    }
                    
                    Thread.Sleep(100); // Small delay to prevent high CPU usage
                }
                catch (Exception ex)
                {
                    LogMessage($"ERROR in monitor loop: {ex.Message}");
                    Thread.Sleep(1000); // Longer delay on error
                }
            }
        }

        private string ProcessOrderCommand(string command)
        {
            try
            {
                if (string.IsNullOrEmpty(command))
                    return "ERROR: Empty command";

                string[] parts = command.Split('|');
                
                if (parts.Length < 6)
                    return "ERROR: Invalid command format. Expected: ACTION|OrderID|Instrument|Side|Quantity|OrderType";

                string action = parts[0].ToUpper();
                string orderId = parts[1];
                string instrument = parts[2];
                string side = parts[3].ToUpper();
                int quantity = int.Parse(parts[4]);
                string orderType = parts[5].ToUpper();

                if (action == "PLACE_ORDER")
                {
                    return PlaceESOrder(orderId, instrument, side, quantity, orderType);
                }
                else if (action == "CANCEL_ORDER")
                {
                    return CancelOrder(orderId);
                }
                else if (action == "STATUS")
                {
                    return GetAccountStatus();
                }
                else
                {
                    return $"ERROR: Unknown action '{action}'";
                }
            }
            catch (Exception ex)
            {
                LogMessage($"ERROR processing command: {ex.Message}");
                return $"ERROR: {ex.Message}";
            }
        }

        private string PlaceESOrder(string orderId, string instrument, string side, int quantity, string orderType)
        {
            try
            {
                if (tradingAccount == null)
                    return "ERROR: No trading account available";

                // Get ES instrument using the correct NinjaTrader API
                Instrument esInstrument = Instrument.GetInstrument("ES 12-24");
                
                // If specific contract not found, try to get current ES contract
                if (esInstrument == null)
                {
                    MasterInstrument masterInstrument = MasterInstrument.All.FirstOrDefault(mi => 
                        mi.Name.StartsWith("ES") && mi.InstrumentType == InstrumentType.Future);

                    if (masterInstrument == null)
                        return "ERROR: ES instrument not found";

                    // Try to get any available ES contract
                    esInstrument = Instrument.GetInstrument(masterInstrument.Name);
                }
                
                if (esInstrument == null)
                    return "ERROR: Current ES contract not available";

                // Determine order action
                OrderAction orderAction = side == "BUY" ? OrderAction.Buy : OrderAction.Sell;

                // Place the order using correct CreateOrder signature
                Order order = null;
                
                if (orderType == "MKT" || orderType == "MARKET")
                {
                    order = tradingAccount.CreateOrder(esInstrument, orderAction, OrderType.Market, TimeInForce.Day, quantity, 0, 0, "", orderId, null);
                }
                else if (orderType == "LMT" || orderType == "LIMIT")
                {
                    // For limit orders, you'd need to specify price - using a placeholder price
                    double limitPrice = 5000.0; // Default ES price - should be passed from Python
                    order = tradingAccount.CreateOrder(esInstrument, orderAction, OrderType.Limit, TimeInForce.Day, quantity, limitPrice, 0, "", orderId, null);
                }
                else
                {
                    return $"ERROR: Unsupported order type '{orderType}'";
                }

                if (order != null)
                {
                    tradingAccount.Submit(new[] { order });
                    LogMessage($"Order submitted: {orderId} - {side} {quantity} {esInstrument.FullName}");
                    return $"SUCCESS: Order {orderId} submitted for {side} {quantity} {esInstrument.FullName}";
                }
                else
                {
                    return "ERROR: Failed to create order";
                }
            }
            catch (Exception ex)
            {
                LogMessage($"ERROR placing order: {ex.Message}");
                return $"ERROR: {ex.Message}";
            }
        }

        private string CancelOrder(string orderId)
        {
            try
            {
                if (tradingAccount == null)
                    return "ERROR: No trading account available";

                var order = tradingAccount.Orders.FirstOrDefault(o => o.Name == orderId);
                
                if (order != null)
                {
                    tradingAccount.Cancel(new[] { order });
                    LogMessage($"Order cancelled: {orderId}");
                    return $"SUCCESS: Order {orderId} cancelled";
                }
                else
                {
                    return $"ERROR: Order {orderId} not found";
                }
            }
            catch (Exception ex)
            {
                LogMessage($"ERROR cancelling order: {ex.Message}");
                return $"ERROR: {ex.Message}";
            }
        }

        private string GetAccountStatus()
        {
            try
            {
                if (tradingAccount == null)
                    return "ERROR: No trading account available";

                var status = $"ACCOUNT_STATUS|" +
                           $"Name:{tradingAccount.Name}|" +
                           $"BuyingPower:{tradingAccount.Get(AccountItem.BuyingPower, Currency.UsDollar)}|" +
                           $"CashValue:{tradingAccount.Get(AccountItem.CashValue, Currency.UsDollar)}|" +
                           $"RealizedPnL:{tradingAccount.Get(AccountItem.RealizedProfitLoss, Currency.UsDollar)}|" +
                           $"Orders:{tradingAccount.Orders.Count}|" +
                           $"Positions:{tradingAccount.Positions.Count}";

                return status;
            }
            catch (Exception ex)
            {
                LogMessage($"ERROR getting account status: {ex.Message}");
                return $"ERROR: {ex.Message}";
            }
        }

        private void LogMessage(string message)
        {
            try
            {
                string logEntry = $"{DateTime.Now:yyyy-MM-dd HH:mm:ss} - {message}";
                File.AppendAllText(logFilePath, logEntry + Environment.NewLine);
                
                // Use NinjaTrader's Print method for output
                Print(logEntry);
            }
            catch
            {
                // Ignore logging errors to prevent crashes
            }
        }
        #endregion
    }
}
