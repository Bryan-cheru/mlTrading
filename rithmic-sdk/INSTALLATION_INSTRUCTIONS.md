
# RITHMIC SDK INSTALLATION INSTRUCTIONS

## ⚠️ IMPORTANT: Manual Steps Required

The Rithmic SDK cannot be automatically downloaded due to licensing requirements.
You must obtain it through official channels:

### Step 1: Contact Your Broker
- Your client needs to have Rithmic access through their broker
- Common brokers with Rithmic: AMP Futures, NinjaTrader, etc.
- Request R|API access and credentials

### Step 2: Download Official SDK
1. Visit: https://www.rithmic.com/developers/
2. Log in with provided credentials
3. Download "R|API SDK" for Windows
4. Extract files to this directory: rithmic-sdk/

### Step 3: Required Files
After extraction, you should have:
- REngine.dll (Main API library)
- RApi.dll (Additional library)  
- Documentation files
- Sample applications
- Protocol buffer definitions

### Step 4: Credentials Setup
Create a file: config/rithmic_credentials.env
```
RITHMIC_USER_ID=your_user_id
RITHMIC_PASSWORD=your_password
RITHMIC_SYSTEM_NAME=system_name_from_broker
```

### Step 5: Test Connection
Run the test script:
```
python data-pipeline/ingestion/test_rithmic_connection.py
```

## 📞 Support Contacts

- Rithmic Support: support@rithmic.com
- R|API Documentation: Available in SDK download
- Your Broker's Trading Desk: For account setup

## 🔐 Security Notes

- Never commit credentials to version control
- Use environment variables for sensitive data
- Test with paper trading first
- Follow exchange data usage policies
