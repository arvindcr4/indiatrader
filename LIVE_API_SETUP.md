# IndiaTrader Live API Setup Guide

This guide will help you configure the Dhan and ICICI Breeze APIs for use with IndiaTrader.

## Installation Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

This will install the necessary packages, including:
- `dhanhq`: Dhan API client
- `breeze-connect`: ICICI Breeze API client

## Dhan API Setup

1. **Create a Dhan Trading Account**
   - Sign up at [Dhan.co](https://dhan.co) if you don't have an account

2. **Generate API Credentials**
   - Log in to your Dhan account
   - Navigate to Settings > API Access
   - Create a new API key
   - Note down your Client ID and Access Token

3. **Configuration Options**

   **Option 1: Set Environment Variables**
   ```bash
   export DHAN_CLIENT_ID="your_client_id_here"
   export DHAN_ACCESS_TOKEN="your_access_token_here"
   ```

   **Option 2: Edit config.yaml file**
   ```yaml
   data_sources:
     market_data:
       dhan:
         client_id: "YOUR_DHAN_CLIENT_ID"
         access_token: "YOUR_DHAN_ACCESS_TOKEN"
   ```

## ICICI Breeze API Setup

1. **Create an ICICI Direct Account**
   - Sign up at [ICICIDirect.com](https://www.icicidirect.com) if you don't have an account

2. **Register for API Access**
   - Visit [ICICI Developer Portal](https://api.icicidirect.com/apiuser/home)
   - Register for API access
   - Generate API key and secret
   - Generate a session token (note: this needs to be refreshed daily)

3. **Configuration Options**

   **Option 1: Set Environment Variables**
   ```bash
   export ICICI_API_KEY="your_api_key_here"
   export ICICI_API_SECRET="your_api_secret_here"
   export ICICI_SESSION_TOKEN="your_session_token_here"
   ```

   **Option 2: Edit config.yaml file**
   ```yaml
   data_sources:
     market_data:
       icici:
         api_key: "YOUR_ICICI_API_KEY"
         api_secret: "YOUR_ICICI_API_SECRET"
         session_token: "YOUR_ICICI_SESSION_TOKEN"
   ```

## Using the Live APIs

1. Launch the unified GUI:
   ```bash
   python unified_gui.py
   ```

2. Go to the "🔴 Live Data" tab

3. Enter a symbol (e.g., "NIFTY") and select the exchange

4. Choose your data type (Quote, Historical, or Index)

5. Select your preferred API provider (Auto, Dhan, or ICICI)

6. Click "🔴 Load Live" to fetch data

## Troubleshooting

If you encounter issues with the APIs:

1. **Authentication Errors**: 
   - Ensure your API credentials are correct
   - For ICICI Breeze, remember that session tokens expire daily

2. **Connection Issues**:
   - Check your internet connection
   - Verify that the API services are online

3. **Rate Limiting**:
   - Be aware of API rate limits imposed by the providers
   - Space out your requests appropriately

4. **Symbol Format**:
   - Make sure you're using the correct symbol format for each API
   - For NSE symbols, try adding "NSE:" prefix if having issues

## API Configuration Dialog

For convenience, the application provides a built-in API configuration dialog:

1. Click "⚙️ Config" in the Live Data tab
2. Enter your API credentials
3. Click "Save to config.yaml"

This will update your config.yaml file with the new credentials. 