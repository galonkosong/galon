#!/bin/bash

echo "Binance AI Trading Bot Setup (Linux)"
echo "=================================="

# Function to validate API key format
validate_api_key() {
    if [[ ! $1 =~ ^[A-Za-z0-9]{64}$ ]]; then
        echo "Error: Invalid API key format. API key should be 64 characters long and contain only letters and numbers."
        return 1
    fi
    return 0
}

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing required packages..."
pip install --upgrade pip
pip install -r requirements.txt

# Get API keys
echo "Please enter your Binance API credentials:"
read -p "API Key: " API_KEY
read -p "API Secret: " API_SECRET

# Validate API keys
if ! validate_api_key "$API_KEY"; then
    exit 1
fi
if ! validate_api_key "$API_SECRET"; then
    exit 1
fi

# Create environment file
echo "Creating environment file..."
cat > .env << EOL
BINANCE_API_KEY=${API_KEY}
BINANCE_API_SECRET=${API_SECRET}
EOL

# Set correct permissions
chmod 600 .env

# Create systemd service file for auto-start
echo "Creating systemd service file..."
sudo tee /etc/systemd/system/binance-bot.service << EOL
[Unit]
Description=Binance AI Trading Bot
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
EnvironmentFile=$(pwd)/.env
ExecStart=$(pwd)/venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOL

# Create start script
cat > start_bot.sh << EOL
#!/bin/bash
source venv/bin/activate
source .env
python main.py
EOL
chmod +x start_bot.sh

# Create monitoring script
cat > monitor_bot.sh << EOL
#!/bin/bash
echo "=== Binance AI Trading Bot Monitor ==="
echo "Last 50 lines of log:"
tail -n 50 logs/bot_output.log
echo
echo "System Status:"
systemctl status binance-bot
EOL
chmod +x monitor_bot.sh

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable binance-bot
sudo systemctl start binance-bot

echo
echo "Setup completed successfully!"
echo "The bot has been installed as a system service and will start automatically on boot"
echo
echo "Useful commands:"
echo "- Start bot: sudo systemctl start binance-bot"
echo "- Stop bot:  sudo systemctl stop binance-bot"
echo "- Check status: sudo systemctl status binance-bot"
echo "- View logs: tail -f logs/bot_output.log"
echo "- Monitor bot: ./monitor_bot.sh"
