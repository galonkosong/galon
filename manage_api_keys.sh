#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${GREEN}Binance API Key Manager${NC}"
echo "====================="

# Function to validate API key format
validate_api_key() {
    if [[ ! $1 =~ ^[A-Za-z0-9]{64}$ ]]; then
        echo -e "${RED}Error: Invalid API key format. API key should be 64 characters long and contain only letters and numbers.${NC}"
        return 1
    fi
    return 0
}

# Function to update API keys
update_api_keys() {
    local api_key=$1
    local api_secret=$2
    
    # Update .env file
    echo "BINANCE_API_KEY=$api_key" > .env
    echo "BINANCE_API_SECRET=$api_secret" >> .env
    chmod 600 .env
    
    # Update systemd service if it exists
    if [ -d "/etc/systemd/system" ]; then
        sudo mkdir -p /etc/systemd/system/binance-bot.service.d
        echo "[Service]" | sudo tee /etc/systemd/system/binance-bot.service.d/override.conf > /dev/null
        echo "Environment=BINANCE_API_KEY=$api_key" | sudo tee -a /etc/systemd/system/binance-bot.service.d/override.conf > /dev/null
        echo "Environment=BINANCE_API_SECRET=$api_secret" | sudo tee -a /etc/systemd/system/binance-bot.service.d/override.conf > /dev/null
        
        # Reload systemd
        sudo systemctl daemon-reload
        
        # Restart bot service if it's running
        if systemctl is-active --quiet binance-bot; then
            echo "Restarting bot service..."
            sudo systemctl restart binance-bot
        fi
    fi
}

# Main menu
while true; do
    echo
    echo "1. Update API Keys"
    echo "2. Check Current API Keys"
    echo "3. Test API Connection"
    echo "4. Exit"
    echo
    read -p "Choose an option (1-4): " choice

    case $choice in
        1)
            echo
            read -p "Enter API Key: " api_key
            read -p "Enter API Secret: " api_secret
            
            if validate_api_key "$api_key" && validate_api_key "$api_secret"; then
                update_api_keys "$api_key" "$api_secret"
                echo -e "${GREEN}API keys updated successfully!${NC}"
            fi
            ;;
        2)
            echo
            if [ -f .env ]; then
                echo "Current API Keys (from .env file):"
                echo "--------------------------------"
                cat .env | sed 's/=.*$/=****/'
            else
                echo -e "${RED}No .env file found${NC}"
            fi
            ;;
        3)
            echo
            echo "Testing API connection..."
            source .env
            python3 -c "
from binance.client import Client
import os

try:
    client = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'))
    client.get_account()
    print('\033[92mAPI Connection successful!\033[0m')
except Exception as e:
    print('\033[91mAPI Connection failed:', str(e), '\033[0m')
"
            ;;
        4)
            echo "Exiting..."
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid option${NC}"
            ;;
    esac
done
