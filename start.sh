#!/bin/bash

# Create Streamlit config directory
mkdir -p ~/.streamlit

# Create Streamlit config file
cat > ~/.streamlit/config.toml << EOF
[server]
port = 8080
address = "0.0.0.0"
headless = true
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"

[logger]
level = "info"
EOF

# Start the Streamlit application
exec streamlit run app_choice.py --server.port=8080 --server.address=0.0.0.0 --server.headless=true
