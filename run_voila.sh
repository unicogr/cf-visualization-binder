#!/bin/bash

# Script to run Voila server for surfCFviz.ipynb
# Activates pyenv environment, starts Voila in background, logs output, and saves PID
# bash run_voila.sh to ge the app served, then follow it by: tail -f voila.log

# nicolas.gravel (at) cea.fr

# Change to the script's directory so it can be run from anywhere
ngravel_results="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ngravel_results"

# Initialize pyenv (if not already done in shell)
eval "$(pyenv init -)"

# Activate the retMapping environment
pyenv activate retMapping

# Check if activation succeeded
if [ $? -ne 0 ]; then
    echo "Failed to activate pyenv environment 'retMapping'"
    exit 1
fi

# Run Voila in background with nohup to prevent termination on terminal close
# Logs output to voila.log
NOTEBOOK_PATH="$ngravel_results/surfCFviz.ipynb"
nohup voila --port=8866 --Voila.ip=0.0.0.0 --no-browser "$NOTEBOOK_PATH" > "$ngravel_results/voila.log" 2>&1 &

# Save the PID to a file
echo $! > "$ngravel_results/voila.pid"

echo "Voila started with PID $(cat "$ngravel_results/voila.pid")"
echo "Output logged to $ngravel_results/voila.log"
echo "To view logs in real-time: tail -f $ngravel_results/voila.log"
echo "To stop: kill \$(cat $ngravel_results/voila.pid)"

# Optionally tail the log (uncomment if you want the script to follow logs)
# tail -f voila.log

# Future improvements: Set up Nginx reverse proxy for elegant URL (e.g., http://nautilus/nicoViz/)
# 1. Install Nginx: sudo apt update && sudo apt install nginx
# 2. Create config file: sudo nano /etc/nginx/sites-available/nautilus_proxy
#    Add the following content:
#    server {
#        listen 80;
#        server_name nautilus;  # Or your machine's hostname/IP
#
#        # Proxy requests to /nicoViz/ to Voila on port 8866
#        location /nicoViz/ {
#            proxy_pass http://localhost:8866/;
#            proxy_set_header Host $host;
#            proxy_set_header X-Real-IP $remote_addr;
#            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
#            proxy_set_header X-Forwarded-Proto $scheme;
#
#            # Optional: Handle WebSocket connections (for interactive notebooks)
#            proxy_http_version 1.1;
#            proxy_set_header Upgrade $http_upgrade;
#            proxy_set_header Connection "upgrade";
#        }
#    }
# 3. Enable site: sudo ln -s /etc/nginx/sites-available/nautilus_proxy /etc/nginx/sites-enabled/
# 4. Test and reload: sudo nginx -t && sudo systemctl reload nginx
# 5. Update Voila command to: nohup voila --port=8866 --Voila.ip=0.0.0.0 --no-browser --base_url=/nicoViz surfCFviz.ipynb > voila.log 2>&1 &
# 6. Access at: http://nautilus/nicoViz/