#!/bin/bash
# StemScribe training heartbeat — sends Telegram status every 30 minutes
# Usage: ./heartbeat.sh &

BOT_TOKEN="8354408066:AAG0JcSOMqOckUzMnMU3wlFf2XGk-BX9CdM"
CHAT_ID="8434369404"

send_telegram() {
    local msg="$1"
    curl -s -X POST "https://api.telegram.org/bot${BOT_TOKEN}/sendMessage" \
        -H "Content-Type: application/json" \
        -d "{\"chat_id\": \"${CHAT_ID}\", \"text\": \"${msg}\", \"parse_mode\": \"Markdown\"}" > /dev/null
}

while true; do
    sleep 1800  # 30 minutes
    TIMESTAMP=$(date '+%I:%M %p')
    send_telegram "⏰ *StemScribe Heartbeat — ${TIMESTAMP}*

Claude Code is still working on chord detection improvements.

Check back or send a message for details."
done
