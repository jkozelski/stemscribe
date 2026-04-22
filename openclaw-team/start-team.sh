#!/bin/bash
# ============================================================
# StemScribe OpenClaw Agent Team - Master Startup Script
# ============================================================
# Usage:
#   ./start-team.sh setup     # First-time setup
#   ./start-team.sh start     # Start all agents
#   ./start-team.sh stop      # Stop all agents
#   ./start-team.sh status    # Check agent status
#   ./start-team.sh scout     # Start only Scout (research)
#   ./start-team.sh wordsmith # Start only Wordsmith (content)
#   ./start-team.sh megaphone # Start only Megaphone (social)
#   ./start-team.sh mailbot   # Start only Mailbot (email)
#   ./start-team.sh numbers   # Start only Numbers (analytics)
# ============================================================

set -e

TEAM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENCLAW_DIR="$HOME/.openclaw"
AGENTS=("research-agent" "marketing-agent" "community-agent" "email-agent" "analytics-agent")
AGENT_NAMES=("Scout" "Wordsmith" "Megaphone" "Mailbot" "Numbers")

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_banner() {
    echo -e "${BLUE}"
    echo "  ╔═══════════════════════════════════════════╗"
    echo "  ║     StemScribe AI Agent Team               ║"
    echo "  ║     Powered by OpenClaw                    ║"
    echo "  ╠═══════════════════════════════════════════╣"
    echo "  ║  Scout     → Research & Intelligence       ║"
    echo "  ║  Wordsmith → Content & SEO                 ║"
    echo "  ║  Megaphone → Social Media & Community      ║"
    echo "  ║  Mailbot   → Email Marketing               ║"
    echo "  ║  Numbers   → Analytics & Reporting         ║"
    echo "  ╚═══════════════════════════════════════════╝"
    echo -e "${NC}"
}

setup() {
    print_banner
    echo -e "${GREEN}>>> Setting up StemScribe Agent Team...${NC}"
    echo ""

    # Check if OpenClaw is installed
    if ! command -v openclaw &> /dev/null; then
        echo -e "${RED}OpenClaw is not installed.${NC}"
        echo "Install it first:"
        echo "  brew install openclaw"
        echo "  # or: git clone https://github.com/openclaw/openclaw.git"
        exit 1
    fi

    # Create OpenClaw directory structure if needed
    mkdir -p "$OPENCLAW_DIR/credentials"

    # Copy shared skills
    echo -e "${YELLOW}Copying shared skills...${NC}"
    mkdir -p "$OPENCLAW_DIR/skills"
    cp -r "$TEAM_DIR/shared-skills/"* "$OPENCLAW_DIR/skills/"

    # Copy shared workspace
    echo -e "${YELLOW}Setting up shared workspace...${NC}"
    mkdir -p "$OPENCLAW_DIR/shared-workspace"/{tasks,reports,content/{blog,social,email},memory}

    # Register each agent
    for i in "${!AGENTS[@]}"; do
        agent="${AGENTS[$i]}"
        name="${AGENT_NAMES[$i]}"
        echo -e "${GREEN}Registering agent: $name ($agent)${NC}"

        # Copy agent workspace
        if [ -d "$TEAM_DIR/$agent" ]; then
            mkdir -p "$OPENCLAW_DIR/$agent"
            cp -r "$TEAM_DIR/$agent/"* "$OPENCLAW_DIR/$agent/"

            # Symlink shared workspace into each agent's directory
            ln -sf "$OPENCLAW_DIR/shared-workspace" "$OPENCLAW_DIR/$agent/shared-workspace"

            # Register with OpenClaw
            openclaw agents add "$agent" --name "$name" 2>/dev/null || echo "  (already registered)"
        fi
    done

    # Check for API keys
    echo ""
    echo -e "${YELLOW}Checking API keys...${NC}"
    if [ ! -f "$OPENCLAW_DIR/credentials/anthropic" ]; then
        echo -e "${RED}  Missing: Anthropic API key${NC}"
        echo "  Run: echo 'YOUR_KEY' > $OPENCLAW_DIR/credentials/anthropic"
    else
        echo -e "${GREEN}  ✓ Anthropic API key found${NC}"
    fi

    if [ ! -f "$OPENCLAW_DIR/credentials/serper" ]; then
        echo -e "${YELLOW}  Missing: Serper API key (needed for web search)${NC}"
        echo "  Get free key at: https://serper.dev"
        echo "  Run: echo 'YOUR_KEY' > $OPENCLAW_DIR/credentials/serper"
    else
        echo -e "${GREEN}  ✓ Serper API key found${NC}"
    fi

    # Lock down credentials
    chmod 600 "$OPENCLAW_DIR/credentials/"* 2>/dev/null || true

    # Run doctor
    echo ""
    echo -e "${GREEN}Running OpenClaw diagnostics...${NC}"
    openclaw doctor --fix 2>/dev/null || echo "(doctor check complete)"

    echo ""
    echo -e "${GREEN}>>> Setup complete!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Add your API keys to $OPENCLAW_DIR/credentials/"
    echo "  2. Start the team: ./start-team.sh start"
    echo "  3. Or start one agent: ./start-team.sh scout"
}

start_all() {
    print_banner
    echo -e "${GREEN}>>> Starting all agents...${NC}"
    for i in "${!AGENTS[@]}"; do
        agent="${AGENTS[$i]}"
        name="${AGENT_NAMES[$i]}"
        echo -e "${BLUE}  Starting $name...${NC}"
        openclaw agents start "$agent" &
        sleep 2
    done
    echo ""
    echo -e "${GREEN}>>> All agents started!${NC}"
    echo "Use './start-team.sh status' to check on them."
}

start_single() {
    local agent_key="$1"
    local agent_dir=""
    local agent_name=""

    case "$agent_key" in
        scout)     agent_dir="research-agent";  agent_name="Scout" ;;
        wordsmith) agent_dir="marketing-agent"; agent_name="Wordsmith" ;;
        megaphone) agent_dir="community-agent"; agent_name="Megaphone" ;;
        mailbot)   agent_dir="email-agent";     agent_name="Mailbot" ;;
        numbers)   agent_dir="analytics-agent"; agent_name="Numbers" ;;
        *)
            echo "Unknown agent: $agent_key"
            echo "Available: scout, wordsmith, megaphone, mailbot, numbers"
            exit 1
            ;;
    esac

    echo -e "${GREEN}>>> Starting $agent_name ($agent_dir)...${NC}"
    openclaw agents start "$agent_dir"
}

stop_all() {
    echo -e "${YELLOW}>>> Stopping all agents...${NC}"
    for agent in "${AGENTS[@]}"; do
        openclaw agents stop "$agent" 2>/dev/null || true
    done
    echo -e "${GREEN}>>> All agents stopped.${NC}"
}

show_status() {
    print_banner
    echo -e "${BLUE}Agent Status:${NC}"
    echo ""
    openclaw agents list 2>/dev/null || echo "Run 'openclaw agents list' to check status"
    echo ""

    # Show recent tasks
    echo -e "${BLUE}Recent Tasks:${NC}"
    if [ -d "$OPENCLAW_DIR/shared-workspace/tasks" ]; then
        ls -lt "$OPENCLAW_DIR/shared-workspace/tasks/"*.json 2>/dev/null | head -5 || echo "  No tasks yet"
    fi
    echo ""

    # Show recent reports
    echo -e "${BLUE}Recent Reports:${NC}"
    if [ -d "$OPENCLAW_DIR/shared-workspace/reports" ]; then
        ls -lt "$OPENCLAW_DIR/shared-workspace/reports/"*.md 2>/dev/null | head -5 || echo "  No reports yet"
    fi
    echo ""

    # Show recent content
    echo -e "${BLUE}Recent Content:${NC}"
    if [ -d "$OPENCLAW_DIR/shared-workspace/content" ]; then
        find "$OPENCLAW_DIR/shared-workspace/content" -name "*.md" -mtime -7 2>/dev/null | head -5 || echo "  No content yet"
    fi
}

# Main command router
case "${1:-}" in
    setup)     setup ;;
    start)     start_all ;;
    stop)      stop_all ;;
    status)    show_status ;;
    scout)     start_single "scout" ;;
    wordsmith) start_single "wordsmith" ;;
    megaphone) start_single "megaphone" ;;
    mailbot)   start_single "mailbot" ;;
    numbers)   start_single "numbers" ;;
    *)
        print_banner
        echo "Usage: $0 {setup|start|stop|status|scout|wordsmith|megaphone|mailbot|numbers}"
        echo ""
        echo "Commands:"
        echo "  setup     - First-time setup (register agents, copy skills)"
        echo "  start     - Start all 5 agents"
        echo "  stop      - Stop all agents"
        echo "  status    - Show agent status and recent activity"
        echo "  scout     - Start only Scout (research)"
        echo "  wordsmith - Start only Wordsmith (content)"
        echo "  megaphone - Start only Megaphone (social)"
        echo "  mailbot   - Start only Mailbot (email)"
        echo "  numbers   - Start only Numbers (analytics)"
        ;;
esac
