#!/bin/bash
# ================================================
# StemScribe AI Agent Teams - Runner Script
# ================================================
# Usage:
#   ./run_agents.sh research    # Run research team only
#   ./run_agents.sh marketing   # Run marketing team only
#   ./run_agents.sh both        # Run both teams
#   ./run_agents.sh setup       # Initial setup
# ================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

case "${1:-both}" in
    setup)
        echo ">>> Setting up StemScribe AI Agent Teams..."

        # Create virtual environment
        python3 -m venv venv
        source venv/bin/activate

        # Install dependencies
        pip install -r ../requirements.txt

        # Create output directories
        mkdir -p reports content

        # Check for .env
        if [ ! -f .env ]; then
            cp .env.example .env
            echo ""
            echo ">>> Created .env file. Please edit it with your API keys:"
            echo "    $SCRIPT_DIR/.env"
            echo ""
            echo "Required keys:"
            echo "  - ANTHROPIC_API_KEY (for Claude LLM)"
            echo "  - SERPER_API_KEY (for web search - free at serper.dev)"
            echo ""
        fi

        echo ">>> Setup complete!"
        ;;

    research)
        echo ">>> Launching Research Team..."
        source venv/bin/activate 2>/dev/null || true
        python3 agents.py --team research
        echo ""
        echo ">>> Reports saved to: $SCRIPT_DIR/reports/"
        ls -la reports/
        ;;

    marketing)
        echo ">>> Launching Marketing Team..."
        source venv/bin/activate 2>/dev/null || true
        python3 agents.py --team marketing
        echo ""
        echo ">>> Content saved to: $SCRIPT_DIR/content/"
        ls -la content/
        ;;

    both)
        echo ">>> Launching Both Teams..."
        source venv/bin/activate 2>/dev/null || true
        python3 agents.py --team both
        echo ""
        echo ">>> All outputs:"
        echo "Reports: $SCRIPT_DIR/reports/"
        echo "Content: $SCRIPT_DIR/content/"
        ;;

    *)
        echo "Usage: $0 {setup|research|marketing|both}"
        exit 1
        ;;
esac
