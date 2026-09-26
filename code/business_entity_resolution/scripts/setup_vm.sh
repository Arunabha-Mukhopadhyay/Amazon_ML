#!/usr/bin/env bash
# One-time setup of an Ubuntu 24.04 VM for this project.
#   bash scripts/setup_vm.sh
# Installs system packages, creates .venv with the pinned requirements and
# installs the idle auto-shutdown (power off after 30 idle minutes).
set -euo pipefail
cd "$(dirname "$0")/.."

sudo apt-get update -y
sudo apt-get install -y python3-venv python3-pip tmux htop pigz libgomp1

python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt
.venv/bin/python -c "import lightgbm, rapidfuzz, sparse_dot_topn, anyascii; print('python deps OK')"

sudo install -m 755 scripts/idle_shutdown.sh /usr/local/bin/er_idle_shutdown.sh
echo "*/5 * * * * root /usr/local/bin/er_idle_shutdown.sh" | sudo tee /etc/cron.d/er_idle_shutdown >/dev/null
echo "idle auto-shutdown installed (30 minutes)"
