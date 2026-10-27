#!/usr/bin/env bash
# quant-mcp-stack installer: TradeBlocks MCP + ML4T MCP + admin (shell) MCP behind Caddy/HTTPS on one Ubuntu 24.04 server.
#
#   curl -fsSL https://raw.githubusercontent.com/carstenf/quant-mcp-stack/main/install.sh | sudo SECRET=<random> bash
#
# Env:
#   SECRET        required, >= 24 chars [A-Za-z0-9]; part of every connector URL (the only access control!)
#   DOMAIN        optional, default <public-ip-with-dashes>.sslip.io (needs ports 80/443 open for Let's Encrypt)
#   WITH_ADMIN    1 (default) = install the admin shell MCP (root via sudo); 0 = skip
#   WITH_TORCH    1 (default) = CPU PyTorch for ml4t-models; 0 = skip (smaller, faster install)
#   REPO_REF      git ref to install (default main)
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

REPO=${REPO:-https://github.com/carstenf/quant-mcp-stack.git}
REPO_REF=${REPO_REF:-main}
WITH_ADMIN=${WITH_ADMIN:-1}
WITH_TORCH=${WITH_TORCH:-1}
TB_IMAGE=${TB_IMAGE:-romeo345/tradeblocks-mcp:4.5.0}
STACK=/opt/quant-mcp-stack
LOG=/var/log/quant-mcp-stack-install.log
exec > >(tee -a "$LOG") 2>&1

[[ $(id -u) -eq 0 ]] || { echo "run as root"; exit 1; }
[[ ${SECRET:-} =~ ^[A-Za-z0-9]{24,}$ ]] || { echo "SECRET missing or too short (>= 24 letters/digits)"; exit 1; }

echo "== [1/8] packages"
apt-get update -q
apt-get install -y -q ca-certificates curl git python3-venv python3-pip docker.io caddy ufw
if ! command -v node >/dev/null || [[ $(node -v | cut -d. -f1 | tr -d v) -lt 20 ]]; then
  curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
  apt-get install -y -q nodejs
fi
systemctl enable --now docker

echo "== [2/8] users and directories"
id quant >/dev/null 2>&1 || useradd -m -s /bin/bash quant
mkdir -p /srv/ml4t/{data,runs,workspace} /srv/tradeblocks
chown -R quant:quant /srv/ml4t /srv/tradeblocks
chmod 2775 /srv/tradeblocks
if [[ $WITH_ADMIN == 1 ]]; then
  id admin >/dev/null 2>&1 || useradd -m -s /bin/bash -G quant admin
  echo "admin ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-admin-mcp && chmod 440 /etc/sudoers.d/90-admin-mcp
fi

echo "== [3/8] code"
if [[ -d $STACK/.git ]]; then git -C $STACK fetch -q origin && git -C $STACK checkout -q "$REPO_REF" && git -C $STACK pull -q || true
else git clone -q --branch "$REPO_REF" "$REPO" $STACK; fi

echo "== [4/8] TradeBlocks MCP (Docker)"
docker pull -q "$TB_IMAGE"
docker rm -f tradeblocks >/dev/null 2>&1 || true
docker run -d --name tradeblocks --restart unless-stopped \
  --user "$(id -u quant):$(id -g quant)" -e HOME=/data \
  -p 127.0.0.1:3100:3100 -v /srv/tradeblocks:/data \
  --entrypoint node "$TB_IMAGE" server/index.js --http --no-auth /data

echo "== [5/8] ML4T MCP (Python venv, all ML4T libraries)"
python3 -m venv $STACK/.venv
$STACK/.venv/bin/pip install -q --upgrade pip
if [[ $WITH_TORCH == 1 ]]; then
  $STACK/.venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu
fi
$STACK/.venv/bin/pip install -q -r $STACK/ml4t-mcp/requirements.txt
cat > /etc/systemd/system/ml4t-mcp.service <<EOF
[Unit]
Description=ML4T MCP server (quant-mcp-stack)
After=network-online.target
Wants=network-online.target
[Service]
User=quant
Group=quant
WorkingDirectory=$STACK/ml4t-mcp
Environment=PYTHONUNBUFFERED=1
Environment=HOME=/home/quant
Environment=ML4T_MCP_HOME=/srv/ml4t
Environment=ML4T_DATA_PATH=/srv/ml4t/data/_cache
Environment=ML4T_MCP_STYLE_DIR=$STACK/ml4t-mcp
Environment=TRADEBLOCKS_DATA=/srv/tradeblocks
ExecStart=$STACK/.venv/bin/python $STACK/ml4t-mcp/server.py --host 127.0.0.1 --port 3250
Restart=on-failure
RestartSec=5
NoNewPrivileges=true
[Install]
WantedBy=multi-user.target
EOF

echo "== [6/8] admin MCP (Node)"
if [[ $WITH_ADMIN == 1 ]]; then
  (cd $STACK/admin-mcp && npm ci --omit=dev --silent)
  cat > /etc/systemd/system/admin-mcp.service <<EOF
[Unit]
Description=Admin shell MCP server (quant-mcp-stack)
After=network-online.target
[Service]
User=admin
WorkingDirectory=$STACK/admin-mcp
Environment=MCP_HOST=127.0.0.1
Environment=MCP_PORT=3001
Environment=MCP_SERVER_NAME=$(hostname)-admin
Environment=MCP_WORK_CWD=/home/admin
ExecStart=/usr/bin/node server.js
Restart=on-failure
RestartSec=5
[Install]
WantedBy=multi-user.target
EOF
fi
systemctl daemon-reload
systemctl enable --now ml4t-mcp
if [[ $WITH_ADMIN == 1 ]]; then systemctl enable --now admin-mcp; fi

echo "== [7/8] HTTPS (Caddy) + firewall"
if [[ -z ${DOMAIN:-} ]]; then
  IP=$(curl -fsS http://169.254.169.254/hetzner/v1/metadata/public-ipv4 2>/dev/null || curl -fsS https://api.ipify.org)
  DOMAIN="${IP//./-}.sslip.io"
fi
proxy() { # $1 path name, $2 port
  cat <<EOF
	handle_path /$SECRET/$1/* {
		reverse_proxy 127.0.0.1:$2 {
			header_up Host 127.0.0.1:$2
			flush_interval -1
		}
	}
EOF
}
{
  echo "$DOMAIN {"
  proxy ml4t 3250
  proxy tradeblocks 3100
  if [[ $WITH_ADMIN == 1 ]]; then proxy admin 3001; fi
  echo "	respond 404"
  echo "}"
} > /etc/caddy/Caddyfile
chmod 640 /etc/caddy/Caddyfile && chown root:caddy /etc/caddy/Caddyfile
systemctl restart caddy
ufw allow 22/tcp >/dev/null; ufw allow 80/tcp >/dev/null; ufw allow 443/tcp >/dev/null; ufw --force enable >/dev/null

echo "== [8/8] self-test (demo data + all recipes)"
sudo -u quant env HOME=/home/quant ML4T_MCP_HOME=/srv/ml4t ML4T_DATA_PATH=/srv/ml4t/data/_cache \
  $STACK/.venv/bin/python $STACK/ml4t-mcp/selftest.py || true

BASE="https://$DOMAIN/$SECRET"
{
  echo "Connector URLs (claude.ai -> Settings -> Connectors -> Add custom connector):"
  echo "  ML4T:        $BASE/ml4t/mcp"
  echo "  TradeBlocks: $BASE/tradeblocks/mcp"
  if [[ $WITH_ADMIN == 1 ]]; then echo "  Admin:       $BASE/admin/mcp"; fi
  echo "TradeBlocks data dir (CSV imports): /srv/tradeblocks"
} | tee /root/connectors.txt
echo "quant-mcp-stack installed."
