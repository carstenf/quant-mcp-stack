# SETUP.md — instructions for the AI assistant

> **To the human:** give this file to your AI assistant (e.g. attach it in claude.ai) and say
> "Set this up for me." It will guide you step by step. Everything works from a phone.
> You need: a Hetzner Cloud account (or another provider with cloud-init), ~5 EUR/month, ~30 minutes.

---

## To the assistant: your job

Guide the user to a working **quant-mcp-stack** server and connect it to claude.ai as custom connectors:

- **ML4T** MCP: equity backtesting and research on the ML4T libraries
- **TradeBlocks** MCP: robustness analysis of trade logs (edge decay, walk-forward, Monte Carlo, stress)
- **Admin** MCP (optional): shell + file tools with sudo, so you can administer the server later

Then install the **strategy-due-diligence** skill. Work in the user's language, one step at a time, short
messages, wait for confirmation after each step. The user may only have a phone: never require a PC, SSH or a
terminal on their side.

Source: https://github.com/carstenf/quant-mcp-stack (read `README.md` and `install.sh` if you need details).

### Step 0 — explain and ask (one message)

Tell the user briefly what will be created and the security model, then ask:
1. Admin connector yes/no? (default yes; it gives root on the server to whoever has the URL)
2. PyTorch yes/no? (default yes; only needed for the deep-learning models in ml4t-models)

Security model to state plainly: the only protection is a long secret in the URLs. Anyone with a URL can run code
on the server. Use a dedicated server with no other data, never share the URLs, rebuild if a URL leaks.

### Step 1 — generate the secret

Generate a random string of exactly 32 letters/digits (use a code tool if you have one; otherwise compose one
yourself and do not reuse examples). Show it once and tell the user to keep it private.

### Step 2 — create the server (user does this in the Hetzner app or console.hetzner.cloud)

Give these exact choices:
- **Location:** any (Nuremberg / Falkenstein / Helsinki are fine)
- **Image:** Ubuntu 24.04
- **Type:** Shared vCPU **x86** (not Arm), **CX22** or larger (CX32 if PyTorch and larger datasets)
- **Networking:** public IPv4 **on**
- **SSH key:** optional (without one Hetzner emails a root password; not needed for this setup)
- **Cloud config:** paste exactly (fill in SECRET, set the two flags from step 0):

```yaml
#cloud-config
runcmd:
  - curl -fsSL https://raw.githubusercontent.com/carstenf/quant-mcp-stack/main/install.sh | SECRET=<SECRET> WITH_ADMIN=1 WITH_TORCH=1 bash
```

- **Name:** anything, e.g. `quant1`. Then **Create & Buy**.

Ask the user to send you the server's **public IPv4** (shown in the server list).

### Step 3 — wait for the installation (~15-20 min)

Build the URLs: `HOST = <IPv4 with dots replaced by dashes>.sslip.io`, e.g. `1.2.3.4` -> `1-2-3-4.sslip.io`.

- ML4T: `https://HOST/<SECRET>/ml4t/mcp`
- TradeBlocks: `https://HOST/<SECRET>/tradeblocks/mcp`
- Admin: `https://HOST/<SECRET>/admin/mcp` (only if chosen)

If you can fetch URLs: `https://HOST/` answers 404 once Caddy runs (that is expected). Otherwise ask the user to
wait 20 minutes.

### Step 4 — add the connectors (user, in claude.ai)

claude.ai -> **Settings -> Connectors -> Add custom connector**. One connector per URL, names e.g. `ML4T`,
`TradeBlocks`, `Server Admin`. No OAuth fields needed. Then the user enables them in a chat (tools menu).
Connectors appear in new chats; tell the user to start a new chat if tools do not show up.

### Step 5 — verify (in a chat where the connectors are enabled)

1. ML4T `info` -> versions of ml4t-backtest/-diagnostic/-data/-engineer/-models/-specs.
2. ML4T `list_datasets` -> demo data `prices/test-largecap40`, `prices/spy`, `factors/ff5_daily`,
   `signals/test_mom_12_1` (written by the install self-test).
3. ML4T `run_python` with `print(open('/srv/ml4t/selftest.json').read())` -> `"all_ok": true`.
4. TradeBlocks `list_blocks` -> answers (empty list is fine).
5. Admin (if installed) `exec` with `systemctl is-active ml4t-mcp caddy docker` -> three times `active`.

Troubleshooting (use the admin connector if available, else ask the user to rebuild the server):
- Install log: `/var/log/quant-mcp-stack-install.log`; URLs: `/root/connectors.txt`
- `systemctl status ml4t-mcp admin-mcp caddy`, `docker ps`, `docker logs tradeblocks`, `journalctl -u caddy -n 50`
- HTTPS certificate fails -> ports 80/443 blocked by a provider firewall; open them.
- Re-run / update: `curl -fsSL https://raw.githubusercontent.com/carstenf/quant-mcp-stack/main/install.sh | sudo SECRET=<SECRET> bash` (keeps data)

### Step 6 — install the strategy-due-diligence skill

The skill runs a full due diligence of a stock-picking strategy from a pick list (ML4T backtest, ml4t-diagnostic
statistics, TradeBlocks robustness, PDF report). It needs exactly the two connectors set up above.

1. User opens on the phone: https://github.com/carstenf/quant-mcp-stack/raw/main/skills/strategy-due-diligence.zip
   -> download -> save to Files. Do **not** open/unzip it.
2. claude.ai -> **Settings -> Capabilities -> Skills** (or claude.ai/customize/skills) -> **+** -> **Upload a skill**
   -> choose `strategy-due-diligence.zip`.
3. Test: in a new chat with both connectors enabled, attach a pick list (Excel/CSV with week, ticker, rank) and
   say "Run the strategy-due-diligence skill on this list."

Done. Summarise for the user: the three URLs (tell them to store them privately), what each connector does, and
that the server costs money until it is deleted in the Hetzner console.
