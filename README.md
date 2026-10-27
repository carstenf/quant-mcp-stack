# quant-mcp-stack

Three MCP servers on one small Ubuntu server, reachable from claude.ai (also on a phone) as custom connectors:

| Connector | What it is |
|---|---|
| **ML4T** | Equity backtesting and research on the [ML4T](https://github.com/ml4t) libraries (ml4t-backtest, -diagnostic, -data, -engineer, -models, -specs). Tools: data download (Yahoo, Fama-French), run store, charts, `run_python`, library search (`list_library_functions`, `describe_function`) and tested recipes (`get_recipe`). |
| **TradeBlocks** | [TradeBlocks](https://github.com/tradeblocks-org/tradeblocks) MCP (Docker image `romeo345/tradeblocks-mcp`): statistics, edge decay, walk-forward, Monte Carlo, stress tests on trade logs. |
| **Admin** (optional) | Shell / file tools with sudo, so Claude can administer the server. |

## Install (phone only, ~15 min)

1. Create a server at Hetzner Cloud (app or console): Ubuntu 24.04, type CX22 or larger, public IPv4.
2. Paste this into **Cloud config** (replace the secret with 32 random letters/digits):

   ```yaml
   #cloud-config
   runcmd:
     - curl -fsSL https://raw.githubusercontent.com/carstenf/quant-mcp-stack/main/install.sh | SECRET=REPLACE_WITH_32_RANDOM_CHARS bash
   ```
3. Wait ~15 minutes after the server is up. The connector URLs are (IP with dashes):
   - `https://1-2-3-4.sslip.io/<SECRET>/ml4t/mcp`
   - `https://1-2-3-4.sslip.io/<SECRET>/tradeblocks/mcp`
   - `https://1-2-3-4.sslip.io/<SECRET>/admin/mcp`
4. claude.ai → Settings → Connectors → Add custom connector, once per URL. Ask Claude to call `info` on ML4T.

Options (env in front of `bash`): `WITH_ADMIN=0` (no shell connector), `WITH_TORCH=0` (no PyTorch), `DOMAIN=my.domain` (own domain instead of sslip.io).

## Security

The secret path is the only access control. Anyone with a URL can run code (`run_python`) and, with the admin
connector, act as root. Use a dedicated server without other data, never share the URLs, and rebuild the server
(new secret) if a URL leaks. TradeBlocks and the MCP servers listen on 127.0.0.1 only; Caddy terminates HTTPS.

## Layout on the server

- `/opt/quant-mcp-stack` code + Python venv, `/srv/ml4t/{data,runs,workspace}` ML4T datasets and runs
- `/srv/tradeblocks` TradeBlocks data dir (write CSVs here for `import_csv`; also `$TRADEBLOCKS_DATA` inside `run_python`)
- services: `ml4t-mcp`, `admin-mcp`, `caddy`, Docker container `tradeblocks`
- install log `/var/log/quant-mcp-stack-install.log`, self-test `/srv/ml4t/selftest.json`, URLs `/root/connectors.txt`

Update: re-run the install command (keeps data).

## License

MIT for the code in this repo. ML4T libraries (MIT) and TradeBlocks (MIT) are installed from their own sources.
