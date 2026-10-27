import express from 'express';
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/streamableHttp.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema
} from '@modelcontextprotocol/sdk/types.js';
import { exec, spawn } from 'child_process';
import { promisify } from 'util';
import {
  writeFileSync, readFileSync, mkdirSync, statSync, existsSync,
  chmodSync, renameSync, unlinkSync
} from 'fs';
import { dirname, basename, resolve as pathResolve } from 'path';
import { createHash, randomBytes } from 'crypto';
import { tmpdir } from 'os';

const execAsync = promisify(exec);
const app = express();
const PORT = parseInt(process.env.MCP_PORT || '3001', 10);
const SERVER_NAME = process.env.MCP_SERVER_NAME || 'mcp-server';
const SERVER_VERSION = '2.0.3';
const SUDO_PASSWORD = process.env.MCP_SUDO_PASSWORD || '';
const WORK_CWD = process.env.MCP_WORK_CWD || process.env.HOME || '/tmp';
const HOST = process.env.MCP_HOST || '127.0.0.1';
const EXEC_TIMEOUT_MS = parseInt(process.env.MCP_EXEC_TIMEOUT_MS || '60000', 10);
const EXEC_MAX_BUFFER = parseInt(process.env.MCP_EXEC_MAX_BUFFER || (50 * 1024 * 1024).toString(), 10);

app.use(express.json({ limit: '50mb' }));
app.use((req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Cache-Control');
  if (req.method === 'OPTIONS') return res.sendStatus(200);
  next();
});

// ------------------------------------------------------------------------
// Helpers
// ------------------------------------------------------------------------

/** Run a command with sudo. If MCP_SUDO_PASSWORD is set, uses `sudo -S` and pipes
 *  the password on stdin. Otherwise uses `sudo -n` (non-interactive) which assumes
 *  the service user has NOPASSWD in sudoers and fails clearly if not. Throws on non-zero exit. */
function sudoExec(cmd, args, { timeoutMs = 30000 } = {}) {
  return new Promise((resolve, reject) => {
    const usePassword = !!SUDO_PASSWORD;
    const sudoFlags = usePassword ? ['-S', '-p', ''] : ['-n'];
    const child = spawn('sudo', [...sudoFlags, cmd, ...args], {
      stdio: ['pipe', 'pipe', 'pipe']
    });
    let stdout = '';
    let stderr = '';
    const timer = setTimeout(() => {
      child.kill('SIGKILL');
      reject(new Error(`sudo ${cmd} timed out after ${timeoutMs}ms`));
    }, timeoutMs);
    child.stdout.on('data', d => { stdout += d.toString(); });
    child.stderr.on('data', d => { stderr += d.toString(); });
    child.on('error', err => { clearTimeout(timer); reject(err); });
    child.on('close', code => {
      clearTimeout(timer);
      if (code === 0) resolve({ stdout, stderr });
      else {
        const hint = !usePassword && /password is required|a password is required/i.test(stderr)
          ? ' (hint: service user lacks NOPASSWD in sudoers, and MCP_SUDO_PASSWORD is not set)'
          : '';
        reject(new Error(`sudo ${cmd} exited with ${code}${hint}: ${stderr.trim() || stdout.trim()}`));
      }
    });
    if (usePassword) {
      child.stdin.write(SUDO_PASSWORD + '\n');
    }
    child.stdin.end();
  });
}

/** Make a unique temp path under /tmp for staging system writes. */
function tmpStagingPath(target) {
  const id = randomBytes(8).toString('hex');
  return `${tmpdir()}/mcp-stage-${id}-${basename(target)}`;
}

/** Render the visible part of a file with 1-indexed line numbers (matches sandbox `view`). */
function renderWithLineNumbers(text, startLine = 1) {
  const lines = text.split('\n');
  const width = String(startLine + lines.length - 1).length;
  return lines
    .map((line, i) => `${String(startLine + i).padStart(width, ' ')}\t${line}`)
    .join('\n');
}

/** Best-effort detection of binary content. */
function looksBinary(buf) {
  const sample = buf.slice(0, Math.min(buf.length, 8192));
  for (const b of sample) {
    if (b === 0) return true;
  }
  return false;
}

/** Resolve a numeric uid/gid pair to "user:group" names. Falls back to "uid:gid" if lookup fails. */
function ownerString(uid, gid) {
  try {
    const passwd = readFileSync('/etc/passwd', 'utf-8');
    const group = readFileSync('/etc/group', 'utf-8');
    const userLine = passwd.split('\n').find(l => l.split(':')[2] === String(uid));
    const groupLine = group.split('\n').find(l => l.split(':')[2] === String(gid));
    const userName = userLine ? userLine.split(':')[0] : String(uid);
    const groupName = groupLine ? groupLine.split(':')[0] : String(gid);
    return `${userName}:${groupName}`;
  } catch {
    return `${uid}:${gid}`;
  }
}

// ------------------------------------------------------------------------
// MCP server + tool list
// ------------------------------------------------------------------------

function createServer() {
  const server = new Server(
    { name: SERVER_NAME, version: SERVER_VERSION },
    { capabilities: { tools: {} } }
  );

  server.setRequestHandler(ListToolsRequestSchema, async () => ({
    tools: [
      {
        name: 'exec',
        description: `Execute shell command on the server. Runs in ${WORK_CWD} by default.`,
        inputSchema: {
          type: 'object',
          properties: {
            command: { type: 'string', description: 'Command to execute' },
            cwd: { type: 'string', description: 'Optional working directory' },
            timeout_ms: { type: 'number', description: `Timeout in ms. Default: ${EXEC_TIMEOUT_MS}` },
            sudo: { type: 'boolean', description: 'If true, run the whole command as root via sudo (password piped from MCP_SUDO_PASSWORD). Use for commands needing root, e.g. rm/usermod/systemctl on system paths. Default: false' }
          },
          required: ['command']
        }
      },
      {
        name: 'write_file',
        description: 'Write a file owned by the MCP service user. Content is base64-encoded. Parent directories created as needed. Use append=true to append.',
        inputSchema: {
          type: 'object',
          properties: {
            path: { type: 'string', description: 'Absolute path to write the file to' },
            content_base64: { type: 'string', description: 'Base64-encoded file content' },
            append: { type: 'boolean', description: 'If true, append to existing file. Default: false' },
            mode: { type: 'string', description: 'Octal file mode, e.g. "0755". Default: 0644' }
          },
          required: ['path', 'content_base64']
        }
      },
      {
        name: 'write_file_sudo',
        description: 'Atomically write a system file requiring root. Stages content into /tmp, then sudo cp + chown + chmod. Avoids heredoc + sudo -S quoting bugs. Use this for /etc/, systemd unit files, etc.',
        inputSchema: {
          type: 'object',
          properties: {
            path: { type: 'string', description: 'Absolute destination path' },
            content_base64: { type: 'string', description: 'Base64-encoded file content' },
            owner: { type: 'string', description: 'Owner as "user:group". Default: "root:root"' },
            mode: { type: 'string', description: 'Octal mode, e.g. "0644". Default: 0644' },
            create_parents: { type: 'boolean', description: 'If true, run sudo mkdir -p on parent dir. Default: false' }
          },
          required: ['path', 'content_base64']
        }
      },
      {
        name: 'read_file',
        description: 'Read a file. Returns base64 content by default. Optional start_line/end_line return a text slice with 1-indexed line-number prefixes (display-only -- do not include the prefix when feeding back into str_replace).',
        inputSchema: {
          type: 'object',
          properties: {
            path: { type: 'string', description: 'Absolute path to the file to read' },
            max_bytes: { type: 'number', description: 'Maximum bytes to read. Default: 10485760 (10 MB)' },
            start_line: { type: 'number', description: 'Optional 1-indexed start line for text view' },
            end_line: { type: 'number', description: 'Optional 1-indexed end line (inclusive). Use -1 for end of file.' },
            with_line_numbers: { type: 'boolean', description: 'If true, return text with line-number prefixes (display-only). Default: true when start_line/end_line set, otherwise false.' }
          },
          required: ['path']
        }
      },
      {
        name: 'stat_file',
        description: 'Get file metadata (size, mtime, sha256 hash).',
        inputSchema: {
          type: 'object',
          properties: {
            path: { type: 'string', description: 'Absolute path to the file' },
            hash: { type: 'boolean', description: 'If true, compute sha256 hash. Default: false' }
          },
          required: ['path']
        }
      },
      {
        name: 'str_replace',
        description: 'Replace one unique occurrence of old_str with new_str in a text file. Fails if old_str is missing or appears more than once. Use sudo=true for system files (stages via /tmp + sudo cp). Do NOT include line-number prefixes from read_file output in old_str -- they are display-only.',
        inputSchema: {
          type: 'object',
          properties: {
            path: { type: 'string', description: 'Absolute path to the file to edit' },
            old_str: { type: 'string', description: 'Exact string to replace (must be unique in file)' },
            new_str: { type: 'string', description: 'Replacement string. Empty string deletes old_str.' },
            sudo: { type: 'boolean', description: 'If true, use sudo for the write-back (system files). Default: false' },
            owner: { type: 'string', description: 'When sudo=true: owner string for the result. Default: preserve existing owner via stat' },
            mode: { type: 'string', description: 'When sudo=true: file mode. Default: preserve existing mode' }
          },
          required: ['path', 'old_str', 'new_str']
        }
      },
      {
        name: 'view_image',
        description: 'Read an image file (png/jpg/jpeg/gif/webp) and return it as a rendered image block so it can be viewed directly, without a base64 round-trip into a sandbox. Use this whenever you need to SEE an image on this server (e.g. a generated report/chart PNG). For non-image files use read_file instead.',
        inputSchema: {
          type: 'object',
          properties: {
            path: { type: 'string', description: 'Absolute path to the image file' },
            max_dim: { type: 'number', description: 'Optional: downscale so the longest side is at most this many pixels (requires sharp; ignored if sharp is unavailable). Saves tokens for large images.' }
          },
          required: ['path']
        }
      }
    ]
  }));

  // ----------------------------------------------------------------------
  // Tool dispatcher
  // ----------------------------------------------------------------------

  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const { name, arguments: args } = request.params;
    const startedAt = Date.now();

    // Structured stderr log: one JSON line per tool call. Captured by systemd
    // journal (`journalctl -u mcp-server-v2.service`). Used for the 2026-05-22
    // to 2026-05-29 observation phase (MASTER.md §4.2 T3, MCP-LENOVO-OBSERVATIONS.md).
    const logCall = (outcome, extra = {}) => {
      const argSummary = {};
      if (args && typeof args === 'object') {
        for (const [k, v] of Object.entries(args)) {
          if (k === 'content_base64') argSummary[k] = `<${(v || '').length} chars b64>`;
          else if (k === 'command') argSummary[k] = String(v).slice(0, 200);
          else if (typeof v === 'string' && v.length > 200) argSummary[k] = v.slice(0, 200) + '...';
          else argSummary[k] = v;
        }
      }
      const entry = {
        ts: new Date().toISOString(),
        evt: 'tool_call',
        tool: name,
        args: argSummary,
        ms: Date.now() - startedAt,
        outcome,
        ...extra
      };
      process.stderr.write(JSON.stringify(entry) + '\n');
    };

    try {
      let result;
      switch (name) {
        case 'exec': result = await toolExec(args); break;
        case 'write_file': result = await toolWriteFile(args); break;
        case 'write_file_sudo': result = await toolWriteFileSudo(args); break;
        case 'read_file': result = await toolReadFile(args); break;
        case 'stat_file': result = await toolStatFile(args); break;
        case 'str_replace': result = await toolStrReplace(args); break;
        case 'view_image': result = await toolViewImage(args); break;
        default:
          logCall('unknown_tool');
          return { content: [{ type: 'text', text: `Unknown tool: ${name}` }], isError: true };
      }
      logCall('ok');
      return result;
    } catch (err) {
      const errMsg = err?.message || String(err);
      logCall('error', { err: errMsg.slice(0, 500) });
      return {
        content: [{ type: 'text', text: `Error: ${errMsg}` }],
        isError: true
      };
    }
  });

  return server;
}

// -----------------------------------------------------------------------
// Tool implementations
// ------------------------------------------------------------------------

async function toolExec(args) {
  const { command, cwd = WORK_CWD, timeout_ms = EXEC_TIMEOUT_MS, sudo = false } = args || {};
  if (!command) throw new Error('command is required');
  // sudo=true: run the whole command as root via `sudo -S bash -c`, piping the
  // password (MCP_SUDO_PASSWORD) on stdin. One sudo at the front instead of
  // scattered sudos inside the script. Falls back to `sudo -n` if no password set.
  if (sudo) {
    const usePassword = !!SUDO_PASSWORD;
    const sudoFlags = usePassword ? ['-S', '-p', ''] : ['-n'];
    return await new Promise((resolve, reject) => {
      const child = spawn('sudo', [...sudoFlags, '/bin/bash', '-c', command], {
        cwd, stdio: ['pipe', 'pipe', 'pipe']
      });
      let stdout = '';
      let stderr = '';
      const timer = setTimeout(() => {
        child.kill('SIGKILL');
        reject(new Error(`sudo exec timed out after ${timeout_ms}ms`));
      }, timeout_ms);
      child.stdout.on('data', d => { stdout += d.toString(); });
      child.stderr.on('data', d => { stderr += d.toString(); });
      child.on('error', err => { clearTimeout(timer); reject(err); });
      child.on('close', () => {
        clearTimeout(timer);
        // strip a leading empty sudo prompt line if any
        const cleanErr = stderr.replace(/^\s*\n/, '');
        const out = (stdout || '') + (cleanErr ? `\n[stderr]\n${cleanErr}` : '');
        resolve({ content: [{ type: 'text', text: out || '(no output)' }] });
      });
      if (usePassword) child.stdin.write(SUDO_PASSWORD + '\n');
      child.stdin.end();
    });
  }
  const { stdout, stderr } = await execAsync(command, {
    cwd, shell: '/bin/bash', timeout: timeout_ms, maxBuffer: EXEC_MAX_BUFFER
  });
  const out = (stdout || '') + (stderr ? `\n[stderr]\n${stderr}` : '');
  return { content: [{ type: 'text', text: out || '(no output)' }] };
}

async function toolWriteFile(args) {
  const { path, content_base64, append = false, mode = '0644' } = args || {};
  if (!path || !content_base64) throw new Error('path and content_base64 are required');
  const dir = dirname(path);
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
  const buf = Buffer.from(content_base64, 'base64');
  if (append && existsSync(path)) {
    const existing = readFileSync(path);
    writeFileSync(path, Buffer.concat([existing, buf]), { mode: parseInt(mode, 8) });
  } else {
    writeFileSync(path, buf, { mode: parseInt(mode, 8) });
  }
  const finalSize = statSync(path).size;
  return {
    content: [{
      type: 'text',
      text: `OK: ${append ? 'appended' : 'wrote'} ${buf.length} bytes to ${path} (file size now ${finalSize} bytes)`
    }]
  };
}

async function toolWriteFileSudo(args) {
  const {
    path: targetPath, content_base64, owner = 'root:root',
    mode = '0644', create_parents = false
  } = args || {};
  if (!targetPath || !content_base64) throw new Error('path and content_base64 are required');

  const staging = tmpStagingPath(targetPath);
  const buf = Buffer.from(content_base64, 'base64');
  writeFileSync(staging, buf, { mode: 0o644 });

  try {
    if (create_parents) {
      await sudoExec('mkdir', ['-p', dirname(targetPath)]);
    }
    await sudoExec('cp', [staging, targetPath]);
    await sudoExec('chown', [owner, targetPath]);
    await sudoExec('chmod', [mode, targetPath]);
    // Use sudo stat for size reporting since the target dir may not be readable
    // by the service user (e.g. writing into another user's $HOME).
    let sizeAfter = buf.length;
    try {
      const { stdout } = await sudoExec('stat', ['-c', '%s', targetPath]);
      sizeAfter = parseInt(stdout.trim(), 10) || buf.length;
    } catch { /* best effort; fall back to buf.length */ }
    return {
      content: [{
        type: 'text',
        text: `OK: wrote ${buf.length} bytes to ${targetPath} as ${owner} mode ${mode} (size now ${sizeAfter})`
      }]
    };
  } finally {
    try { unlinkSync(staging); } catch { /* best effort */ }
  }
}

async function toolReadFile(args) {
  const {
    path, max_bytes = 10 * 1024 * 1024,
    start_line, end_line, with_line_numbers
  } = args || {};
  if (!path) throw new Error('path is required');
  if (!existsSync(path)) throw new Error(`File not found: ${path}`);

  const lineRangeRequested = start_line !== undefined || end_line !== undefined;

  if (lineRangeRequested) {
    const raw = readFileSync(path, 'utf-8');
    if (looksBinary(Buffer.from(raw, 'utf-8'))) {
      throw new Error('Line-range view is not supported for binary files; omit start_line/end_line to get base64');
    }
    const allLines = raw.split('\n');
    const s = Math.max(1, start_line || 1);
    const e = end_line === undefined || end_line === -1
      ? allLines.length
      : Math.min(allLines.length, end_line);
    if (s > e) throw new Error(`start_line (${s}) must be <= end_line (${e})`);
    const slice = allLines.slice(s - 1, e).join('\n');
    const useNumbers = with_line_numbers !== false; // default true when range given
    const out = useNumbers ? renderWithLineNumbers(slice, s) : slice;
    return {
      content: [{
        type: 'text',
        text: out + `\n\n[lines ${s}-${e} of ${allLines.length}]`
      }]
    };
  }

  // Default: base64 of full file (capped)
  const buf = readFileSync(path);
  const capped = buf.length > max_bytes ? buf.subarray(0, max_bytes) : buf;
  return {
    content: [{
      type: 'text',
      text: JSON.stringify({
        path,
        size: buf.length,
        returned_bytes: capped.length,
        truncated: capped.length < buf.length,
        content_base64: capped.toString('base64')
      })
    }]
  };
}

const IMAGE_MIME = {
  png: 'image/png', jpg: 'image/jpeg', jpeg: 'image/jpeg',
  gif: 'image/gif', webp: 'image/webp'
};

async function toolViewImage(args) {
  const { path, max_dim } = args || {};
  if (!path) throw new Error('path is required');
  if (!existsSync(path)) throw new Error(`File not found: ${path}`);
  const ext = (path.split('.').pop() || '').toLowerCase();
  const mimeType = IMAGE_MIME[ext];
  if (!mimeType) throw new Error(`Unsupported image extension: .${ext} (use png/jpg/jpeg/gif/webp; for other files use read_file)`);

  let buf = readFileSync(path);
  // Optional downscale if sharp is available; silently skip otherwise.
  if (max_dim && Number(max_dim) > 0) {
    try {
      const sharp = (await import('sharp')).default;
      buf = await sharp(buf)
        .resize({ width: Number(max_dim), height: Number(max_dim), fit: 'inside', withoutEnlargement: true })
        .toBuffer();
    } catch { /* sharp not installed: return original bytes */ }
  }
  return {
    content: [{
      type: 'image',
      data: buf.toString('base64'),
      mimeType
    }]
  };
}

async function toolStatFile(args) {
  const { path, hash = false } = args || {};
  if (!path) throw new Error('path is required');
  if (!existsSync(path)) {
    return { content: [{ type: 'text', text: JSON.stringify({ path, exists: false }) }] };
  }
  const st = statSync(path);
  const out = {
    path,
    exists: true,
    size: st.size,
    mtime: st.mtime.toISOString(),
    mode: '0' + (st.mode & 0o777).toString(8),
    is_file: st.isFile(),
    is_dir: st.isDirectory()
  };
  if (hash && st.isFile()) {
    const data = readFileSync(path);
    out.sha256 = createHash('sha256').update(data).digest('hex');
  }
  return { content: [{ type: 'text', text: JSON.stringify(out, null, 2) }] };
}

async function toolStrReplace(args) {
  const {
    path, old_str, new_str = '',
    sudo = false, owner, mode
  } = args || {};
  if (!path) throw new Error('path is required');
  if (old_str === undefined || old_str === null) throw new Error('old_str is required');

  // Read the file. In sudo mode, use `sudo cat` so we can edit files in
  // directories where the service user lacks read access.
  let original;
  let sudoStUid = null, sudoStGid = null, sudoStMode = null;
  if (sudo) {
    try {
      await sudoExec('test', ['-f', path]);
    } catch {
      throw new Error(`File not found: ${path}`);
    }
    const { stdout: catOut } = await sudoExec('cat', [path]);
    original = catOut;
    try {
      const { stdout: statOut } = await sudoExec('stat', ['-c', '%u %g %a', path]);
      const [u, g, m] = statOut.trim().split(/\s+/);
      sudoStUid = parseInt(u, 10); sudoStGid = parseInt(g, 10); sudoStMode = m;
    } catch { /* best effort */ }
  } else {
    if (!existsSync(path)) throw new Error(`File not found: ${path}`);
    original = readFileSync(path, 'utf-8');
  }
  const matches = original.split(old_str).length - 1;
  if (matches === 0) throw new Error(`old_str not found in ${path}`);
  if (matches > 1) throw new Error(`old_str matched ${matches} times in ${path}; must be unique`);

  const updated = original.replace(old_str, new_str);

  if (!sudo) {
    writeFileSync(path, updated, 'utf-8');
    return {
      content: [{
        type: 'text',
        text: `OK: replaced 1 occurrence in ${path} (${old_str.length} -> ${new_str.length} chars)`
      }]
    };
  }

  // Sudo path: stage to /tmp, sudo cp, preserve or set owner/mode
  const finalOwner = owner ?? (sudoStUid !== null ? ownerString(sudoStUid, sudoStGid) : 'root:root');
  const finalMode = mode ?? (sudoStMode !== null ? '0' + sudoStMode : '0644');
  const staging = tmpStagingPath(path);
  writeFileSync(staging, updated, { mode: 0o644 });
  try {
    await sudoExec('cp', [staging, path]);
    await sudoExec('chown', [finalOwner, path]);
    await sudoExec('chmod', [finalMode, path]);
    return {
      content: [{
        type: 'text',
        text: `OK: replaced 1 occurrence in ${path} (sudo, owner=${finalOwner}, mode=${finalMode})`
      }]
    };
  } finally {
    try { unlinkSync(staging); } catch { /* best effort */ }
  }
}

// ------------------------------------------------------------------------
// HTTP / MCP transport
// ------------------------------------------------------------------------

const transports = new Map();

app.post('/mcp', async (req, res) => {
  const sessionId = req.headers['mcp-session-id'] || randomBytes(16).toString('hex');
  let transport = transports.get(sessionId);
  if (!transport) {
    transport = new StreamableHTTPServerTransport({ sessionIdGenerator: () => sessionId });
    const server = createServer();
    await server.connect(transport);
    transports.set(sessionId, transport);
    transport.onclose = () => { transports.delete(sessionId); };
  }
  await transport.handleRequest(req, res, req.body);
});

app.get('/mcp', async (req, res) => {
  const sessionId = req.headers['mcp-session-id'];
  const transport = transports.get(sessionId);
  if (!transport) return res.status(404).send('Session not found');
  await transport.handleRequest(req, res);
});

app.delete('/mcp', async (req, res) => {
  const sessionId = req.headers['mcp-session-id'];
  const transport = transports.get(sessionId);
  if (transport) {
    await transport.close();
    transports.delete(sessionId);
  }
  res.sendStatus(204);
});

app.get('/health', (req, res) => {
  res.json({
    name: SERVER_NAME,
    version: SERVER_VERSION,
    port: PORT,
    sudo_mode: SUDO_PASSWORD ? 'password' : 'nopasswd'
  });
});

app.listen(PORT, HOST, () => {
  console.log(`${SERVER_NAME} v${SERVER_VERSION} listening on ${HOST}:${PORT} (sudo-mode=${SUDO_PASSWORD ? 'password' : 'nopasswd'})`);
});
