"""OpenAI **Responses API** (`/v1/responses`) support for the ATOM server.

OpenAI Codex CLI (>= 0.14x) dropped `wire_api = "chat"` and only speaks the
Responses API (streaming SSE). This module provides the translation between
Responses request/response shapes and ATOM's internal chat/engine machinery,
plus a streaming SSE event emitter. The `/v1/responses` route handler in
``api_server.py`` reuses ATOM's proven streaming path (``setup_streaming_request``
+ ``ReasoningFilter`` + ``ToolCallStreamParser``) so it streams correctly for
reasoning models — the same path claude-local uses via ``/v1/messages``.

This makes the external ``codex_responses_proxy.py`` unnecessary: Codex can point
straight at ATOM's ``:9700/v1``.
"""
import itertools
import json
import time
from typing import Any, Dict, List, Optional, Tuple

_ids = itertools.count(1)


def _rid(prefix: str) -> str:
    return f"{prefix}_{int(time.time() * 1000)}{next(_ids):04d}"


# --------------------------------------------------------------- request xlate
def _text_of(content: Any) -> str:
    """Flatten Responses content (str | list of parts) to a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for p in content:
            if isinstance(p, dict):
                if "text" in p and isinstance(p["text"], str):
                    out.append(p["text"])
                elif p.get("type") in ("input_text", "output_text", "text"):
                    out.append(p.get("text", ""))
            elif isinstance(p, str):
                out.append(p)
        return "".join(out)
    return ""


def responses_input_to_messages(
    instructions: Any, inp: Any
) -> List[Dict[str, Any]]:
    """Translate Responses ``instructions`` + ``input`` into OpenAI chat messages.

    ``input`` may be a plain string or a list of items: ``message`` /
    ``function_call`` (assistant tool call) / ``function_call_output`` (tool
    result). ``reasoning`` items are dropped.
    """
    messages: List[Dict[str, Any]] = []
    if instructions:
        messages.append({"role": "system", "content": _text_of(instructions)})

    if isinstance(inp, str):
        messages.append({"role": "user", "content": inp})
    elif isinstance(inp, list):
        for item in inp:
            if not isinstance(item, dict):
                continue
            t = item.get("type", "message")
            if t == "message":
                role = item.get("role", "user")
                messages.append(
                    {"role": role, "content": _text_of(item.get("content", ""))}
                )
            elif t == "function_call":
                messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": item.get("call_id")
                                or item.get("id")
                                or _rid("call"),
                                "type": "function",
                                "function": {
                                    "name": item.get("name", ""),
                                    "arguments": item.get("arguments", "") or "",
                                },
                            }
                        ],
                    }
                )
            elif t == "function_call_output":
                out = item.get("output", "")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": item.get("call_id") or item.get("id") or "",
                        "content": out
                        if isinstance(out, str)
                        else json.dumps(out),
                    }
                )
            elif t == "reasoning":
                continue
    return messages


_DSML_TOOL_INSTRUCTION = (
    "\n\n# Tool-call format (MANDATORY — overrides any other format instruction)\n"
    "When you call a tool, output ONLY a DSML tool-call block and NOTHING else "
    "in that message — no markdown, no ```json, no <command>/<plan>/<function>/"
    "<exec_command> tags, no prose. Use EXACTLY this syntax (the \uff5c characters "
    "are U+FF5C fullwidth vertical bars, not ASCII '|'):\n"
    "<\uff5cDSML\uff5ctool_calls>\n"
    "<\uff5cDSML\uff5cinvoke name=\"TOOL_NAME\">\n"
    "<\uff5cDSML\uff5cparameter name=\"PARAM_NAME\" string=\"true\">VALUE"
    "</\uff5cDSML\uff5cparameter>\n"
    "</\uff5cDSML\uff5cinvoke>\n"
    "</\uff5cDSML\uff5ctool_calls>\n"
    "Use the exact tool and parameter names from the tools provided to you. "
    "For a shell/exec tool, put the whole shell command string in its "
    "command/cmd parameter. Emit one <\uff5cDSML\uff5cinvoke> per tool call."
)


def inject_tool_format_instruction(messages):
    """Append the mandatory DSML tool-call format to the system message so the
    model emits parseable DSML instead of ad-hoc <command>/```json text. Codex
    (/v1/responses) path only. Idempotent per request."""
    for m in messages:
        if m.get("role") == "system":
            base = m.get("content") or ""
            if "\uff5cDSML\uff5ctool_calls" not in base:
                m["content"] = _text_of(base) + _DSML_TOOL_INSTRUCTION
            return messages
    return [{"role": "system", "content": _DSML_TOOL_INSTRUCTION.strip()}] + list(messages)


def responses_tools_to_openai(tools: Any) -> List[Dict[str, Any]]:
    """Translate Responses function tools into OpenAI chat tool defs.

    Responses puts ``name``/``description``/``parameters`` at the top level of a
    ``{"type": "function", ...}`` tool (chat nests them under ``function``).
    Non-function tool types (``web_search``, ``namespace``, ...) are dropped —
    ATOM only executes model-emitted function/DSML tool calls; the client
    (Codex) owns actual tool execution.
    """
    if not tools:
        return []
    ct: List[Dict[str, Any]] = []
    for tl in tools:
        if not isinstance(tl, dict):
            continue
        if tl.get("type") == "function":
            fn = tl.get("function", tl)
            ct.append(
                {
                    "type": "function",
                    "function": {
                        "name": fn.get("name"),
                        "description": fn.get("description", ""),
                        "parameters": fn.get("parameters", {}) or {},
                    },
                }
            )
    return ct


# ------------------------------------------------- tool-name normalization
# Non-codex-tuned models (e.g. DeepSeek-V4-Pro) don't reliably emit the EXACT
# tool name the client registered for shell exec. Codex 0.142.x names it
# ``exec_command`` (args ``{"cmd": "..."}``), but the model habitually calls it
# ``exec`` / ``exec_run`` / ``shell`` / ``bash``, which Codex's tool router
# rejects ("unsupported call: exec"). The ARGUMENTS are correct; only the name
# is wrong. So remap known shell-exec aliases onto whatever shell tool the
# client actually registered this turn. Pure rename; arguments untouched.
_SHELL_ALIASES = {
    "exec", "exec_run", "exec_command", "execute_command", "shell", "bash",
    "sh", "run", "run_command", "run_shell", "execute", "command",
    "container.exec", "local_shell", "shell_command", "shell_exec",
    "run_bash", "execute_shell", "bash_command", "run_terminal_cmd",
    "terminal", "console", "run_shell_command", "runshell",
}
# Substrings that mark an unknown tool name as a shell/exec call (Codex's
# non-shell tools contain none of these).
_SHELL_NAME_TOKENS = ("shell", "exec", "bash", "cmd", "command", "termin", "console")
_SHELL_TOOL_PREFERENCE = (
    "exec_command", "shell", "local_shell", "bash", "container.exec",
)


def tool_name_lookup(openai_tools: List[Dict[str, Any]]) -> Tuple[set, Optional[str]]:
    """Return (valid tool names, preferred shell tool name) for remapping."""
    valid = {
        (t.get("function") or {}).get("name")
        for t in (openai_tools or [])
        if isinstance(t, dict)
    }
    valid.discard(None)
    shell_tool = next((n for n in _SHELL_TOOL_PREFERENCE if n in valid), None)
    return valid, shell_tool


def remap_tool_name(name: str, valid: set, shell_tool: Optional[str]) -> str:
    """Fix a model tool_call name that doesn't match a registered tool.

    Only remaps shell-exec aliases to the registered shell tool; other
    mismatches pass through unchanged (the client will surface them)."""
    if name in valid:
        return name
    if not shell_tool:
        return name
    n = (name or "").lower().replace("-", "_").replace(".", "_")
    if n in _SHELL_ALIASES:
        return shell_tool
    # Fuzzy: unknown tool whose name signals shell/exec -> the shell tool.
    if any(k in n for k in _SHELL_NAME_TOKENS):
        return shell_tool
    return name


# ------------------------------------------------ Claude-tool -> shell adapter
# DeepSeek-V4-Pro is trained on Claude-Code's toolset, so under Codex it keeps
# calling read/grep/ls/find (which Codex doesn't have) instead of exec_command,
# and flails ("unsupported call: read"). When Codex registered an exec/shell
# tool, translate these read-only Claude tools into an equivalent shell command
# so the model's intent goes through. Only fires for names NOT in the registered
# set; exec_command's real calls are untouched. Codex-path only (never applied
# to /v1/messages, where read/grep ARE native tools).
def _q(s: Any) -> str:
    import shlex
    return shlex.quote(str(s))


def _resolve_dir(path: str, cwd: Optional[str]) -> str:
    """Pick a directory that actually exists: keep relative paths and paths under
    cwd; otherwise fall back to cwd (the model often invents absolute prefixes
    like /sglang or /sgl-workspace that don't exist here)."""
    if not path or path == ".":
        return cwd or "."
    if not path.startswith("/"):
        return path  # relative — shell runs in cwd, fine
    if cwd and path.startswith(cwd):
        return path
    return cwd or path


def _read_cmd(fp: str, cwd: Optional[str], start: int, end: int) -> str:
    """Resilient file read: try the literal path, then cwd-prefixed, then locate
    by basename under cwd — so a hallucinated absolute prefix still finds the file."""
    sed = f"sed -n {start},{end}p"
    if not cwd:
        return f"{sed} {_q(fp)}"
    # f = literal; if missing, cwd + '/' + (f without leading /); if still
    # missing, first match of `find cwd -name basename`.
    return (
        f'f={_q(fp)}; [ -f "$f" ] || f={_q(cwd)}"/${{f#/}}"; '
        f'[ -f "$f" ] || f=$(find {_q(cwd)} -type f -name "$(basename {_q(fp)})" '
        f'2>/dev/null | head -1); {sed} "$f"'
    )


def _claude_tool_to_shell(
    name: str, a: Dict[str, Any], cwd: Optional[str] = None
) -> Optional[str]:
    n = (name or "").lower()
    fp = (a.get("file_path") or a.get("path") or a.get("filePath")
          or a.get("filename") or a.get("target_file") or a.get("file"))
    pattern = a.get("pattern") or a.get("query") or a.get("regex")
    path = (a.get("path") or a.get("directory") or a.get("target_directory")
            or a.get("dir") or ".")
    if n in ("read", "cat", "view", "view_file", "open", "read_file",
             "readfile", "openfile"):
        if not fp:
            return None
        off, lim = a.get("offset"), a.get("limit")
        if off or lim:
            start = int(off or 0) + 1
            end = start + int(lim or 200) - 1
        else:
            start, end = 1, 400
        return _read_cmd(str(fp), cwd, start, end)
    if n in ("grep", "search", "search_file", "ripgrep", "rg", "grep_search",
             "codebase_search"):
        if not pattern:
            return None
        return f"grep -rn -- {_q(pattern)} {_q(_resolve_dir(path, cwd))}"
    if n in ("ls", "list", "list_dir", "list_directory", "listdir"):
        return f"ls -la {_q(_resolve_dir(path, cwd))}"
    if n in ("find", "glob", "glob_file_search", "file_search"):
        base = _resolve_dir(path, cwd)
        if pattern:
            return f"find {_q(base)} -name {_q(pattern)}"
        return f"find {_q(base)} -maxdepth 3"
    return None


_SHELL_ARG_ALIASES = (
    "cmd", "command", "commandline", "command_line", "script", "bash", "sh",
    "shell", "shell_command", "code", "input", "run", "cmd_string",
)


def shell_arg_key(openai_tools, shell_tool):
    """Required (or first) param name of the registered shell tool, e.g. Codex's
    exec_command -> "cmd". Used to normalize model arg keys."""
    if not shell_tool:
        return None
    for t in openai_tools or []:
        fn = t.get("function", t)
        if (fn.get("name") or t.get("name")) != shell_tool:
            continue
        params = fn.get("parameters") or {}
        props = params.get("properties") or {}
        for r in (params.get("required") or []):
            if props.get(r, {}).get("type") in (None, "string"):
                return r
        if props:
            return next(iter(props))
    return "cmd"


def _is_shell_name(name, shell_tool):
    return name == shell_tool or name in _SHELL_ALIASES


def _normalize_shell_args(args_json, req):
    """If the shell tool's required param `req` is absent but a known alias is
    present, rename it. Keeps exec_command calls valid when the model uses
    `command`/`script`/... instead of `cmd`."""
    if not req:
        return args_json
    try:
        a = json.loads(args_json) if args_json else {}
    except Exception:
        return args_json
    if not isinstance(a, dict) or req in a:
        return args_json
    for alias in _SHELL_ARG_ALIASES:
        if alias != req and isinstance(a.get(alias), str):
            a[req] = a.pop(alias)
            return json.dumps(a)
    return args_json


def translate_client_tool(
    name: str, args_json: str, valid: set, shell_tool: Optional[str],
    cwd: Optional[str] = None, shell_param: Optional[str] = None,
):
    """Return (name, args_json) with Claude read-only tools rewritten to the
    registered shell tool. exec_command's own calls and any already-valid tool
    pass through untouched; unknown non-shell tools fall back to name-remap.
    ``cwd`` (from the request's <cwd>) makes hallucinated absolute paths resolve."""
    if name in valid:
        if shell_param and _is_shell_name(name, shell_tool):
            args_json = _normalize_shell_args(args_json, shell_param)
        return name, args_json
    exec_tool = "exec_command" if "exec_command" in valid else shell_tool
    if exec_tool:
        try:
            a = json.loads(args_json) if args_json else {}
        except Exception:
            a = {}
        if isinstance(a, dict):
            cmd = _claude_tool_to_shell(name, a, cwd)
            if cmd is not None:
                return exec_tool, json.dumps({(shell_param or "cmd"): cmd})
    _remapped = remap_tool_name(name, valid, shell_tool)
    if shell_param and _is_shell_name(_remapped, shell_tool):
        args_json = _normalize_shell_args(args_json, shell_param)
    return _remapped, args_json


_CWD_RE = None


def extract_cwd(body: Dict[str, Any]) -> Optional[str]:
    """Pull the working directory from Codex's <cwd>...</cwd> environment_context
    (sent in instructions/input), so path fix-ups target the real directory."""
    import re
    global _CWD_RE
    if _CWD_RE is None:
        _CWD_RE = re.compile(r"<cwd>\s*([^<\s]+)\s*</cwd>")
    blob = _text_of(body.get("instructions"))
    inp = body.get("input")
    if isinstance(inp, str):
        blob += "\n" + inp
    elif isinstance(inp, list):
        for it in inp:
            if isinstance(it, dict):
                blob += "\n" + _text_of(it.get("content", ""))
    m = _CWD_RE.search(blob or "")
    return m.group(1) if m else None


# --------------------------------------------------------------- SSE emitter
class ResponsesStreamEmitter:
    """Builds the ordered Responses SSE event stream from incremental text /
    tool-call events. Each method returns a list of SSE strings to yield.

    Output-item lifecycle (Codex expects these exact event types):
      message:       output_item.added -> content_part.added -> output_text.delta*
                     -> output_text.done -> content_part.done -> output_item.done
      function_call: output_item.added -> function_call_arguments.delta*
                     -> function_call_arguments.done -> output_item.done
      end:           response.completed
    """

    def __init__(self, resp_id: str, model: str):
        self.resp_id = resp_id
        self.model = model
        self._seq = itertools.count(0)
        self.out_index = 0
        self.final_output: List[Dict[str, Any]] = []
        self._open: Optional[Dict[str, Any]] = None  # current open output item

    def _ev(self, ev: str, extra: Dict[str, Any]) -> str:
        d = {"type": ev, "sequence_number": next(self._seq)}
        d.update(extra)
        return f"event: {ev}\ndata: {json.dumps(d)}\n\n"

    def _base(self, status: str, output: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "id": self.resp_id,
            "object": "response",
            "status": status,
            "model": self.model,
            "output": output,
        }

    def created(self) -> List[str]:
        return [
            self._ev("response.created", {"response": self._base("in_progress", [])}),
            self._ev(
                "response.in_progress", {"response": self._base("in_progress", [])}
            ),
        ]

    def _close_open(self) -> List[str]:
        if self._open is None:
            return []
        o = self._open
        self._open = None
        if o["kind"] == "message":
            mid, cur, txt = o["id"], o["index"], o["text"]
            item = {
                "id": mid, "type": "message", "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": txt}],
            }
            self.final_output.append(item)
            return [
                self._ev("response.output_text.done", {
                    "item_id": mid, "output_index": cur,
                    "content_index": 0, "text": txt}),
                self._ev("response.content_part.done", {
                    "item_id": mid, "output_index": cur, "content_index": 0,
                    "part": {"type": "output_text", "text": txt}}),
                self._ev("response.output_item.done", {
                    "output_index": cur, "item": item}),
            ]
        # function_call
        fid, cur = o["id"], o["index"]
        item = {
            "id": fid, "type": "function_call", "status": "completed",
            "call_id": o["call_id"], "name": o["name"], "arguments": o["args"],
        }
        self.final_output.append(item)
        return [
            self._ev("response.function_call_arguments.done", {
                "item_id": fid, "output_index": cur, "arguments": o["args"]}),
            self._ev("response.output_item.done", {
                "output_index": cur, "item": item}),
        ]

    def text_delta(self, delta: str) -> List[str]:
        out: List[str] = []
        if self._open and self._open["kind"] != "message":
            out += self._close_open()
        if not self._open:
            mid, cur = _rid("msg"), self.out_index
            self.out_index += 1
            self._open = {"kind": "message", "id": mid, "index": cur, "text": ""}
            out.append(self._ev("response.output_item.added", {
                "output_index": cur,
                "item": {"id": mid, "type": "message", "role": "assistant",
                         "status": "in_progress", "content": []}}))
            out.append(self._ev("response.content_part.added", {
                "item_id": mid, "output_index": cur, "content_index": 0,
                "part": {"type": "output_text", "text": ""}}))
        self._open["text"] += delta
        out.append(self._ev("response.output_text.delta", {
            "item_id": self._open["id"], "output_index": self._open["index"],
            "content_index": 0, "delta": delta}))
        return out

    def tool_start(self, call_id: str, name: str) -> List[str]:
        out = self._close_open()
        fid, cur = _rid("fc"), self.out_index
        self.out_index += 1
        self._open = {
            "kind": "fc", "id": fid, "index": cur,
            "call_id": call_id or _rid("call"), "name": name, "args": "",
        }
        out.append(self._ev("response.output_item.added", {
            "output_index": cur,
            "item": {"id": fid, "type": "function_call", "status": "in_progress",
                     "call_id": self._open["call_id"], "name": name,
                     "arguments": ""}}))
        return out

    def tool_args(self, delta: str) -> List[str]:
        if not self._open or self._open["kind"] != "fc" or not delta:
            return []
        self._open["args"] += delta
        return [self._ev("response.function_call_arguments.delta", {
            "item_id": self._open["id"], "output_index": self._open["index"],
            "delta": delta})]

    def tool_end(self) -> List[str]:
        return self._close_open()

    def finish(self, input_tokens: int, output_tokens: int) -> List[str]:
        out = self._close_open()
        completed = self._base("completed", self.final_output)
        completed["usage"] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
        out.append(self._ev("response.completed", {"response": completed}))
        return out


# ------------------------------------------------------- non-stream response
def build_responses_object(
    resp_id: str,
    model: str,
    content_text: str,
    tool_calls: List[Any],
    input_tokens: int,
    output_tokens: int,
    valid: set,
    shell_tool: Optional[str],
    cwd: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a full (non-streaming) Responses object from parsed output.

    ``tool_calls`` are ATOM ``ToolCall`` objects (``.id``, ``.function`` dict)."""
    output: List[Dict[str, Any]] = []
    if content_text:
        output.append({
            "id": _rid("msg"), "type": "message", "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": content_text}],
        })
    for tc in tool_calls or []:
        fn = getattr(tc, "function", None) or {}
        name, args = translate_client_tool(
            fn.get("name", ""), fn.get("arguments", "") or "", valid, shell_tool, cwd
        )
        output.append({
            "id": _rid("fc"), "type": "function_call", "status": "completed",
            "call_id": getattr(tc, "id", None) or _rid("call"),
            "name": name, "arguments": args,
        })
    return {
        "id": resp_id, "object": "response", "status": "completed",
        "model": model, "output": output,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }
