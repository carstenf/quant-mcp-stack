"""Introspect the installed ML4T libraries (run as a subprocess by the MCP server).

    python api_catalog.py build <out.json>      -> catalog of public classes/functions with one-line docs
    python api_catalog.py describe <dotted.name> -> JSON with signature, full docstring, public methods
"""
from __future__ import annotations

import importlib
import importlib.metadata as imd
import inspect
import json
import pkgutil
import sys
import warnings

warnings.filterwarnings("ignore")
PACKAGES = ("ml4t.backtest", "ml4t.diagnostic", "ml4t.data", "ml4t.engineer", "ml4t.models", "ml4t.specs")


def _doc1(obj) -> str:
    return (inspect.getdoc(obj) or "").strip().split("\n")[0][:160]


def build(out: str) -> None:
    items: dict[str, dict] = {}
    exported: set[str] = set()
    errors = []
    for top in PACKAGES:
        try:
            pkg = importlib.import_module(top)
        except Exception as e:
            errors.append([top, repr(e)[:200]])
            continue
        for k in getattr(pkg, "__all__", []):
            v = getattr(pkg, k, None)
            if inspect.isclass(v) or inspect.isfunction(v):
                exported.add(f"{v.__module__}.{v.__qualname__}")
        for m in pkgutil.walk_packages(pkg.__path__, top + "."):
            parts = m.name.split(".")
            if any(p.startswith("_") for p in parts[2:]) or "tests" in parts:
                continue
            try:
                mod = importlib.import_module(m.name)
            except Exception as e:
                errors.append([m.name, type(e).__name__])
                continue
            for k in getattr(mod, "__all__", []):
                v = getattr(mod, k, None)
                if inspect.isclass(v) or inspect.isfunction(v):
                    exported.add(f"{v.__module__}.{v.__qualname__}")
            for k, v in vars(mod).items():
                if k.startswith("_") or not (inspect.isclass(v) or inspect.isfunction(v)):
                    continue
                if not str(getattr(v, "__module__", "")).startswith("ml4t."):
                    continue
                q = f"{v.__module__}.{v.__qualname__}"
                if q not in items:
                    items[q] = {"kind": "class" if inspect.isclass(v) else "function", "doc": _doc1(v),
                                "import": f"from {m.name} import {k}"}
                # prefer the shortest public import path
                elif len(m.name) < len(items[q]["import"].split()[1]):
                    items[q]["import"] = f"from {m.name} import {k}"
    for q, it in items.items():
        it["top_level"] = q in exported
        top = ".".join(q.split(".")[:2])
        for k in getattr(importlib.import_module(top), "__all__", []):
            if q.endswith("." + k) and f"{top}." in q:
                obj = getattr(importlib.import_module(top), k, None)
                if obj is not None and f"{obj.__module__}.{obj.__qualname__}" == q:
                    it["import"] = f"from {top} import {k}"
    versions = {}
    for p in ("ml4t-backtest", "ml4t-diagnostic", "ml4t-data", "ml4t-engineer", "ml4t-models", "ml4t-specs"):
        try:
            versions[p] = imd.version(p)
        except imd.PackageNotFoundError:
            pass
    json.dump({"versions": versions, "items": items, "import_errors": errors}, open(out, "w"))
    print(json.dumps({"items": len(items), "versions": versions}))


def describe(name: str) -> None:
    parts = name.split(".")
    obj, err = None, None
    # prefer an attribute of the parent (function re-exported under a module of the same name) over a submodule
    for i in list(range(len(parts) - 1, 0, -1)) + [len(parts)]:
        try:
            obj = importlib.import_module(".".join(parts[:i]))
            for a in parts[i:]:
                obj = getattr(obj, a)
            break
        except Exception as e:
            err, obj = e, None
    if obj is None:
        print(json.dumps({"ok": False, "error": f"cannot resolve {name}: {err}"}))
        return
    out = {"ok": True, "name": name, "kind": type(obj).__name__, "module": getattr(obj, "__module__", None)}
    try:
        out["signature"] = f"{parts[-1]}{inspect.signature(obj)}"
    except (TypeError, ValueError):
        pass
    out["doc"] = (inspect.getdoc(obj) or "")[:8000]
    if inspect.isclass(obj):
        if hasattr(obj, "__dataclass_fields__"):
            out["fields"] = {k: str(f.type) for k, f in obj.__dataclass_fields__.items()}
        meths = {}
        for k, v in inspect.getmembers(obj):
            if k.startswith("_") or not callable(v):
                continue
            try:
                sig = str(inspect.signature(v))
            except (TypeError, ValueError):
                sig = "(...)"
            meths[k] = f"{sig}  {_doc1(v)}".strip()
        out["methods"] = meths
        try:
            out["source_file"] = inspect.getsourcefile(obj)
        except TypeError:
            pass
    elif inspect.ismodule(obj):
        out["members"] = sorted(getattr(obj, "__all__", [k for k in vars(obj) if not k.startswith("_")]))[:300]
    print(json.dumps(out, default=str))


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(sys.argv[2])
    else:
        describe(sys.argv[2])
