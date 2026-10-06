"""Simulated desktop: every experiment can run without touching the real OS.

Actions only modify a ``DesktopState``. The action whitelist is the set of
registry intents with a handler here; anything else is refused.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field

from ..intents.registry import IntentRegistry


@dataclass
class DesktopState:
    open_apps: list = field(default_factory=lambda: ["explorer"])
    active_app: str | None = "explorer"
    volume: int = 50
    muted: bool = False
    media_state: str = "stopped"          # stopped | playing | paused
    scroll: int = 0
    documents: dict = field(default_factory=dict)   # app -> {"text": str, "saved": bool}
    files: list = field(default_factory=lambda: ["informe.docx", "notas.txt", "foto.png", "datos.csv",
                                                 "report.docx", "notes.txt"])
    trash: list = field(default_factory=list)
    sent: list = field(default_factory=list)
    searches: list = field(default_factory=list)
    screenshots: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ActionResult:
    success: bool
    intent: str
    slots: dict
    message: str
    before: dict
    after: dict
    key: str = ""
    params: dict = field(default_factory=dict)


MESSAGES = {
    "invalid": "{reason}", "opened": "opened {app}", "not_open": "{app} is not open", "closed": "closed {app}",
    "closed_lost": "closed {app} (unsaved changes lost)", "switched": "switched to {app}", "volume": "volume {value}",
    "muted": "muted", "playing": "playing", "nothing_playing": "nothing is playing", "paused": "paused",
    "searched": "searched '{query}'", "no_text_target": "the active window does not accept text",
    "typed": "typed into {app}", "scroll": "scroll {value}", "no_document": "no document is active",
    "saved": "saved {app}", "sent": "sent to {contact}", "file_not_found": "{file} not found",
    "trashed": "moved {file} to trash", "screenshot": "screenshot taken",
}


class Sandbox:
    def __init__(self, registry: IntentRegistry | None = None, state: DesktopState | None = None):
        self.registry = registry or IntentRegistry.default()
        self.state = state or DesktopState()
        self.log: list[ActionResult] = []

    @property
    def whitelist(self) -> set[str]:
        return {n for n in self.registry.names if hasattr(self, f"_do_{n}")}

    def context(self) -> dict:
        s = self.state
        return {"active_app": s.active_app, "category": self.registry.app_category(s.active_app or ""),
                "open_apps": list(s.open_apps), "media_state": s.media_state}

    def validate(self, intent: str, slots: dict) -> tuple[bool, str]:
        if intent not in self.whitelist:
            return False, f"action '{intent}' is not whitelisted"
        spec = self.registry.spec(intent)
        missing = [s for s in spec.slots if s not in slots and not (s == "app" and intent == "close_app")]
        if missing:
            return False, "missing slot(s): " + ", ".join(missing)
        if "value" in slots and not 0 <= int(slots["value"]) <= 100:
            return False, "value must be within [0, 100]"
        if "app" in slots and slots["app"] not in self.registry.apps:
            return False, f"unknown application '{slots['app']}'"
        return True, "ok"

    def execute(self, intent: str, slots: dict | None = None) -> ActionResult:
        slots = dict(slots or {})
        before = copy.deepcopy(self.state.as_dict())
        ok, reason = self.validate(intent, slots)
        key, params = "invalid", {"reason": reason}
        if ok:
            ok, key, params = getattr(self, f"_do_{intent}")(slots)
            reason = MESSAGES[key].format(**params)
        res = ActionResult(ok, intent, slots, reason, before, copy.deepcopy(self.state.as_dict()), key, params)
        self.log.append(res)
        return res

    # ---------------------------------------------------------------- actions
    def _do_open_app(self, s):
        app = s["app"]
        if app not in self.state.open_apps:
            self.state.open_apps.append(app)
        if self.registry.app_category(app) == "document":
            self.state.documents.setdefault(app, {"text": "", "saved": True})
        self.state.active_app = app
        return True, "opened", {"app": app}

    def _do_close_app(self, s):
        app = s.get("app") or self.state.active_app
        if app not in self.state.open_apps:
            return False, "not_open", {"app": app}
        self.state.open_apps.remove(app)
        lost = not self.state.documents.pop(app, {"saved": True})["saved"]
        self.state.active_app = self.state.open_apps[-1] if self.state.open_apps else None
        return True, "closed_lost" if lost else "closed", {"app": app}

    def _do_switch_window(self, s):
        app = s["app"]
        if app not in self.state.open_apps:
            return False, "not_open", {"app": app}
        self.state.active_app = app
        return True, "switched", {"app": app}

    def _do_volume_up(self, s):
        self.state.volume = min(100, self.state.volume + 10)
        return True, "volume", {"value": self.state.volume}

    def _do_volume_down(self, s):
        self.state.volume = max(0, self.state.volume - 10)
        return True, "volume", {"value": self.state.volume}

    def _do_set_volume(self, s):
        self.state.volume = int(s["value"])
        return True, "volume", {"value": self.state.volume}

    def _do_mute(self, s):
        self.state.muted = True
        return True, "muted", {}

    def _do_play_media(self, s):
        self.state.media_state = "playing"
        return True, "playing", {}

    def _do_pause_media(self, s):
        if self.state.media_state != "playing":
            return False, "nothing_playing", {}
        self.state.media_state = "paused"
        return True, "paused", {}

    def _do_search_web(self, s):
        self.state.searches.append(s["query"])
        return True, "searched", {"query": s["query"]}

    def _do_type_text(self, s):
        app = self.state.active_app
        if app not in self.state.documents:
            return False, "no_text_target", {}
        doc = self.state.documents[app]
        doc["text"] = (doc["text"] + " " + s["text"]).strip()
        doc["saved"] = False
        return True, "typed", {"app": app}

    def _do_scroll_down(self, s):
        self.state.scroll += 1
        return True, "scroll", {"value": self.state.scroll}

    def _do_scroll_up(self, s):
        self.state.scroll = max(0, self.state.scroll - 1)
        return True, "scroll", {"value": self.state.scroll}

    def _do_save_document(self, s):
        app = self.state.active_app
        if app not in self.state.documents:
            return False, "no_document", {}
        self.state.documents[app]["saved"] = True
        return True, "saved", {"app": app}

    def _do_send_message(self, s):
        self.state.sent.append(s["contact"])
        return True, "sent", {"contact": s["contact"]}

    def _do_delete_file(self, s):
        f = s["file"]
        if f not in self.state.files:
            return False, "file_not_found", {"file": f}
        self.state.files.remove(f)
        self.state.trash.append(f)
        return True, "trashed", {"file": f}

    def _do_take_screenshot(self, s):
        self.state.screenshots += 1
        return True, "screenshot", {}
