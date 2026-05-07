import json
from pathlib import Path
from core.logger import get_logger

_log = get_logger("user_manager")

_USERS_FILE = Path(__file__).resolve().parent.parent / "users.json"


class UserManager:
    """Gestiona el usuario activo y persiste su progreso en users.json."""

    def __init__(self):
        self.current_user = "Invitado"
        self._data = self._load()

    def _load(self):
        if _USERS_FILE.exists():
            try:
                with open(_USERS_FILE, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save(self):
        try:
            with open(_USERS_FILE, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            _log.error("Error al guardar progreso: %s", e)

    def set_user(self, name):
        self.current_user = name.strip() if name.strip() else "Invitado"
        if self.current_user not in self._data:
            self._data[self.current_user] = {"best_score": 0, "total_games": 0}
            self._save()

    def get_user_name(self):
        return self.current_user

    def update_stats(self, score):
        if self.current_user not in self._data:
            self._data[self.current_user] = {"best_score": 0, "total_games": 0}
        entry = self._data[self.current_user]
        entry["total_games"] += 1
        if score > entry["best_score"]:
            entry["best_score"] = score
        self._save()

    def get_best_score(self):
        return self._data.get(self.current_user, {}).get("best_score", 0)


# Instancia única para ser compartida entre escenas
user_manager = UserManager()
