"""
Binder endpoints: set lists and per-user song preferences.

GET    /api/setlists                  — all of this user's set lists
POST   /api/setlists                  — create {name, items?}
PUT    /api/setlists/<id>             — rename and/or replace items {name?, items?}
DELETE /api/setlists/<id>             — delete
POST   /api/setlists/<id>/items       — add one {kind, id}, appended to the running order
DELETE /api/setlists/<id>/items       — remove one {kind, id}

GET    /api/song-prefs                — {"job:abc": {"song_key": "Am"}, ...}
PUT    /api/song-prefs                — set one {kind, id, song_key}

WHY THIS EXISTS: both of these used to live in localStorage, which meant a set list
built on a laptop was simply missing from the phone at the gig, and a key the band
actually plays in had nowhere to be recorded at all. Everything here is scoped to
the authenticated user; there is no sharing and no cross-user read.

The user's chosen key is deliberately kept SEPARATE from the detected key. The
detector's answer stays on the job/chart untouched, so we never lose it and can
always show both.
"""

import logging
from flask import Blueprint, request, jsonify, g

from auth.middleware import auth_required
from db import query_one, query_all, execute, execute_returning

logger = logging.getLogger(__name__)
binder_bp = Blueprint("binder", __name__)

MAX_SETLISTS = 200          # generous; exists only to stop runaway creation
MAX_ITEMS_PER_LIST = 500
NAME_MAX = 80
VALID_KINDS = ("job", "chart")


def _uid():
    return str(g.current_user.id)


def _clean_name(raw):
    name = (raw or "").strip()
    return name[:NAME_MAX] if name else None


def _clean_ref(data):
    """Return (kind, id) or None. Refs identify a song: a processed job or a library chart."""
    kind = (data.get("kind") or "").strip().lower()
    ref_id = str(data.get("id") or "").strip()
    if kind not in VALID_KINDS or not ref_id or len(ref_id) > 128:
        return None
    return kind, ref_id


def _clean_item(raw):
    """A set-list entry: the ref plus a cached display title, so the panel can
    render before the full chart list has loaded."""
    ref = _clean_ref(raw or {})
    if not ref:
        return None
    item = {"kind": ref[0], "id": ref[1]}
    title = ((raw or {}).get("title") or "").strip()
    if title:
        item["title"] = title[:200]
    return item


def _row_to_setlist(r):
    return {
        "id": r["id"],
        "name": r["name"],
        "items": r["items"] or [],
        "created_at": r["created_at"].isoformat() if r.get("created_at") else None,
        "updated_at": r["updated_at"].isoformat() if r.get("updated_at") else None,
    }


# ─────────────────────────── set lists ───────────────────────────

@binder_bp.route("/api/setlists", methods=["GET"])
@auth_required
def list_setlists():
    rows = query_all(
        "SELECT id, name, items, created_at, updated_at FROM setlists "
        "WHERE user_id = %s ORDER BY lower(name)",
        (_uid(),),
    )
    return jsonify({"setlists": [_row_to_setlist(r) for r in rows]})


@binder_bp.route("/api/setlists", methods=["POST"])
@auth_required
def create_setlist():
    data = request.get_json(silent=True) or {}
    name = _clean_name(data.get("name"))
    if not name:
        return jsonify({"error": "Name is required"}), 400

    n = query_one("SELECT COUNT(*) AS n FROM setlists WHERE user_id = %s", (_uid(),))
    if int((n or {}).get("n") or 0) >= MAX_SETLISTS:
        return jsonify({"error": f"You have reached the limit of {MAX_SETLISTS} set lists."}), 429

    items = []
    for raw in (data.get("items") or [])[:MAX_ITEMS_PER_LIST]:
        it = _clean_item(raw)
        if it:
            items.append(it)

    try:
        row = execute_returning(
            "INSERT INTO setlists (user_id, name, items) VALUES (%s, %s, %s::jsonb) "
            "RETURNING id, name, items, created_at, updated_at",
            (_uid(), name, __import__("json").dumps(items)),
        )
    except Exception as e:
        # the case-insensitive unique index is the only expected failure here
        if "idx_setlists_user_name" in str(e):
            return jsonify({"error": "You already have a set list with that name."}), 409
        logger.exception("setlist create failed")
        return jsonify({"error": "Could not create the set list."}), 500

    return jsonify(_row_to_setlist(row)), 201


@binder_bp.route("/api/setlists/<int:setlist_id>", methods=["PUT"])
@auth_required
def update_setlist(setlist_id):
    data = request.get_json(silent=True) or {}
    owned = query_one("SELECT id FROM setlists WHERE id = %s AND user_id = %s", (setlist_id, _uid()))
    if not owned:
        return jsonify({"error": "Not found"}), 404

    sets, params = [], []
    if "name" in data:
        name = _clean_name(data.get("name"))
        if not name:
            return jsonify({"error": "Name cannot be empty"}), 400
        sets.append("name = %s"); params.append(name)
    if "items" in data:
        items = []
        for raw in (data.get("items") or [])[:MAX_ITEMS_PER_LIST]:
            it = _clean_item(raw)
            if it:
                items.append(it)
        sets.append("items = %s::jsonb"); params.append(__import__("json").dumps(items))
    if not sets:
        return jsonify({"error": "Nothing to update"}), 400

    sets.append("updated_at = NOW()")
    params += [setlist_id, _uid()]
    try:
        row = execute_returning(
            f"UPDATE setlists SET {', '.join(sets)} WHERE id = %s AND user_id = %s "
            "RETURNING id, name, items, created_at, updated_at",
            tuple(params),
        )
    except Exception as e:
        if "idx_setlists_user_name" in str(e):
            return jsonify({"error": "You already have a set list with that name."}), 409
        logger.exception("setlist update failed")
        return jsonify({"error": "Could not update the set list."}), 500
    return jsonify(_row_to_setlist(row))


@binder_bp.route("/api/setlists/<int:setlist_id>", methods=["DELETE"])
@auth_required
def delete_setlist(setlist_id):
    execute("DELETE FROM setlists WHERE id = %s AND user_id = %s", (setlist_id, _uid()))
    return jsonify({"ok": True})


@binder_bp.route("/api/setlists/<int:setlist_id>/items", methods=["POST"])
@auth_required
def add_item(setlist_id):
    ref = _clean_ref(request.get_json(silent=True) or {})
    if not ref:
        return jsonify({"error": "kind must be job or chart, and id is required"}), 400
    row = query_one("SELECT items FROM setlists WHERE id = %s AND user_id = %s", (setlist_id, _uid()))
    if not row:
        return jsonify({"error": "Not found"}), 404

    items = row["items"] or []
    entry = _clean_item(request.get_json(silent=True) or {}) or {"kind": ref[0], "id": ref[1]}
    if any(i.get("kind") == entry["kind"] and str(i.get("id")) == str(entry["id"]) for i in items):
        return jsonify({"ok": True, "items": items})          # already there, no duplicate
    if len(items) >= MAX_ITEMS_PER_LIST:
        return jsonify({"error": "That set list is full."}), 429
    items.append(entry)                                        # append: order is the running order

    updated = execute_returning(
        "UPDATE setlists SET items = %s::jsonb, updated_at = NOW() "
        "WHERE id = %s AND user_id = %s RETURNING items",
        (__import__("json").dumps(items), setlist_id, _uid()),
    )
    return jsonify({"ok": True, "items": updated["items"]})


@binder_bp.route("/api/setlists/<int:setlist_id>/items", methods=["DELETE"])
@auth_required
def remove_item(setlist_id):
    ref = _clean_ref(request.get_json(silent=True) or {})
    if not ref:
        return jsonify({"error": "kind must be job or chart, and id is required"}), 400
    row = query_one("SELECT items FROM setlists WHERE id = %s AND user_id = %s", (setlist_id, _uid()))
    if not row:
        return jsonify({"error": "Not found"}), 404

    items = [i for i in (row["items"] or [])
             if not (i.get("kind") == ref[0] and str(i.get("id")) == str(ref[1]))]
    updated = execute_returning(
        "UPDATE setlists SET items = %s::jsonb, updated_at = NOW() "
        "WHERE id = %s AND user_id = %s RETURNING items",
        (__import__("json").dumps(items), setlist_id, _uid()),
    )
    return jsonify({"ok": True, "items": updated["items"]})


# ──────────────────────── song preferences ───────────────────────

@binder_bp.route("/api/song-prefs", methods=["GET"])
@auth_required
def get_song_prefs():
    rows = query_all(
        "SELECT ref_kind, ref_id, song_key FROM song_prefs WHERE user_id = %s", (_uid(),)
    )
    return jsonify({"prefs": {f"{r['ref_kind']}:{r['ref_id']}": {"song_key": r["song_key"]} for r in rows}})


@binder_bp.route("/api/song-prefs", methods=["PUT"])
@auth_required
def set_song_pref():
    data = request.get_json(silent=True) or {}
    ref = _clean_ref(data)
    if not ref:
        return jsonify({"error": "kind must be job or chart, and id is required"}), 400

    raw_key = data.get("song_key")
    song_key = (raw_key or "").strip()[:12] or None      # empty clears the override

    execute(
        "INSERT INTO song_prefs (user_id, ref_kind, ref_id, song_key) VALUES (%s, %s, %s, %s) "
        "ON CONFLICT (user_id, ref_kind, ref_id) "
        "DO UPDATE SET song_key = EXCLUDED.song_key, updated_at = NOW()",
        (_uid(), ref[0], ref[1], song_key),
    )
    return jsonify({"ok": True, "kind": ref[0], "id": ref[1], "song_key": song_key})
