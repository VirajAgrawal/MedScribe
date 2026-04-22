"""
agent/routes.py
===============
Flask Blueprint for the Healthcare AI Agent.

Registers all agent routes under the /agent prefix.
Registered in app.py with just these two lines:

    from agent import agent_bp
    app.register_blueprint(agent_bp)

Routes:
    GET  /agent/         -> Serves agent.html UI
    POST /agent/run      -> Full synchronous agent pipeline
    GET  /agent/stream   -> Streaming SSE (live progress updates)
"""

from flask import Blueprint, request, jsonify, send_from_directory, Response, stream_with_context
from .executor import run_agent, stream_agent

agent_bp = Blueprint("agent", __name__, url_prefix="/agent")


@agent_bp.route("/")
def agent_ui():
    return send_from_directory("static", "agent.html")


@agent_bp.route("/run", methods=["POST"])
def run():
    data = request.get_json()
    goal = (data or {}).get("goal", "").strip()
    if not goal:
        return jsonify({"success": False, "error": "Goal cannot be empty."}), 400
    try:
        return jsonify(run_agent(goal))
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@agent_bp.route("/stream")
def stream():
    goal = request.args.get("goal", "").strip()
    if not goal:
        return jsonify({"error": "Goal query param required"}), 400

    def generate():
        try:
            for evt in stream_agent(goal):
                yield evt
        except Exception as e:
            import json
            yield f"data: {json.dumps({'phase': 'error', 'message': str(e)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )