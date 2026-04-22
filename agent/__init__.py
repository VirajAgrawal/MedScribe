"""
agent/__init__.py
=================
Exposes the Flask Blueprint so app.py can register it with two lines:

    from agent import agent_bp
    app.register_blueprint(agent_bp)

That is the ONLY change needed in app.py.
"""

from .routes import agent_bp

__all__ = ["agent_bp"]