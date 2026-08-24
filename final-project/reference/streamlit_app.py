"""Launch the certified Financial Analyst Copilot reference application."""

from finai_academy.capstone import build_reference_copilot, render_capstone

render_capstone(lambda _request: build_reference_copilot())
