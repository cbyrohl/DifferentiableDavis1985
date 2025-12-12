# Development Notes

## Python Environment

Use `uv run` to execute Python scripts and commands within the project environment:

```bash
uv run python script.py
uv run davis1985 <command>
```

## CLI Tool

The project provides a `davis1985` CLI tool built with Typer.

Available commands:
- `uv run davis1985 info` - Show project information
- `uv run davis1985 plot` - Run simulation and generate density field plots
