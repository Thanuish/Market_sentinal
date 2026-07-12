"""MCP server exposing the deterministic sizing engine.

Any MCP client (Claude Desktop, Claude Code, another agent) can request an
exact position size — but the math runs here, in tested Python, outside the
model's reach. That is the deterministic-execution-guardrail pattern as a
Model Context Protocol interface.

Run:  python src/mcp_server/sizing_server.py     (stdio transport)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from sentinel.tools.sizing import SizingDecision, size_position  # noqa: E402

mcp = FastMCP("market-sentinel-sizing")


@mcp.tool()
def compute_position_size(ticker: str, price: float, signal_strength: float,
                          equity: float, cash: float) -> dict:
    """Compute an exact, risk-capped position size (fractional Kelly).

    The caller supplies the validated signal and portfolio state; the share
    quantity is computed deterministically and cannot be altered by the model.
    """
    d: SizingDecision = size_position(ticker, price, signal_strength, equity, cash)
    return {"ticker": d.ticker, "qty": d.qty, "notional": d.notional,
            "capped_by": d.capped_by}


if __name__ == "__main__":
    mcp.run()
