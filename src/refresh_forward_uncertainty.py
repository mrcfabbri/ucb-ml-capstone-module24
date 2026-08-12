"""Refresh aggregate clustered-bootstrap evidence from saved forward predictions."""

from __future__ import annotations

from module24_workflow import refresh_forward_uncertainty

if __name__ == "__main__":
    comparisons = refresh_forward_uncertainty()
    print(f"Refreshed {len(comparisons)} customer-clustered forward comparisons.")
