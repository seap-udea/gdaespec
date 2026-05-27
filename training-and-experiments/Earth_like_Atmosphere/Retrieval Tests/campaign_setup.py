#!/usr/bin/env python3
"""Create the directory tree and CSV headers for the 5-observation campaign.

This lightweight setup script does not generate spectra or run retrievals. It
only prepares the folder layout expected by the campaign scripts and prints the
configured cases for a quick sanity check.
"""

from __future__ import annotations

from campaign_common import (
    BRANCHES,
    CAMPAIGN_DIR,
    N_TRANSITS,
    STRATEGIES,
    TEST_IDS,
    ensure_campaign_layout,
    iter_cases,
)


def describe_campaign() -> str:
    """Return a short human-readable summary of the configured campaign."""
    lines = [
        f"Campaign directory: {CAMPAIGN_DIR}",
        f"Tests: {', '.join(TEST_IDS)}",
        f"Branches: {', '.join(BRANCHES)}",
        f"Strategies: {', '.join(STRATEGIES)}",
        f"Transits per observation: {N_TRANSITS}",
        "Cases:",
    ]
    for case in iter_cases():
        lines.append(f"  - {case.branch}: f_spot={case.f_spot:.2f}, f_fac={case.f_fac:.2f}")
    return "\n".join(lines)


def main() -> None:
    """Create the campaign layout and print the configuration summary."""
    ensure_campaign_layout()
    print(describe_campaign())


if __name__ == "__main__":
    main()
