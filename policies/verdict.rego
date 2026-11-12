# Three-valued verdicts and per-domain deployment decisions.
#
# Input:
#   {
#     "domains": ["e0", "e1", ...],
#     "checks": [
#       {"id": "R1/global", "domain": "*" | "e3",
#        "kind": "upper" | "lower" | "sufficient_upper" | "alarm_upper",
#        "threshold": 0.1, "estimate": 0.04, "lower": 0.02, "upper": 0.07}
#     ]
#   }
#
# kind "upper": the statistic must stay at or below threshold.
# kind "lower": the statistic must stay at or above threshold.
# kind "sufficient_upper": a local sufficient condition for a global upper bound. Passing it implies the
#   global property; failing it implies nothing about the global property, so it yields "inconclusive".
# kind "alarm_upper": a detector, not a certificate. It fails only when the evidence shows the statistic above
#   threshold (lower bound above it) and passes otherwise; missing evidence never escalates.
# A point estimate is passed with lower == upper == estimate, which makes the verdict two-valued.
#
# Output:
#   verdict[id] in {"pass", "fail", "inconclusive"}
#   decision[domain] in {"deploy", "hold", "escalate"}
package raimon.verdict

import rego.v1

verdict[c.id] := "pass" if {
	some c in input.checks
	c.kind == "upper"
	c.upper <= c.threshold
}

verdict[c.id] := "fail" if {
	some c in input.checks
	c.kind == "upper"
	c.lower > c.threshold
}

verdict[c.id] := "inconclusive" if {
	some c in input.checks
	c.kind == "upper"
	c.lower <= c.threshold
	c.upper > c.threshold
}

verdict[c.id] := "pass" if {
	some c in input.checks
	c.kind == "lower"
	c.lower >= c.threshold
}

verdict[c.id] := "fail" if {
	some c in input.checks
	c.kind == "lower"
	c.upper < c.threshold
}

verdict[c.id] := "inconclusive" if {
	some c in input.checks
	c.kind == "lower"
	c.lower < c.threshold
	c.upper >= c.threshold
}

verdict[c.id] := "pass" if {
	some c in input.checks
	c.kind == "sufficient_upper"
	c.upper <= c.threshold
}

verdict[c.id] := "inconclusive" if {
	some c in input.checks
	c.kind == "sufficient_upper"
	c.upper > c.threshold
}

verdict[c.id] := "fail" if {
	some c in input.checks
	c.kind == "alarm_upper"
	c.lower > c.threshold
}

verdict[c.id] := "pass" if {
	some c in input.checks
	c.kind == "alarm_upper"
	c.lower <= c.threshold
}

# Checks that govern a domain: its own checks and every global check.
applies(c, _) if c.domain == "*"

applies(c, d) if c.domain == d

# The verdict of a check, computed from the check itself. Looking it up as verdict[c.id] would make OPA scan
# every check for the id, so a round's decisions would cost domains x checks^2 rule evaluations.
fails(c) if {
	c.kind in {"upper", "alarm_upper"}
	c.lower > c.threshold
}

fails(c) if {
	c.kind == "lower"
	c.upper < c.threshold
}

undecided(c) if {
	c.kind == "upper"
	c.lower <= c.threshold
	c.upper > c.threshold
}

undecided(c) if {
	c.kind == "lower"
	c.lower < c.threshold
	c.upper >= c.threshold
}

undecided(c) if {
	c.kind == "sufficient_upper"
	c.upper > c.threshold
}

failing(d) if {
	some c in input.checks
	applies(c, d)
	fails(c)
}

open(d) if {
	some c in input.checks
	applies(c, d)
	undecided(c)
}

# A failing check holds the model; otherwise an inconclusive check escalates.
decision[d] := "hold" if {
	some d in input.domains
	failing(d)
}

decision[d] := "escalate" if {
	some d in input.domains
	not failing(d)
	open(d)
}

decision[d] := "deploy" if {
	some d in input.domains
	not failing(d)
	not open(d)
}
