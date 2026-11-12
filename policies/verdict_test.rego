package raimon.verdict_test

import rego.v1

import data.raimon.verdict

chk(id, dom, kind, t, lo, hi) := {
	"id": id, "domain": dom, "kind": kind, "threshold": t,
	"estimate": (lo + hi) / 2, "lower": lo, "upper": hi,
}

test_upper_three_values if {
	inp := {"domains": ["e0"], "checks": [
		chk("a", "*", "upper", 0.1, 0.01, 0.05),
		chk("b", "*", "upper", 0.1, 0.12, 0.2),
		chk("c", "*", "upper", 0.1, 0.05, 0.15),
	]}
	v := verdict.verdict with input as inp
	v.a == "pass"
	v.b == "fail"
	v.c == "inconclusive"
}

test_lower_three_values if {
	inp := {"domains": ["e0"], "checks": [
		chk("a", "*", "lower", 0.7, 0.75, 0.8),
		chk("b", "*", "lower", 0.7, 0.5, 0.6),
		chk("c", "*", "lower", 0.7, 0.65, 0.75),
	]}
	v := verdict.verdict with input as inp
	v.a == "pass"
	v.b == "fail"
	v.c == "inconclusive"
}

test_sufficient_never_fails if {
	inp := {"domains": ["e0"], "checks": [chk("a", "e0", "sufficient_upper", 0.05, 0.2, 0.3)]}
	v := verdict.verdict with input as inp
	v.a == "inconclusive"
}

test_point_estimate_is_two_valued if {
	inp := {"domains": ["e0"], "checks": [chk("a", "*", "upper", 0.1, 0.1, 0.1)]}
	v := verdict.verdict with input as inp
	v.a == "pass"
}

test_domain_decisions if {
	inp := {"domains": ["e0", "e1", "e2"], "checks": [
		chk("g", "*", "upper", 0.1, 0.0, 0.05),
		chk("d0", "e0", "upper", 0.1, 0.2, 0.3),
		chk("d1", "e1", "upper", 0.1, 0.05, 0.2),
		chk("d2", "e2", "upper", 0.1, 0.0, 0.05),
	]}
	d := verdict.decision with input as inp
	d.e0 == "hold"
	d.e1 == "escalate"
	d.e2 == "deploy"
}

test_global_failure_holds_every_domain if {
	inp := {"domains": ["e0", "e1"], "checks": [chk("g", "*", "upper", 0.1, 0.2, 0.3)]}
	d := verdict.decision with input as inp
	d.e0 == "hold"
	d.e1 == "hold"
}

test_alarm_only_fails_on_evidence if {
	inp := {"domains": ["e0"], "checks": [
		chk("a", "e0", "alarm_upper", 0.1, 0.0, 0.6),
		chk("b", "e0", "alarm_upper", 0.1, 0.2, 0.6),
	]}
	v := verdict.verdict with input as inp
	v.a == "pass"
	v.b == "fail"
}
