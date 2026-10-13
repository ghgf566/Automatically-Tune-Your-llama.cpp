"""Interpret help-vs-documentation differences without declaring unsupported features."""

# Confirmed by the pinned tools/llama-bench/llama-bench.cpp:
# help only prints '--rpc' when llama_supports_rpc() is true.
PINNED_CONDITIONALS = {
    ("llama-bench", "--rpc"): {
        "classification": "conditional_in_upstream",
        "condition": "llama_supports_rpc()",
        "source": "tools/llama-bench/llama-bench.cpp",
    },
}


def interpret_differences(program, comparison):
    """Create an advisory report; never mutate the registry or infer model support."""
    absent = comparison.get("documented_not_observed_in_help")
    extra = comparison.get("unregistered_flags_observed_in_help")
    if absent is None or extra is None:
        return {"status": "unknown", "missing_documented": None,
                "unregistered_observed": None}
    missing = []
    for flag in absent:
        rule = PINNED_CONDITIONALS.get((program, flag))
        if rule:
            missing.append({
                "flag": flag, **rule, "support": "not_confirmed",
                "reason": "Upstream displays this help flag only when the build condition holds.",
            })
        else:
            missing.append({
                "flag": flag, "classification": "not_observed_in_help",
                "support": "not_confirmed",
                "reason": "Missing from the parsed help; may be conditional or undocumented.",
            })
    new = [{
        "flag": flag, "classification": "not_in_pinned_inventory",
        "support": "observed_in_help_only",
        "reason": "Requires source and behavior review before adding to the pinned registry.",
    } for flag in extra]
    return {"status": "observed", "missing_documented": missing,
            "unregistered_observed": new}


def inspect_fit_reference(observed_flags, reference, inventory_commit):
    """Compare a curated fit parameter subset, never the complete help inventory."""
    if reference["upstream"]["commit"] != inventory_commit:
        return {"status": "reference_version_mismatch",
                "reference_commit": reference["upstream"]["commit"],
                "observed": None, "not_observed": None}
    if observed_flags is None:
        return {"status": "unknown",
                "reference_commit": inventory_commit,
                "observed": None, "not_observed": None}
    known = {entry["flag"] for entry in reference["entries"]}
    observed = set(observed_flags)
    return {"status": "curated_reference_checked",
            "reference_commit": inventory_commit,
            "observed": sorted(observed & known),
            "not_observed": sorted(known - observed),
            "meaning": "Only observed in CLI --help, not tested for model/backends."}
