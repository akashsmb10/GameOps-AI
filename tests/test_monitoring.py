from src.monitoring import telemetry


def test_summary_computes_rates_and_percentiles():
    for i in range(20):
        telemetry.record_request("GET", "/unit-test", 500 if i < 2 else 200, float(i))
    telemetry.record_agent({"mode": "x", "tool_calls": [{"status": "ok"}, {"status": "error"}], "citations": ["R:1"],
                            "invalid_citations": ["R:2"], "latency_ms": 5})
    s = telemetry.summary()
    ep = s["api"]["endpoints"]["/unit-test"]
    assert ep["requests"] == 20 and ep["error_rate"] == 0.1 and ep["p95_ms"] >= 18
    assert s["agent"]["failed_tool_calls"] >= 1 and s["agent"]["citation_validity"] < 1
