from src.spc import SPCEngine


def test_weco_rule1_and_capability():
    # create sample data with one extreme outlier
    data = [100.0 for _ in range(20)]
    data[5] = 140.0  # big outlier
    samples = [{"outer_area": v} for v in data]
    spc = SPCEngine(samples)
    signals = spc.apply_weco_rules("outer_area")
    assert any("Rule1" in s for s in signals)

    # capability should be low because of large spread
    cp, cpk, pp, ppk = spc.calculate_capability("outer_area", target=100.0, tol=10.0)
    assert cp >= 0


def test_weco_rule2_3_4_detection():
    # construct sequences triggering rule2 and rule4
    base = [100.0] * 30
    # rule2: two of three beyond +2σ -> make three values high
    base[2] = 110.0
    base[3] = 111.0
    base[4] = 100.0
    # rule4: nine consecutive above mean
    for i in range(10, 19):
        base[i] = 105.0

    samples = [{"outer_area": v} for v in base]
    spc = SPCEngine(samples)
    signals = spc.apply_weco_rules("outer_area")
    assert any("Rule2" in s for s in signals) or any("Rule3" in s for s in signals) or any("Rule4" in s for s in signals)
