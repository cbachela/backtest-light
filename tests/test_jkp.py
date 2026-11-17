import numpy as np
import pandas as pd

from btlight.data.jkp import FEATURES, ISSUE_FLAGS, prepare_jkp
from btlight.data.panel import Panel


def source(months=6, companies=14):
    rng = np.random.default_rng(0)
    rows = []
    for eom in pd.date_range("2020-01-31", periods=months, freq="ME"):
        for i in range(companies):
            row = dict(eom=eom, id=float(1000 + i), excntry="FRA", me=float(companies - i),
                       rvol_252d=0.02, ret_exc_lead1m=rng.normal(0, 0.05))
            row.update({f: 1 for f in ISSUE_FLAGS})
            row.update({f: rng.normal() for f in FEATURES})
            rows.append(row)
    return pd.DataFrame(rows)


def test_monthly_panel_screen_missing_and_roundtrip(tmp_path):
    jkp = source()
    gone = jkp.id.eq(1000) & jkp.eom.eq("2020-03-31")
    jkp.loc[gone, "ret_exc_lead1m"] = np.nan  # delisted during April
    jkp.loc[jkp.id.eq(1001) & jkp.eom.eq("2020-02-29"), "be_me"] = np.nan
    jkp.loc[jkp.id.eq(1002) & jkp.eom.eq("2020-02-29"), "ret_exc_lead1m"] = -1.002
    panel, audit, summary = prepare_jkp(jkp)

    # floor(0.2 x 14) = 2 smallest dropped every month; one missing target dropped.
    assert not audit.id.isin(["1012", "1013"]).any()
    assert summary["dropped_missing_targets"] == 1 and summary["capped_targets"] == 1
    assert not ((audit.id == "1000") & (audit.eom == "2020-03-31")).any()
    assert audit.target.min() == -1
    panel.validate_portfolio_dates(range(len(panel.dates)))
    assert panel.label_end[panel.day(0)][0] == panel.entry_dates[panel.day(1)][0]

    names = list(panel.feature_names)
    row = audit.index[(audit.id == "1001") & (audit.eom == "2020-02-29")][0]
    assert panel.features[row, names.index("be_me_missing")] == 1
    assert panel.features[row, names.index("be_me_normal")] == 0
    np.testing.assert_allclose(panel.risk_variance, 0.02**2 * 21)
    panel.save(tmp_path / "monthly.npz")
    np.testing.assert_array_equal(Panel.load(tmp_path / "monthly.npz").features, panel.features)
