import pandas as pd

from solar_forecast.dataset import parse_nyiso_time


def test_parse_nyiso_time_est():
    df = pd.DataFrame(
        {
            "time_stamp": ["02/12/2021 00:00"],
            "time_zone": ["EST"],
            "zone_name": ["SYSTEM"],
        }
    )

    result = parse_nyiso_time(df)

    assert result.loc[0, "time_stamp"] == pd.Timestamp(
        "2021-02-12 05:00:00+00:00"
    )


def test_parse_nyiso_time_edt():
    df = pd.DataFrame(
        {
            "time_stamp": ["07/06/2022 00:00"],
            "time_zone": ["EDT"],
            "zone_name": ["SYSTEM"],
        }
    )

    result = parse_nyiso_time(df)

    assert result.loc[0, "time_stamp"] == pd.Timestamp(
        "2022-07-06 04:00:00+00:00"
    )


def test_parse_nyiso_time_normalizes_zone():
    df = pd.DataFrame(
        {
            "time_stamp": ["07/06/2022 00:00"],
            "time_zone": ["EDT"],
            "zone_name": [" system "],
        }
    )

    result = parse_nyiso_time(df)

    assert result.loc[0, "zone_name"] == "SYSTEM"
