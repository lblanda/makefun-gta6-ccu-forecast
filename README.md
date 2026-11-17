# makefun-gta6-ccu-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/lblanda/makefun-gta6-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/lblanda/makefun-gta6-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts Grand Theft Auto VI's launch-day Twitch concurrent viewers, hour by hour, for the make.fun bounty `gta6-twitch-ccu-forecast-01`.

The entry is a single Python 3.13 program (`main.py`, standard library only). It reads a case file, takes hour 1's print, and forecasts hours 2–24 as five quantiles each. The forecast is a launch-day shape model fitted at run time from the case's own `comparables`: for each hour it regresses log(viewers / hour-1 viewers) on log(hour-1 viewers) across past big launches, and widens the quantiles by the residual spread. Hour 1 of a rolling-midnight console launch is not the peak — the day dips, climbs through the European morning, and peaks in the US evening — and the rise is smaller the bigger hour 1 already is. That shape is the whole edge over the house persistence baseline.

Leave-one-out backtest over the 38 comparable launch days (pinball loss on the log scale, lower is better): house baseline 0.93 overall / 0.32 on launches with hour-1 viewers ≥ 20,000; this model 0.64 overall / 0.25 on those launches, 0.18 on the console-only launches.

## Run it

```sh
python3 main.py
```

With `MAKEFUN_INPUT_DIR` and `MAKEFUN_OUTPUT_DIR` set, it reads every `<id>.in` there and writes `<id>.out` for each: `{"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]}`.

---

<sub>Entered on [make.fun](https://make.fun/?ref=3zwarjv0&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
