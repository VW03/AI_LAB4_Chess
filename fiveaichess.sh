#!/bin/bash
P=/home/v/PycharmProjects/AI_LAB4_Chess
cd "$P" || exit 1
env > /tmp/fiveai_env.log
tee /tmp/fiveai_in.log | env -i HOME="$HOME" PATH=/usr/bin:/bin PYTHONPATH="$P" PYTHONUNBUFFERED=1 /usr/bin/python3 -u -m fiveaichess.engines.example_engine 2>/tmp/fiveai_err.log