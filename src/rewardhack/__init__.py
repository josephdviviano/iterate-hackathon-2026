"""Reward-hacking measurements for the program synthesizer.

The synthesizer is scored by exact replay on the train transitions it can
see. The hack is a program that tabulates those transitions instead of
modelling the mechanics. This package measures that propensity and reports
it per program, without touching the committee pipeline.
"""
