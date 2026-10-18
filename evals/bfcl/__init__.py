"""BFCL (Berkeley Function Calling Leaderboard) evaluation of models served by InferScale.

Generation and scoring are done by the official, pinned bfcl-eval package in an isolated
environment (tools/bfcl). This package only prepares ID files, runs that tooling as a
subprocess, and turns its result/score files into per-entry rows and summaries.
"""
