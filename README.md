# autoresearch harness

A task-agnostic harness for autonomous research loops, after karpathy/autoresearch: the problem
lives in a task spec, the research method lives in a framework, and the two never mix.

```
framework/
  core/ar.py              runs and measures experiments for ANY framework, from task.json
  baseline/program.md     upstream's greedy loop (try, keep if better, else reset)
  hypothesis/program.md   hierarchical hypotheses + idea queue + world model + meta-steps,
  hypothesis/research.py  with pre-registration and planning while experiments run
tasks/
  <name>/task.md          the problem in words: goal, what may be edited, rules, measurement
  <name>/task.json        editable paths, setup, run command, metric regexes, objective, constraints
arena/
  make_workspace.py       task repo + task spec + one framework -> a workspace
  run_arm.sh              one autonomous agent per workspace, same prompt for every arm
  compare.py              compares arms by the task's own metrics, plus an isolation audit
```

Two arms built from the same task get byte-identical task files and the identical runner, so a
comparison measures the framework and nothing else. Tests (`python3 -m unittest discover tests`)
check that no framework file contains task knowledge and that the programs share every section
except the research method.
