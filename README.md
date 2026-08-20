# Five-seed overnight replication

Run from the repository root, with these three files together.

Default fresh seeds are fixed as:

`101 211 307 401 503`

Each replicate independently runs:

1. `collision_search_3x3.py` from schoolbook rank 27 to rank 26.
2. `autonomous_state_machine_3x3.py` from that seed's rank-26 checkpoint toward rank 23 using the frozen specialist Pareto beam.

The default stopping budget is 120 beam generations. All other controller parameters are defaults.

Launch in the foreground:

```bash
bash run_5seed_replication.sh
```

Or detach it:

```bash
nohup bash run_5seed_replication.sh > results/replication_5seeds.nohup.log 2>&1 &
echo $!
```

Watch progress:

```bash
tail -f results/replication_5seeds.nohup.log
```

Partial and final summary:

```bash
cat results/replication_5seeds/SUMMARY.md
```

The launcher records git provenance and continues to later seeds if one seed fails.
