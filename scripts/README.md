# Running the AdaPBS experiments outside the notebooks

`run_glacier.py` does, for one glacier, what the notebooks
`dynamic_equifinality.ipynb` and `dynamic_runoff_sensitivity.ipynb` do:

1. AdaPBS: the plausible parameter sets (`dynamic_equifinality_adapbs`), saved
   to `<rgi_id>_adapbs.nc`;
2. the plausible runs again, with OGGM's hydrological output
   (`dynamic_hydro_rerun`), saved to `<rgi_id>_hydro.nc`, and their runoff
   metrics (`runoff_metrics`), saved to `<rgi_id>_runoff_metrics.nc`;
3. a log, `<rgi_id>_log.json` (settings, run times, errors).

The defaults are the settings of the notebooks (wide priors, 500 particles,
up to 20 iterations, 30-year spinup, 20-year initial state period). See
`python scripts/run_glacier.py --help` for all the options.

## How long does it take?

On Vernagtferner (8.6 km²), a run of the model takes about 0.2-0.3 s, and a
rerun with hydrological output about 0.7 s. With the defaults, AdaPBS needs
about 9500 runs (with the initial state period) and the rerun about 2700:
roughly 75 minutes on one processor, or 10-15 minutes on 6. The run time
grows with the glacier size (longer flowlines): time one run of a large
glacier before launching many.

## On one computer

The particles of one glacier run in parallel with `--n-workers`:

```
python scripts/run_glacier.py RGI60-11.00719 --working-dir wd --output-dir out --n-workers 6
```

The notebooks read and write the same files: with
`--output-dir example_workflows/out/wide_priors`, the notebooks load these runs
instead of computing them again (change `rgi_id` in the notebooks for another
glacier).

For several glaciers, run them one after the other (each with its pool of
processes). Two rules:

- one working directory per run happening at the same time: two runs on the
  same glacier directory would overwrite each other's files;
- don't combine `--n-workers` with OGGM's own multiprocessing
  (`use_multiprocessing=True`): OGGM's processes cannot start processes of
  their own.

## On a cluster with slurm

`submit.sh` submits one slurm array task per glacier; each task
(`run_slurm.sh`) runs `run_glacier.py` in an apptainer container, with
`--n-workers` set to the cpus of the task.

1. Copy `config.sh.example` to `config.sh` (not tracked by git) and fill it
   in: slurm partition and resources, where the outputs go, the job scratch,
   the container, and the OGGM version.
2. Write the list of glaciers, one RGI id per line (see
   `glaciers_example.txt`).
3. `scripts/submit.sh my_glaciers.txt` (or `scripts/submit.sh my_glaciers.txt
   my_config.sh`: one config file per experiment, e.g. with other
   `RUN_ARGS`)

What the scripts do, and why:

- **The code is frozen at submission.** `submit.sh` copies the package, the
  scripts, `config.sh` and the glacier list to
  `$OUTPUT_ROOT/$EXP_NAME/_snapshot_<date>` and makes the copy read-only. The
  jobs run this copy: editing the repository while jobs wait in the queue
  changes nothing, and the copy says what was run. (To delete a snapshot
  later: `chmod -R u+w` first.)
- **The container sees only what it needs**: the frozen code and the download
  cache (both read-only) and the job's own scratch. `--containall` gives it an
  empty home and environment; `--no-mount bind-paths` ignores the folders
  which the cluster's apptainer configuration may bind by default (on some
  clusters, all of `/home`, read-write). This also drops the DNS
  configuration of the node, so `/etc/resolv.conf` is bound back (read-only):
  without it, no host name resolves on the compute nodes (on our cluster, the
  login node still worked, which hid the problem). A job which tries to write elsewhere
  fails with `Read-only file system`: fix the path in the code, don't add
  binds. The job stops at the start if the frozen code is writable.
- **The download cache must be a real folder.** Binding a folder into the
  container does not bind what its symlinks point to: if the cache is made of
  links to other folders (as on some clusters), the container does not see
  them, and the jobs download what they need into their scratch (a few
  hundred MB per glacier bundle). Leave `DOWNLOAD_CACHE` empty then.
- **OGGM's version is fixed**: each job installs OGGM at `OGGM_COMMIT` (with
  pip, from GitHub) on top of the container's packages. The compute nodes need
  internet access for this; if they have none, build the environment once on
  the login node and bind it read-only instead.
- **The outputs are copied after the container exits** (`rsync` to
  `$OUTPUT_ROOT/$EXP_NAME`), even if the run failed (for the log), and the
  scratch is deleted. Slurm logs: `$OUTPUT_ROOT/$EXP_NAME/logs/`.

Choosing `SLURM_CPUS`: the particles of one iteration (500 by default) are
shared between the processes, and starting a process costs a few seconds
(each one imports OGGM). Up to about 16-32 processes per glacier is useful;
beyond that, run more glaciers at the same time instead.
