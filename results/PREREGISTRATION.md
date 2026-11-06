# Pre-registration of final Test100 evaluations

Rule: a configuration may be evaluated on **Test100** only after it is listed here, with its selection evidence on **Val400**, in a commit made *before* that Test100 evaluation runs. The checkpoint selection rule is fixed here too (e.g. "final epoch" or "best Val400 AP epoch").

Exception: the upstream *released* checkpoints, and our official-recipe retrains, are reproduction targets rather than tuned candidates, so they are evaluated on Test100 directly.

| # | Registered at | Config / RUN_IDs | Selection evidence (Val400) | Checkpoint rule | Test100 result (filled after) |
|---|---|---|---|---|---|
