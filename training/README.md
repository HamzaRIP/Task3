## Usage

from the root directory, run:

`python training/run.py`

### Parameters:

- `max-iterations`: Maximum number of training iterations
- `num-agents`: Number of archer agents
- `max-zombies`: Maximum number of zombies
- `checkpoint-dir`: Directory to save checkpoints
- `plot-dir`: Directory to save training plots
- `eval-episodes`: Number of episodes for evaluation
- `no-live-plot`: Disable live plotting (useful for headless servers) (ie; when sshing to KUL machines)


eg:
`python training/run.py --max-iterations 500 --num-agents 1 --max-zombies 10`