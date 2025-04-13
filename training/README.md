## Training

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


### Output

The training session will be given a unique ID and data (including graphs and NN params) will be saved under training/training_plots/ID.

## Visualisation

To visualise the agent, run:

`python training/visualise.py`

### Parameters:

- `l`: Path to the agent file to load
- `s`: Set render mode to human (show game)
- `h`: See other parameters (help)
