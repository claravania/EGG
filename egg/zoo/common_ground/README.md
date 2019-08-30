# Common Ground

The `common_ground` is a Sender/Receiver game where each agent is equipped with a knowledge base (KB) which dynamically changes over time. We adopt a single symbol game between two agents, where in each episode of the game, one agent acts as a Fact Agent (FA), and the other one acts as a Question Agent (QA). At the beginning of the game, both agents are initialized with the same KBs, which is a vector of discrete values, e.g., [2, 1, 1]. Then, at each episode FA will be given an update about a particular fact (feature), which might have the same or different value with the old fact. FA then sends a single-symbol message to QA. QA receives a question (also a feature) which might be the same or different with the fact (feature). Given the question, message, and its (old) KB, QA needs to provide the correct answer for the given question.

This game is trained using REINFORCE algorithm, with a reward based on the cross-entropy loss of the QA's output.

We adopt 'external_game' for the game input and regimes. The inputs and labels (target outputs) of this game come from an external CSV file (see 'example_data' directory). Each row in the input file contains five fields, separated by a semi-colon ';'. The first field specifies the KB (a vector of discrete values), the second and third specify the fact feature and the fact value, while the fourth and fifth specify the question feature and the question value that must be predicted by QA. The scripts assumes that the size of KB vector as the number of feature and the number of possible values range from 0 to the largest integer found for the relevant feature. Examples of the input file can be seen under the 'example_data' directory.

Similar to 'external_game', the `game.py` script also supports two regimes, for training and testing.

For the training regime:
```bash
python -m egg.zoo.common_ground.game.py --train_data=./egg/zoo/common_ground/example_data/10k_qa_f2_c2.train \
    --validation_data=./egg/zoo/common_ground/example_data/10k_qa_f2_c2.dev --n_epoch=100 --random_seed=21 \
    --lr=1e-3 --vocab_size=1000 --checkpoint_dir=./
```

After training, a model file named `100.tar` will be saved in the current directory.
For the `dump` regime, the trained model can be run on new data, with output behaviour printed to standard output (as in this example), or to a text file (as shown below).
```bash
python -m egg.zoo.common_ground.game.py --train_data=./egg/zoo/common_ground/example_data/10k_qa_f2_c2.train   \
    --dump_data=./egg/zoo/common_ground/example_data/10k_qa_f2_c2.dev --vocab_size=1000 \
    --load_from_checkpoint=100.tar

```

To dump the results to the file `dump.txt` (again, recycling the same data we did for training for dumping), we run:
```bash
python -m egg.zoo.common_ground.game.py --train_data=./egg/zoo/common_ground/example_data/10k_qa_f2_c2.train   \
    --dump_data=./egg/zoo/common_ground/example_data/10k_qa_f2_c2.dev --vocab_size=1000 \
    --load_from_checkpoint=100.tar --dump_output=dump.txt
```

The format of the dumped output (printed to stdout or to a file) is as follows:
```
kb;fact_feature;fact_value;fa_message;question;gold_value;predicted_feature;predicted_value
```
where:
* `kb` is the KB vector
* `fact_feature` is the feature slot/index of the new fact
* `fact_value` is the feature value
* `fa_message` is the symbol used by FA
* `question` is the question feature given to QA
* `gold_value` is the gold value to predict for the question
* `predicted_feature` is the predicted feature by QA
* `predicted_value` is the predicted value by QA


## Game configuration parameters:
 * `train_data/validation_data` -- paths for the train and validation files. 
 * `data_prefix` -- paths for the train and validation files, if they both have the same prefix (train file needs to end with `.train` and validation file needs to end with `.dev`). 
 * `random_seed` -- random seed to be used (if not specified, a random value will be used).
 * `checkpoint_dir` and `checkpoint_freq` specify where the model and optimizer state will be checkpointed and how often. For instance, `--checkpoint_dir=./checkpoints --checkpoint_freq=10`, configures the checkpoints to be stored in 
     `./checkpoints` every 10 epochs.
     If `checkpooint_dir` is specified, then the model obtained at the end of training will be saved there under the name
     `{n_epochs}.tar`.
 * `load_from_checkpoint` -- loads the model and optimizer state from a checkpoint. If `--n_epochs` is larger than that
    in the checkpoint, training will continue.
 * `dump_data`/`dump_output` -- if `dump_data` is specified, input/message/output/gold label are dumped for this 
     dataset. If `dump_output` is not specified, the dump is printed to stdout, otherwise, it is written into the speficied `data_output` file.

## Model parameters:
 * `read_function` -- the function used for agent to read its KB (default: concat175).
 * `embedding_dim` -- the size of the embedding space (default: 100).
 * `hidden_dim` -- the size of the hidden layers for agents (default: 250).
 * `action_dim` -- the size of the embedding for action flag --- FA or QA (default: 10).
 * `vocab_size` -- the number of unique symbols in the vocabulary (inluding `<eos>`!). `<eos>` is conventionally mapped to 0. Default: 10.

## Training hyper-parameters:
 * `lr` -- the learning rates for the agents' parameters (it might be useful to have Sender's learning rate
 lower, as Receiver has to adjust to the changes in Sender) (default: 1e-1).
 * `optimizer` -- selects the optimizer (`adam/adagrad/sgd`, defaults to Adam).
 * `no_cuda` -- disable usage of CUDA even if it is available (by default, CUDA is used if present).
 * `fa_entropy_coeff/qa_entropy_coeff` -- the regularisation coefficients for the
 entropy term in the loss; used to encourage exploration in Reinforce (default: 1e-2).
 * `n_epochs` -- number of training epochs (default: 10).
 * `batch_size` -- size of a batch. Note that it will be capped by dataset size (e.g., if the training dataset has
    1000 rows, batch cannot be larger than 1000) (default: 32).
 * `weight_decay` -- float constant number for weight decay (default: 0.).
 * `early_stopping` -- integer specifies the threshold for training to stop if development accuracy does not improve after some threshold (default: 500).

