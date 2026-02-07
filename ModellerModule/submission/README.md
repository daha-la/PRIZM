# Zero-shot Model Submissions
This folder contains submission scripts for running zero-shot models in PRIZM, and all scripts are intended to be executed from this directory. Before running any models, please adapt [zero-shot configuration file](../proteingym/scripts/zero_shot_config.sh) to reflect your folder structure and desired PRIZM phase. Also ensure that you have downloaded all model checkpoints (see the [checkpoint folder](../checkpoints/) for more information).

It should be noted that many scoring scripts have additional parameters. These can be modified directly in the individual scripts located in the [scoring scripts folder](../proteingym/scripts/scoring_DMS_zero_shot/).

## Recommended: Global Submission Script
We recommend running PRIZM using the [global submission script](./submit_zero_shot_global.sh). While individual model submission scripts are still provided, the recommended and default way to run models is via the global submission script, which handles:
- model ordering
- training dependencies (EVE / eUniRep)
- training reuse across datasets
- conda environment selection
- background scheduling with GPU capacity limits
- dry-run and no-op testing modes

To run the global script:
```bash
bash submit_zero_shot_global.sh --first <idx1> --last <idx2>
```
Here, "idx1" and "idx2" refer to the first and last reference file index, respectively, corresponding to the datasets processed by PRIZM. The reference file uses zero-based indexing, meaning that the index of the low-N dataset is typically at index "0", while "idx2" is inclusive, such that idx1 = idx2 will run that specific index.

The default and recommended mode is `sequential`, where each model is run one after another. Please use tmux while running the `sequential` mode to avoid any problems with SSH disconnection:
```bash
tmux new -s prizm
bash submit_zero_shot_global.sh ...
```
This let's you detach the session by using "Ctrl+B" a then "D". To re-attach the session, use:
```bash
tmux attach -t prizm
```
To run models in parallel, set the mode to `background`:
```bash
--mode background --gpu-capacity X
```
Here, "X" denotes the maximally utilized approximate GPU memory capacity (16 units by default). This will run multiple models simultaneously using approximate GPU costs. Models are only started if the capacity is not exceeded.

Many models rely on MSA-based training, and PRIZM automatically reuses trained EVE and eUniRep models across datasets when they share the same MSA. This is controlled by DONE markers located in the [EVE DONE folder](../../finetuned_models/EVE/done/) and [eUniRep DONE folder](../../finetuned_models/eUniRep/done/). DONE markers are keyed by the MSA filename (without extension), ensuring that trained models are reused across datasets and libraries that share the same MSA. Each DONE marker contains meta data relevant for model training. By default, existing DONE markers are respected, and to force retraining, use the `--overwrite-training` flag when running the global script.

To only run specific models, please use the `--only` flag, such as:
```bash
--only esm1b,gemme
```
Note that `--only` applies to scoring models, while the training steps (EVE / eUniRep) will still run automatically if required by the selected models.

To inspect what would happen without executing anything, use the `--dry-run` flag, which reports selected models, required training steps, conda environments, GPU cost estimates, MSAs selected for training, and DONE marker paths that would be used. This is the safest way to validate configuration and ordering.

To test the entire pipeline (envs, scheduling, logging) without running any real computation, use the `--submit-noop` flag. In NOOP mode:
- Training steps are replaced with harmless commands
- Scoring scripts are submitted as no-op jobs
- Conda environments are activated and verified
- Logs and PIDs are created
- No models or results are modified
- No DONE markers are written

This is the recommended way to validate conda environments, background submission, GPU scheduling logic, and SSH / tmux robustness.

## Legacy: Individual Submission Scripts
Individual submission scripts can be found in the [Legacy folder](./legacy/), and can be run using:
```bash
bash legacy/<script_name>-sh
```
All submission scripts are set up using `nohup` to execute the scoring scripts in the background, with the log information being fed to the [log file folder](logfiles). If your compute environment uses a specific submission approach - such as qsub or slurm - then please adapt all submission scripts accordingly before running them.

When running the submission scripts, please specify the DMS index of your dataset in the reference file. The reference file uses zero-indexing, meaning that the index of the low-N dataset will most likely be 0. You can also run multiple indexes by setting the "LAST_INDEX" variable to the last index desired.

While using individual submission scripts, the ordering matters. While most models can be run in arbitrary order, the MSA models require a specific order of execution to work properly. The [EVE training script](eve_train_submit.sh) script must be run as the first MSA model to obtain both trained EVE models and correct MSA weights. Afterwards, the remaining MSA models can be used to score the datasets. Furthermore, to use the evotuned UniRep model (eUniRep), please first fine-tune the UniRep model using the [UniRep evotuning script](unirep_evotune.sh), after which the eUniRep model can be used to score the datasets. As the MSA is the same between the low-N dataset and the _in silico_ libraries, the analysis in the **Exploitation Phase** does not require any new MSA models to be trained/fine-tuned.

## Model Specific Setup
When running SaProt using a structure generated by tools such as Alphafold, please delete the second-to-last columns containing Chain IDs of the PDB file (located in the [structure folder](../../data/protein_information/structure/)). We recommend using editing tools such as Vim, which allow entire columns to be selected and deleted efficiently.

To run ProtSSN, please create a new subfolder in the [ProtSSN DATASET folder](../../data/special_folders/ProtSSN/DATASET/) and copy both the dataset file and the corresponding PDB structure into it. The folder naming should match the dataset name:
```text
DATASET/
├── dataset_name1/
│ ├── dataset_name1.csv
│ └── dataset_name1.pdb
├── dataset_name2/
├── dataset_name3/
└── dataset_name4/
```