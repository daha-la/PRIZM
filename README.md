# PRIZM
Protein Ranking using Informed Zero-shot Modelling (PRIZM) is a two-phased approach that efficiently examines the mutational space using machine learning (ML) guidance without requiring high-throughput methods. PRIZM combines the specific information of small datasets with the general knowledge of pretrained zero-shot predictors to discover enhanced protein variants, thereby removing the need for large datasets common for traditional ML techniques.

PRIZM provides the necessary tools to implement all elements of the workflow easily. In the **Model Selection Phase**, the protein information of an experimental low-N dataset is parsed through a diverse collection of zero-shot models based on the [ProteinGym](https://github.com/OATML-Markslab/ProteinGym) code. The resulting predicted scores are correlated with the original experimental values, identifying the model most suitable for predicting better mutants. In the **Variant Ranking Phase**, PRIZM can be used to create, predict, and rank a large _in silico_ mutant dataset using the models identified as most suitable in the previous phase. Visualization tools are also provided, such as the plotting of individual correlations, comparisons of model performances, or the construction of single-mutant landscapes. For more information on all tools available in PRIZM, please see the example notebook.

Importantly, the best models identified by PRIZM consistently outperform the worst models across three different enzyme benchmark datasets.

## Installation
Before installing PRIZM, make sure either Anaconda or Miniconda is installed, as this will be used to manage the environments. To install PRIZM, clone the github:
```bash
git clone https://github.com/daha-la/PRIZM.git
```
To run the notebooks found in the [notebook folder](notebooks/), please create the following environment:
```bash
conda env create -f environments/PRIZM_notebook_Mac.yaml
```
or
```bash
conda env create -f environments/PRIZM_notebook_Windows.yaml
```
To run the zero-shot models in the [Modeller Module](ModellerModule/), we recommend installing PRIZM on a remote server, as some of the models require significant computational power. On the remote server, please create the full PRIZM environment:
```bash
conda env create -f environments/PRIZM.yaml
conda activate PRIZM
pip install evcouplings
```
Please note that the full PRIZM environment requires a Linux-based system. To conduct evotuning of the UniRep models, please also create a specialized environment:
```bash
conda env create -f environments/unirep_evotune.yaml
```
We also recommend running the UniRep models using this environment.

## Protein Information
PRIZM requires both the sequence, structure, and MSA of the wildtype enzyme to function. For the structure, we recommend either using a high-quality crystal structure with no gaps or a _in silico_ predicted structure using the [AlphaFold3 web server](https://alphafoldserver.com/). For the MSA, PRIZM requires the MSA to be in a specific format (A2M, with no gaps/deletions in the query sequence), and we therefore recommend using the [EVcouplings web server](https://v2.evcouplings.org/).

## Run
PRIZM consists of multiple phases. In the pre-setup, first ensure that your low-N dataset is formatted correctly and saved in the [low-N folder](data/lowN/). Your dataset file should contain three columns:
- "mutant", a column containing all the mutants in the variant the format of {WT}{POS}{MUT}, separated by a colon, such as M1A:S10A
- "mutated_sequence", a column containing the sequence of the variant
- "DMS_score", a column containing the experimental values of the variants
Secondly, save an AlphaFold structure (or crystal structure without gaps) in the [structure folder](data/protein_information/structure/), and an MSA in the a2m format in the [MSA folder](data/protein_information/msa/files/) (can be created using the [EVcouplings website](https://v2.evcouplings.org/)). Lastly, create a reference file using the [Reference Builder notebook](notebooks/Reference_builder.ipynb).

For the **EModel Selection Phase** of PRIZM, all zero-shot model submission scripts can be found in the [submission folder](/ModellerModule/submission/). Please see the [README file](ModellerModule/submission/README.md) in the submission folder for a more in-depth description. After running all models, please run the **Model Selection Phase** part of the [PRIZM notebook](/notebooks/PRIZM.ipynb) to identify the best models that have the highest correlation with your low-N dataset.

In the **Variant Ranking Phase**, a large _in silico_ library can be created. This dataset is saved in the [_in silico_ library folder](data/insilico_libraries/), and this large dataset can then be run using the best model identified in the previous phase. Please remember to update the reference file using the [Reference Builder notebook](notebooks/Reference_builder.ipynb) and change the data location variable in the [zero-shot configuration file](ModellerModule/proteingym/scripts/zero_shot_config.sh). The resulting ranked dataset can be examined using the [PRIZM notebook](/notebooks/PRIZM.ipynb) to select mutants for experimental validation.

## Collection of Zero-shot models.
PRIZM leverages pre-trained zero-shot models developed and published by other research groups and adapted in the [ProteinGym](https://github.com/OATML-Markslab/ProteinGym) workflow. We do not claim any rights to their work or associated code.
| Model           | Model Input  | Repository URL                                                                                      | Reference                                                                                              |
|------------------|--------------|-----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| ESM-1b          | Sequence     | [https://github.com/facebookresearch/esm](https://github.com/facebookresearch/esm)                 | [Rives, A. et al. (2021). Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences. PNAS, 118.](https://www.pnas.org/doi/10.1073/pnas.2016239118)                                     |
| ESM-1v          | Sequence     | [https://github.com/facebookresearch/esm](https://github.com/facebookresearch/esm)                 | [Meier, J. et al. (2021). Language models enable zero-shot prediction of the effects of mutations on protein function. NeurIPS.](https://proceedings.neurips.cc/paper/2021/hash/f51338d736f95dd42427296047067694-Abstract.html) |
| ESM-2           | Sequence     | [https://github.com/facebookresearch/esm](https://github.com/facebookresearch/esm)                 | [Lin, Z et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. Science, 379.](https://www.science.org/doi/10.1126/science.ade2574) |
| ProGen2         | Sequence     | [https://github.com/salesforce/progen](https://github.com/salesforce/progen)                       | [Nijkamp, E. et al. (2023). ProGen2: Exploring the Boundaries of Protein Language Models. Cell Systems.](https://www.sciencedirect.com/science/article/pii/S2405471223002727) |
| ProtGPT2        | Sequence     | [https://huggingface.co/nferruz/ProtGPT2](https://huggingface.co/nferruz/ProtGPT2)                 | [Ferruz, N. et al. (2022). ProtGPT2 is a deep unsupervised language model for protein design. Nature Communications, 13.](https://www.nature.com/articles/s41467-022-32007-7) |
| RITA            | Sequence     | [https://github.com/lightonai/RITA](https://github.com/lightonai/RITA)                             | [Hesslow, D. et al. (2022). RITA: a Study on Scaling Up Generative Protein Sequence Models. ArXiv.](https://arxiv.org/abs/2205.05789) |
| UniRep          | Sequence     | [https://github.com/churchlab/UniRep](https://github.com/churchlab/UniRep)                         | [Alley, E.C. et al. (2019). Unified rational protein engineering with sequence-based deep representation learning. Nature Methods.](https://www.nature.com/articles/s41592-019-0598-1)     |
| EVE             | MSA          | [https://github.com/OATML-Markslab/EVE](https://github.com/OATML-Markslab/EVE)                     | [Frazer, J. et al. (2021). Disease variant prediction with deep generative models of evolutionary data. Nature.](https://www.nature.com/articles/s41586-021-04043-8) |
| eUniRep        | MSA          | [https://github.com/chloechsu/combining-evolutionary-and-assay-labelled-data](https://github.com/chloechsu/combining-evolutionary-and-assay-labelled-data) | [Alley, E.C., Khimulya, G., Biswas, S., AlQuraishi, M., & Church, G.M. (2019). Unified rational protein engineering with sequence-based deep representation learning. Nature Methods.](https://www.nature.com/articles/s41592-019-0598-1)     |
| TranceptEVE     | MSA          | [https://github.com/OATML-Markslab/ProteinGym](https://github.com/OATML-Markslab/ProteinGym)       | [Notin, P. et al (2022). TranceptEVE: Combining Family-specific and Family-agnostic Models of Protein Sequences for Improved Fitness Prediction. NeurIPS, LMRL workshop.](https://www.biorxiv.org/content/10.1101/2022.12.07.519495v1?rss=1) |                                                              |
| Tranception     | Sequence     | [https://github.com/OATML-Markslab/Tranception](https://github.com/OATML-Markslab/Tranception)     | [Notin, P. el al. (2022). Tranception: protein fitness prediction with autoregressive transformers and inference-time retrieval. ICML.](https://proceedings.mlr.press/v162/notin22a.html) |
| ESM-IF1         | Structure    | [https://github.com/facebookresearch/esm](https://github.com/facebookresearch/esm)                 | [Hsu, C et al. (2022). Learning Inverse Folding from Millions of Predicted Structures. ICML.](https://www.biorxiv.org/content/10.1101/2022.04.10.487779v2.full) |
| ProteinMPNN     | Structure    | [https://github.com/dauparas/ProteinMPNN](https://github.com/dauparas/ProteinMPNN)                 | [Dauparas, J. et al. (2022). Robust deep learning-based protein sequence design using ProteinMPNN. Science, 378.](https://www.science.org/doi/10.1126/science.add2187) |

## Validation DMS datasets
All validation datasets were extracted from the [ProteinGym](https://github.com/OATML-Markslab/ProteinGym) benchmark library. We used the following datasets:
| ProteinGym ID                 | Protein                                    | DMS Property                  | Reference                                                                 |
|-----------------------------------|--------------------------------------------|--------------------------------|---------------------------------------------------------------------------|
| ANCSZ_Hobbs_2022                  | Tyrosine Kinase                            | Enzyme activity               | [Hobbs, H. T. et al.](https://pubmed.ncbi.nlm.nih.gov/36173161/)          |
| Q59976_STRSQ_Romero_2015          | β-glucosidase                              | Enzyme activity               | [Romero, P. A. et al.](https://www.pnas.org/doi/10.1073/pnas.1422285112) |
| VKOR1_HUMAN_Chiasson_2020_activity| Epoxide reductase                          | Enzyme activity               | [Chiasson, M. A. et al.](https://elifesciences.org/articles/58026)       |

## Reproduction of Experimental Figures
To reproduce all the experimental figures found in the PRIZM publication for fluorinase engineering, please run the notebooks in the [experimental validation folder](experimental_validation/). The analysis is split up into the characterization of [relative activity](experimental_validation/RelAct_analysis.ipynb), [kinetic parameters](experimental_validation/Kin_analysis.ipynb), and [thermal stability](experimental_validation/Tm_analysis.ipynb). All experimental validation data is also located in this folder.

## Acknowledgments
PRIZM was developed based on multiple open-source zero-shot models and builds on code from the [ProteinGym repository](https://github.com/OATML-Markslab/ProteinGym). We thank the authors of ProteinGym for making their framework publicly available under the MIT License.

## Reference
If you use PRIZM in your work, please cite the following paper:
** INSERT PAPER REFERENCE HERE **

## Contact
For any questions regarding PRIZM, please forward them to David Harding-Larsen at [dahala@dtu.dk](mailto:dahala@dtu.dk). For any collaboration proposals, please refer to Dr. Ditte Hededam Welner at [diwel@dtu.dk](mailto:diwel@dtu.dk).

## License
This project is available under the MIT license found in the [LICENSE file](LICENSE).
