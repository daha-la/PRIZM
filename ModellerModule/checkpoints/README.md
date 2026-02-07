# Checkpoints
As model checkpoint files can be large, PRIZM does not come with all of these pre-downloaded. To run PRIZM, please download the following checkpoint files and save them in their respective folders. This can either be done manually (see below), or by using the [download_checkpoints.sh](./download_checkpoints.sh) script:
```bash
bash download_checkpoints.sh
```
or, if a separate checkpoint folder is utilized:
```bash
bash download_checkpoints.sh /path/to/custom/checkpoint
```
The If a separate folder is utilized, remember to also copy the content of the [ProteinMPNN](./ProteinMPNN/) folder and the "contact-regression" files from the [esm](./esm/) folder.

## CARP
Checkpoint files can be found at the [CARP Zenodo](https://zenodo.org/records/6564798). Please download checkpoints for the following models:
- 600K
- 38M
- 76M
- 640M

## ESM & MSA Transformer
Checkpoint files can be found at the [ESM Github](https://github.com/facebookresearch/esm). Please download checkpoints for the following models:
- ESM1b
- ESM1v (1-5)
- ESM-IF1
- ESM2
    - 8M
    - 35M
    - 150M
    - 650M
    - 3B
- MSA Transformer (save this in its own checkpoint folder)

## MIF & MIFST
Checkpoint files can be found at the [MIF/MIFST Zenodo](https://zenodo.org/records/6573779#.YqjXT-zMI-Q). Please download checkpoints for the following models:
- mif
- mifst

## MULAN
Directions for how to download the checkpoint files can be found at the [MULAN Github](https://github.com/DFrolova/MULAN). Please download checkpoints for the following models:
- Small


## Progen2
Checkpoint files can be found at the [ProGen2 Github](https://github.com/enijkamp/progen2). Please download checkpoints for the following models:
- Small
- Medium
- Base
- Large
The files come in tar.gz format, so you need to extract the checkpoint folders and save them in the ProGen2 folder

## ProtGPT2
All files can be found at the [ProteinMPNN Hugging Face](https://huggingface.co/nferruz/ProtGPT2/tree/main). Please download all and save them in the ProtGTP2 folder.

## ProtSSN
Directions for how to download all checkpoint files can be found at the [ProtSSN Github](https://github.com/ai4protein/ProtSSN). Please download all the checkpoint files.

## RITA
All files can be found at the RITA Hugging Face repositories, with each model having it's own entry. Please download the following models:
- [Small](https://huggingface.co/lightonai/RITA_s)
- [Medium](https://huggingface.co/lightonai/RITA_m)
- [Large](https://huggingface.co/lightonai/RITA_l)
- [XLarge](https://huggingface.co/lightonai/RITA_xl)

## SaProt
Directions for how to download the checkpoint files can be found at the [SaProt Github](https://github.com/westlake-repl/SaProt). Please download checkpoints for the following models:
- SaProt_650M_AF2

## Tranception
Directions for how to download all checkpoint files can be found at the [Tranception Github](https://github.com/OATML-Markslab/Tranception). Please download the checkpoints for the following models:
- Small
- Medium
- Large

## UniRep
Directions for how to download all checkpoint files can be found at the [UniRep Github](https://github.com/churchlab/UniRep). Please download the following weights:
- 1900_weights
- 1900_weights_random