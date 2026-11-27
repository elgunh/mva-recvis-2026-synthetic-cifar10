# Code of the final model

Elgün Hasanov. Kaggle competition "MVA RecVis 2026 - Assignment 3: Synthetic CIFAR-10 Image Classification".

This folder contains the code of my final model. The two files I selected for the final ranking are `runs/x2_s1_final_a_20261123` (public 99.406%, private 99.345%) and `runs/x2_s1_final_ab_20261123` (public 99.373%, private 99.376%, the final score). There are no weights, images or saved features here. The report describes the method. This file says which script does what and in which order the scripts ran.

## Setup

Everything ran on one server with two RTX 2080 Ti GPUs (11 GB each), in separate Python environments:

| Environment | Packages | Used for |
|---|---|---|
| `.venv` (Python 3.8) | `requirements-server.txt` | starter ResNet-18 baseline, the DFN judge run |
| `.venv-modern` (Python 3.11) | `requirements-modern.txt` | PE-Core bigG, DINOv2, training, exports |
| `.venv-siglip2` (Python 3.11) | `requirements-siglip2.txt` | SigLIP 2 (it needs its own tokenizer) |
| `.venv-gen` (Python 3.11) | created by `setup_gen_env.py` | Stable Diffusion with diffusers 0.31.0 |

`requirements.txt` is the file of the starter repository; its license is in `../LICENSES/starter-MIT.txt`. `requirements-external-cache.txt` is the one extra package of `prepare_domainnet_cache.py`.

The run scripts expect the code and the competition files (`data/train.npz`, `validation.npz`, `test.npz`, `sample_submission.csv`) in `$HOME/recvis/A3` and the environments in `$HOME/recvis`.

Notes on the code:

- The `.sh` files are the scripts that ran on the server. They also contain waiting loops for busy GPUs, smoke tests and some side experiments, which are named below. Some of them wait for or reuse outputs of earlier experiments that are not part of the final model and are not in this repository (for example `runs/dfn_complementarity`, `runs/lowres_student_kandinsky22`, `logs/student_pilot.log`, the ViT-L download record `models/dinov2_vitl14_reg4/provenance.json`). `runs/current_recipe.json` is a small record of the best configuration that I updated by hand after each public score.
- Comments and docstrings refer to my experiment journal (`EXPERIMENTS.md`), to plan notes (`*_PLAN.md`) and to a few scripts of earlier experiments. These files are not in this repository. Five scripts (`check_dino_giant.py`, `confirm_dino_giant.py`, `check_dino_geometry.py`, `confirm_dino_geometry.py`, `external_cache.py`) save the SHA-256 of a plan note in their output, so they need a file with that name to run. The same holds for the two `synth_style_*` modules when they are run as their own experiments; the final model only imports helper functions from them.
- Experiments that I rejected before any submission (for example DINOv3 features, TransCLIP, TDA) are not in this repository. One field name in `prepare_cifar10_2_check.py` still mentions the DINOv3 check.

## Models (open weights from the Hugging Face Hub)

| Model | Hub id | Role |
|---|---|---|
| PE-Core bigG/14, 448px | `timm/PE-Core-bigG-14-448` | zero-shot prior, graph features, self-training |
| SigLIP 2 g-opt/16, 384px | `timm/ViT-gopt-16-SigLIP2-384` | zero-shot prior, graph features, self-training |
| DINOv2 ViT-g/14 reg4 | `timm/vit_giant_patch14_reg4_dinov2.lvd142m` | graph features, self-training, supervised student |
| DINOv2 ViT-L/14 reg4 | `timm/vit_large_patch14_reg4_dinov2.lvd142m` | earlier version of the graph (97.520%); ViT-g is compared with it |
| Stable Diffusion 2.1, 768px | `sd2-community/stable-diffusion-2-1` | generation of my labelled images |
| DFN5B CLIP ViT-H/14, 378px | `apple/DFN5B-CLIP-ViT-H-14-378` | judge that cleans generated images; not part of the classifier |
| SD-Turbo, SD 2.1-base | `stabilityai/sd-turbo`, `sd2-community/stable-diffusion-2-1-base` | pilots that I compared and did not keep |
| SD 1.5, Kandinsky 2.2, SDXL-Turbo | `stable-diffusion-v1-5/stable-diffusion-v1-5`, `kandinsky-community/kandinsky-2-2-decoder` (with `-prior`), `stabilityai/sdxl-turbo` | generators of earlier experiments; not used for the final model |

## Data

- Competition files in `data/`. The test file has no labels, and the scripts that load it check this.
- My generated images: `generate_style_images.py` makes them from the 13 prompt templates listed in the CIFAR-10-Warehouse paper and the 10 class names. The label is the class in the prompt. I did not download or use any CIFAR-10-Warehouse data.
- CIFAR-10.1 (v6), CIFAR-10.2 and two small selections of DomainNet images are labelled check sets for early design choices, for example before DINOv2 ViT-g replaced ViT-L in the graph. No part of the final system is trained on them. `prepare_cifar10_2_check.py` and `prepare_domainnet_cache.py` prepare two of these sets; the scripts that prepared CIFAR-10.1 and the second DomainNet selection (`external_data/dfn_check`) belong to earlier experiments and are not in this repository.

## Order of the runs

### Stage 1: zero-shot prior and graph (public 97.565%)

1. `clip_baseline.py`, once per encoder, with the 80 ImageNet templates:

   ```bash
   python clip_baseline.py --data data --output runs/pe_bigg_imagenet80_fp16 --model hf-hub:timm/PE-Core-bigG-14-448 --batch-size 32 --templates imagenet --precision fp16 --save-features
   python clip_baseline.py --data data --output runs/siglip2_gopt_imagenet80_fp16 --model hf-hub:timm/ViT-gopt-16-SigLIP2-384 --batch-size 32 --templates imagenet --precision fp16 --save-features
   ```

2. `make_flip_selection.py`: the rows that get a reflected view. A row is selected if the two encoders disagree or the 0.75/0.25 mixture is below 0.9 (6,241 test images). During the competition I wrote this mask with an inline command. The script is the same rule written as a file afterwards; I checked that it reproduces the saved mask exactly (572 validation rows, 6,241 test rows).
3. `selective_flip.py`, once per encoder: adds the reflected view on these rows.
4. `download_dino_giant.py` and `encode_dino_giant.py`: DINOv2 ViT-g/14 features.
5. `check_dino_giant.py` (pilot on CIFAR-10 validation and DomainNet images), `confirm_dino_giant.py` (CIFAR-10.2), `export_dino_giant.py` (test file `runs/dinov2_giant_candidate`). `run_giant_confirmation.py` runs the encoding, the confirmation and the export in this order.
6. `style_text_prototypes.py`, once per encoder (`runs/style_text_bigg_20261114`, `runs/style_text_siglip2_20261114`): saves the 80-template class prototypes and the embeddings of the 13 x 10 style captions. The export and check scripts of Stages 2 and 3 load these files.

`joint_graph_diffusion.py` builds the k-NN graph on the features of both encoders and propagates the mixed probabilities (k = 20, temperature 0.07, alpha 0.5). `graph_diffusion.py` is the earlier version for one encoder; run as a script it compares four values of alpha on the CIFAR-10 validation set. `check_dino_geometry.py` adds the DINOv2 features to the graph with weight 0.25. The ViT-L version (`encode_dino_geometry.py`, `confirm_dino_geometry.py`, `run_dino_confirmation.py`, `export_dino_geometry.py`) is included because the ViT-g scripts import it and compare with its file.

### Stage 2: cross-fitted self-training on the test images (public 98.048%)

| Script | Part of the final model |
|---|---|
| `run_selftrain.sh` | bigG students, split A (teacher: the Stage 1 file) |
| `run_selftrain_export.sh` | first export with these students (97.761%); it is the teacher of the SigLIP 2 and DINOv2 students |
| `run_selftrain2.sh` | SigLIP 2 students, split A |
| `run_queue_dino_r3.sh` | DINOv2 students, split A |
| `run_bagging_b.sh` | bigG students, split B |
| `run_split_cd.sh` | bigG students, split C, with strong input noise; export `bag3` = `runs/adapted_bag3_graph_20261116` (98.048%) |

Not part of the final model, but started by the same scripts: in `run_selftrain.sh` the adaptation on CIFAR-10 and a student trained on all test images without cross-fitting, bigG round 2 in `run_selftrain2.sh` (97.598%, worse), bigG round 3 in `run_queue_dino_r3.sh`, the split B students of SigLIP 2 and DINOv2 in `run_bagging_b.sh`, and split D in `run_split_cd.sh`.

Main files: `train_test_selftrain.py` (LoRA rank 8 on the linear layers of the image tower, soft targets of the teacher, one epoch on one half of the test set), `encode_test_selftrain.py` (the student predicts the other half), `train_dino_selftrain.py`, `encode_dino_selftrain.py`, `selftrain_export.py` (prior mixture and graph with the student features).

### Stage 3: supervised DINOv2 student on my generated images (public 99.406%)

1. Recipe check without test labels, `run_x1.sh`: pilots with SD-Turbo, SD 2.1-base and SD 2.1 at 768px. `prepare_synth_manifest.py kernel` chooses the reduction to 32px by the radial power spectrum, and `synth_signature.py` computes the mixing rate with the test set; these two checks use no label at all. `synth_signature.py` also scores the Stage 1 pipeline on the pilots with their prompt labels. `x1_pick.py` ranks the pilots.
2. Generation, `generate_style_images.py --generator sd21_768`: 25 DPM-Solver multistep steps, guidance 7.5, fp16. The seed of an image is a hash of generator, style, class and index. Seeds 0-1399 were generated in several sessions (`run_x1.sh`, `run_night_1117.sh`, `run_night2_1117.sh`, `run_eve_1117.sh`, `run_d18.sh`, `run_d20_scale_v2.sh`, `run_d22_generation.sh`). Seeds 1400-1641 were generated with the command of `run_d22_generation.sh` typed by hand with `--start 1400 --end 1785`; I stopped it when all seeds below 1642 were complete.
3. Manifest and cleaning: `prepare_synth_manifest.py build`, then `judge_synth.py` (an image is kept if its prompt class is in the top two of the judge). The judge reads a zero-shot run of the DFN model:

   ```bash
   python clip_baseline.py --data data --output runs/clip_vith14_dfn5b_imagenet80 --model hf-hub:apple/DFN5B-CLIP-ViT-H-14-378 --batch-size 16 --templates imagenet
   ```

4. Training, `train_synth_supervised.py`: DINOv2 ViT-g/14, LoRA rank 8, cosine head, 3 epochs, batch 32, 224px, label smoothing 0.1, flip and 2-pixel shift. The training sets grew step by step: `run_x2.sh` (first students, 6,204 images, started by `run_night_1117.sh`), `run_night2_1117.sh` (12,417 images), `run_eve_1117.sh` (20,953), `run_d18.sh` (48,636), `run_d20_scale_v2.sh` (95,433), `run_d22_scale.sh` (161,656), `run_d23_final.sh` (194,495). The final file is `s1_final_a` of `run_d23_final.sh`.
5. Export: `encode_dino_selftrain.py` (student probabilities on the test set), `x2_export.py` (S1 rule: on the 8,664 test rows where the Stage 2 file is below 0.9, the geometric mean of the two probability vectors), `audit_synth_style_submission.py` (format, ids and hash of the CSV).
6. Held-out checks with prompt labels: `x2_pilot_check.py` (seeds 0-23), `compare_synth_students.py` (seeds 400-437), `check_s1_fresh_20261120.py` and `x4_fresh_ensemble.py` (seeds 900-937 and 1350-1387; started by `run_s1_fresh_20261120.sh`, `run_d22_scale.sh`, `run_d23_final.sh`, `run_d23_fallback.sh`). These seeds are never used for training: seeds 0-23 and 400-437 are removed by `--val-seeds 24` and `--exclude-train-range 400 438`, and the other two ranges are in none of the three training manifests.

Not part of the final model, but started by the same scripts: the 336px control in `run_d18.sh`, joint training with pseudo-labelled test images in `run_eve_1117.sh`, supervised bigG in `run_night2_1117.sh` (`train_bigg_synth.py`), the other seeds and ensembles of `run_d23_final.sh` (students b, c, d), and the spare file of `run_d23_fallback.sh`, which was not needed. The report gives their results.

## Baseline

`main.py`, `model.py`, `data.py`: the starter training script. ResNet-18 from scratch, 10 epochs: 85.42% on validation, 48.205% public.

```bash
python main.py --data data --output runs/resnet18_baseline_s42 --model resnet18 --epochs 10 --batch-size 128 --workers 4 --seed 42 --train-only
```

Other scripts import `data.py` to read the competition files.

## Other files

Imported by the scripts above: `synth_style_experiment.py` and `synth_style_v2.py` (helper functions from earlier experiments with other generators), `train_lowres_student.py` (LoRA helpers), `external_cache.py` (smoothing and loading helpers; its own experiment, a few-shot cache on DomainNet images, was not used), `encode_domainnet.py` (encodes any folder of external or generated images with the frozen encoders). Run by hand: `compact_natives.py` (replaces full-size generated files by their 32px versions to save disk space).
