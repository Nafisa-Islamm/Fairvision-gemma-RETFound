# FairVision: Gemma and MedGemma

Local **Gemma 3 4B** and **MedGemma 4B** evaluation on the same FairVision
750-patient cohort: 250 glaucoma, 250 AMD, and 250 diabetic retinopathy cases.
This fork retains the upstream RETFound research code and adds standalone
multimodal baselines without paid API calls or retraining existing retinal models.

**Start here: [Newton GPU / Stokes CPU, step by step](equi-agent/local_vlm/STOKES_NEWTON.md).**

Clone the branch containing the new scripts:

```bash
git clone --branch fairvision-local-gemma --single-branch \
  https://github.com/Nafisa-Islamm/Fairvision-gemma-RETFound.git \
  "$HOME/Fairvision-gemma-RETFound"
cd "$HOME/Fairvision-gemma-RETFound"
```

## Workflow

1. On Stokes CPU, install the lightweight environment and prepare the selected
   patient manifest. Check all selected NPZ images where the dataset is accessible.
2. On Newton, install the separate GPU environment using a CPU allocation and
   point it at the already downloaded local Gemma/MedGemma model directories.
3. Run six-patient GPU smoke jobs before full 750-patient inference for each model.
   Use the real Newton partition/account, confirmed by the cluster inventory.
4. On Stokes CPU, report macro-F1, worst-group macro-F1, sensitivity, specificity,
   and balanced accuracy from both completed result directories.

The loader supports disease directories named Amd, Dr and Glaucoma, as well
as AMD/DR, and case variations in Test. Both models receive SLO, a representative
eight-slice OCT grid and demographic metadata; true labels are never prompted.
Runs preserve raw responses, errors and provenance and can resume interruptions.
Incomplete cohorts are not reported as complete results.

See [code and scoring details](equi-agent/local_vlm/README.md). The original
MedGemma checkpoint is google/medgemma-4b-it; this setup does not substitute
MedGemma 1.5.

## Validation

Ten targeted tests passed; the lightweight CPU installer prepared the complete
manifest in the development environment. Full pretrained inference, dataset
integrity and Newton GPU scheduling are validated by the user's smoke jobs.

Datasets and model weights remain in cluster storage. Follow the original
dataset and model licenses/access terms. The source cohort contains one missing
glaucoma label: preparation creates an audited corrected copy using tracked
source metadata, preserving the original file and patient selection.
