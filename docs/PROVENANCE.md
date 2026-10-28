# Provenance and attribution

## Source snapshot

- Source repository: `github.sydney.edu.au/yden0162/SOFT3888_TU18_04` (institutional access required).
- Requested branch: `feature/final-web-demo`.
- Source revision: `575cf726b807f5282a71797a5a4d52069bc611d2`.
- Portfolio extraction date: 2026-10-28.
- Published component: `demo_web/`, including frontend, API, adapters and CNN checkpoint.

Yuchen Deng confirmed independent implementation of this Web project. The institutional repository contains earlier shared history; this attribution concerns the Web implementation, not a claim of authorship over every historical repository contribution. The original source checkout and a Git bundle were preserved locally during migration. Unrelated repository scaffolding, private project-administration links, local caches and generated outputs are outside this portfolio snapshot.

The portfolio edition adds portable dependency pins, lazy optional-model loading, validation and concurrency fixes, robust image decoding, conditioning bug fixes, frontend request handling, automated tests, CI and documentation. It does not retrain or replace the supplied CNN.

## Bundled CNN

- File: `demo_web/backend/model/preference_controlled_cnn.pt`
- SHA-256: `4b2aba3f17496882c47b11340fabc30ffb93d97672b6c41a7e99dcd7ff734cb4`
- Size: 2,959,230 bytes.
- Trainable architecture parameters: 244,035.
- State entries: `cnn` and `optimizer`; inference reads `cnn` with strict matching.

Optimizer state alone does not establish dataset, training recipe, experiment lineage or quality. No training-reproduction or quantitative image-quality claim is made for this snapshot.

## Third-party models and metadata

| Component | Source and attribution | Distribution in this repository |
| --- | --- | --- |
| Semantic segmentation | [Meta Mask2Former, Swin-Large, ADE20K](https://huggingface.co/facebook/mask2former-swin-large-ade-semantic); [original research implementation](https://github.com/facebookresearch/Mask2Former) | Model weights downloaded on demand; not committed. See upstream model terms. |
| Instruction interpretation | [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct), Qwen team, Apache-2.0 model card | Downloaded on demand; not committed. |
| Optional small development profile | [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), Qwen team, Apache-2.0 model card | Explicit environment override; not committed. |
| ADE20K class names | `id2label` from the [official model configuration](https://huggingface.co/facebook/mask2former-swin-large-ade-semantic/blob/main/config.json) | 150 names in `z3_use/ade20k_labels.json`, allowing labels to load without downloading a model. |

Mask2Former model card labels its license as “other”; do not infer that every upstream asset has the same license as a code repository. Model cards and upstream terms remain authoritative.

## Sample images and screenshots

Images under `demo_web/frontend/public/examples/` were inherited from the source Web demo. The source snapshot did not provide original photographer or licensing metadata. They are demonstration fixtures, not claimed as original photography or offered under a new blanket license. Screenshots in `docs/images/` show the actual local application and these inherited fixtures.

## Licensing

The extracted Web component did not include a project-level license. No new project-wide open-source license has been assigned during migration. Third-party software, weights and assets retain their own terms. This repository is presented as an implementation portfolio; confirm the rights for the relevant component before redistribution or commercial reuse.
