# Verification record

Verified on **2026-10-28** against the portfolio edition extracted from source commit `575cf726b807f5282a71797a5a4d52069bc611d2`.

## Environment

- macOS, 8GB memory; CPU inference (`OMP_NUM_THREADS=2` for the live API).
- Python 3.12.0, PyTorch 2.13.0, torchvision 0.28.0, Transformers 5.16.1.
- Node.js 22.15.0, pnpm 11.25.0, React 19.2.8, Vite 8.2.2.
- Bundled CNN checksum is recorded in [provenance](PROVENANCE.md).
- Actual optional-model smoke profile: Mask2Former Swin-Large ADE20K plus the explicitly selected Qwen2.5-0.5B-Instruct. **Default Qwen 7B and CUDA were not run.**

## Automated checks

| Check | Result | What it verifies |
| --- | --- | --- |
| `python -m pip check` | Passed | Installed Python dependency compatibility |
| `python -m pytest --cov --cov-report=term` | **79 passed** | 40 API cases and 39 inference/adapter cases |
| Backend coverage | **84.44% lines**, **74.77% branches** | 651/771 executable statements; 166/222 branch opportunities; combined coverage rounds to 82% |
| `pnpm --dir demo_web/frontend test` | **15 passed** | jsdom interaction and asynchronous-state regressions |
| `pnpm --dir demo_web/frontend test:compare` | Passed | Before/after layers and label rendering |
| `pnpm --dir demo_web/frontend build` | Passed | Production Vite build |
| `python -m demo_web.backend.test_api_smoke --max-side 64` | Passed | Actual CNN, manual/reference endpoints and downloadable PNG bytes |

Backend tests use the real bundled CNN for checkpoint compatibility, conditioning effects, output dimensions/range, and API inference. Optional model I/O is explicitly substituted with controlled semantic masks or language-model responses. This makes CI independent of large downloads and does **not** certify pretrained model accuracy.

Coverage includes image orientation, palette/CMYK/16-bit decoding, malformed uploads, file/pixel limits, non-finite and out-of-range controls, stale region sessions, concurrent service access, reference fallback, region blending, global contrast, bounded maps, invalid LLM output, and the valid ADE20K bicycle class at ID 127.

Frontend tests cover all three mode contracts, explicit segmentation, global reference defaults, source replacement during pending requests, late-response rejection, cancellation, TIFF preview errors, object URL cleanup, full reset, text-to-manual plan handoff, image-click coordinates and the comparison control.

The inherited reference-adapter and regional-map assertion scripts also passed on the untouched source checkout before changes. The new regression suites were added in this edition; their counts are not retroactive claims about the original branch.

## Real-model and browser smoke checks

| Path | Observed result | Scope |
| --- | --- | --- |
| Browser Global Fine Tune | Passed; real output and before/after UI rendered | `demo-coast.png`, 508×338; red +0.06, blue −0.06, contrast +0.06 |
| Mask2Former via `/api/segment` | Passed; returned sky, mountain, sea, rock and land | Actual pretrained weights; first call in process about 17.3s, weights already downloaded |
| Browser regional Fine Tune | Passed; selected rock, added green +0.06, generated result | Actual cached segmentation and CNN; global controls retained |
| Global reference API | Passed; output downloaded and decoded at 508×338 | 508×338 coast source and 333×500 reference, strength 0.7 |
| Semantic reference API | Passed through global fallback | Actual models on both images; no shared classes in this pair, so the fallback is expected |
| Output HTTP download | Passed | Returned PNG bytes decoded with correct dimensions; no browser-save-location claim |
| Small-model text: English | **Failed** | “Make the whole image slightly warmer.” produced an invalid plan; API returned an error |
| Small-model text: Chinese | **Instruction-following failure** | “整个画面暖一点” returned HTTP 200 and an image, but all controls were zero; this is not counted as a successful requested edit |
| Default Qwen2.5-7B | **Not tested on this machine** | CPU float32 memory requirement exceeds the practical 8GB test profile |
| CUDA / bitsandbytes 4-bit | **Not tested** | No compatible CUDA device in this environment |
| Windows | **Not tested** | Setup commands documented; old background-process helpers excluded |

The 0.5B profile therefore demonstrates that the optional integration can execute, but is **not a validated text-editing profile**. Use manual/reference modes for a reliable lightweight demonstration. Evaluate the default 7B profile on suitable hardware before claiming text-mode quality or giving a live text demonstration.

Wall times above describe individual local smoke runs, not latency guarantees or benchmarks. No PSNR, SSIM, user-study or general image-quality score was measured.

### Mask2Former load-report notes

Transformers 5.16.1 reports missing final `swin.layernorm` parameters and unexpected `relative_position_index` entries for this checkpoint. Inspection of the installed implementation showed that semantic feature extraction uses `reshaped_hidden_states` and loaded per-stage norms; the missing final norm only affects a discarded output. The position index is a deterministic, nonpersistent window buffer in the current implementation, separate from learned relative-position bias. Warnings remain visible; actual segmentation was tested. This analysis does not imply every checkpoint warning is harmless.

## Reproduce additional diagnostics

Default offline checks are listed in the root README. The inherited standalone diagnostics now require explicit model opt-in:

```bash
# CNN only; no optional model download.
python -m demo_web.backend.test_api_smoke

# Loads/downloads the segmentation model.
python -m demo_web.backend.test_api_smoke --with-segmentation

# Loads/downloads segmentation and language models. Use adequate hardware.
python -m demo_web.backend.test_full_modes --with-models
python -m demo_web.backend.smoke_test --with-models
```

The last two commands are diagnostic programs, not proof that text semantics are correct: inspect the returned plan as well as the generated image. `PIE_MODEL_LOCAL_FILES_ONLY=1` requires pre-existing local weights. `PIE_LLM_MODEL` changes the model profile explicitly.

[GitHub Actions](https://github.com/MakenzieAhern/preference-image-studio/actions/workflows/ci.yml) runs the backend suite with model-network access disabled and separately tests/builds the frontend. The badge links to the current remote result; CI does not run optional models.
