# Local demonstration checklist

Use the [root setup guide](../README.md) and review the [verified test scope](../docs/VERIFICATION.md) before presenting.

1. Start the API and frontend in two terminals. Confirm **Model Ready**; **CPU Mode** is expected on macOS.
2. Select **Try Demo Image → Fine Tune → Global**. Adjust red/blue/contrast and apply.
3. Compare the output with the before/after slider and side-by-side view. Download the PNG.
4. Demonstrate global Reference Image mode with semantic matching unchecked.
5. If optional models are installed and memory allows, select **Detect Regions**, choose a visible region, adjust and apply. Text Preference additionally requires Qwen.
6. For a text demonstration, name the actual model profile in use. Do not present the smaller development model as the default 7B checkpoint.

If the backend is unavailable, check its terminal and `/api/health`. First model loading can take time; do not repeatedly submit requests. A cached segmentation expires after eight different sources or a backend restart; detect regions again when prompted. Use smaller images on low-memory machines.

Stop both processes with `Ctrl+C` after presenting. Output images and masks remain in `demo_web/outputs/`; clear them manually when no longer needed. Use the foreground terminal commands in the root README; the original machine-specific background-process helpers are not included in this edition.
