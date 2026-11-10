# Deferred (v1 is "complete, not optimized")
- Tune C and the review threshold (0.70) on the **val** split; report test once.
- Jina v4 vs SigLIP comparison / ensemble.
- Extraction prompt tuning (null handling, number formats, few-shot); try structured outputs; smarter field-accuracy metric.
- Image downscale (max side 2000) for extraction may hurt dense documents.
- Speed: embed_image does 4 SigLIP forward passes on CPU; could skip rotations when angle 0 is confident.
