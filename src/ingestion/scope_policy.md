# Document Scope Policy

## In-scope (classifier training classes)
The 18 common document types (~925 docs total, ~92.5% of the dataset) are treated
as in-scope legal document types for the classifier.
# TODO: list the 18 class names once labels are parsed from the raw dataset.

## Out-of-scope (OOD detector training signal, not classifier classes)
The 18 rare document types (~75 docs total, several with only 1 example) are NOT
trained as classifier classes. They are used to validate the out-of-scope detector:
a working detector should flag these as "not a known type" at inference time.

## Rationale
Classes with n=1 or n<5 cannot be reliably split into train/val/test and would give
the classifier near-zero signal per class. Treating them as OOD signal instead is
both more realistic (a real intake queue will contain document types the system
was never trained on) and statistically sound.

## Open question
Whether "capture method" (e.g. PHOTO_TABLE vs TABLE) should be a separate label
dimension from "document type" — pending team decision.
