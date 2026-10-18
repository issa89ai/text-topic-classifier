# Development record

Updated October 17, 2026.

## Project separation

This exercise originated in a broader learning repository and was extracted into
its own repository on October 17. It has no runtime imports, dependency includes,
CI steps, or artifact paths requiring the PDF assistant. Each project has its own
setup instructions, dependencies and tests. Common Python tooling and the author's
MIT license are shared conventions, not a connection between the applications.

The original training script and broad roadmap remain in the source repository's
Git history. Local migration backups preserve originals as well; history was not
rewritten or published as part of this repository.

## Preservation and verification

- Preserved 35 historical local MLflow runs and relocated artifact paths.
- Verified SQLite integrity and loaded the saved final classifier; it predicted
  the space class for a satellite/orbit example.
- Three synthetic regression tests passed: experiment configurations, disjoint
  reproducible stratified splits, and fitting/prediction for every pipeline.
- No full training rerun was performed during separation. README scores refer
  to the recorded October 15 experiment, not a new benchmark.
- The original full summary omitted alpha-sweep results; the standalone script
  now includes all 14 validation runs and explicitly seeds SVM randomness.
- Removed the inherited application dependency include and unused pandas dependency.

## Limitations

Historical evaluation included test-guided tuning before the corrected split design.
The recorded final test is not a fresh external benchmark. Dependencies are not
fully locked; synthetic tests do not establish generalization accuracy. Raw datasets,
the MLflow database, and trained models remain ignored and are not published to GitHub.
