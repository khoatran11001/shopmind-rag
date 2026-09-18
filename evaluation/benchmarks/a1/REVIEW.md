# A1 qrels review checklist

The generator creates one grade-2 target and one explicit grade-0 distractor per query.

1. Reviewer 1 checks every target/distractor against catalog metadata and the transformed image.
2. Reviewer 2 independently repeats the check.
3. Record disagreements by `query_id`, adjudicate them, and keep grades in `{0, 1, 2}`.
4. Record final reviewer names/date and the adjudication count here.

Current status: seed labels generated; second-member review and adjudication pending.
