SURVIVED — independent assessment of C CPU-prepared inputs

Reader: review_3628. Reviewed only local frozen evidence in work/gpu-executor/c-input-preparation-run-v3. No blocker found to accepting these exact CPU-prepared artifacts. This is not C model v4 admission.

Identity anchors independently recomputed:
- FREEZE.json: 2d09179e5e342591d8d64f87f2571c5db9c86360a569ed7cce5653267ee92dc6
- prepared-input-receipt.json: 09f893e2cb74744df35a30cf418d60a97e12d45f140107ccbcd5b29f81d557c2
- requests/manifest.json: 9f40147888cf3ee9c10eec979f2088db4197b9115c9289df75971210ad5fc8b5
- output-own-tmpfs.tar: 7480649fc75291f2bd7a95e9737e07f39f820b8f4a7a2d31254a55797331325a

All 24 FREEZE file records match byte lengths and SHA256. The request directory contains exactly ten regular files: eight requests, population and manifest. All nine manifest data hashes and all ten receipt file records match. The request lengths/max_tokens are 300/1, 300/32, 512/1, 512/32, 513/1, 513/32, 8192/1 and 8193/1. Token IDs are nonnegative integers; every request is the exact corresponding prefix of the 8193-token input. The 1/32 pairs differ only in max_tokens. Fixed fields consistently specify model /input/model-target, temperature 0, ignore_eos true and return_token_ids true. Population contains exactly 1319 rows with ordered unique doc_id 0..1318 and nonempty question/answer strings; no duplicate full documents. Manifest/receipt helper, asset-manifest and data-revision identities agree.

The original run record is identical across preuse, diagnosis and retrieval: one helper attempt, exit 0, elapsed 3.6725040059536695 seconds. Receipt timing matches it; epoch subtraction differs by less than one microsecond. Preuse age at helper start is 0.00641632080078125 seconds. The receipt binds helper 6e852485, admitted spec 73bfba67, inspector cf6165bd and owned-keeper binding 8c5048cc. Frozen preuse/postuse inspector observations consistently report 23 input hashes, 23 runtime files, CPUs 0–3, and the same image/spec/binding. Prior inspector admission is consumed as supplied; its implementation and external input/runtime files were not re-reviewed here.

The preserved failed Docker copy is 1536 bytes and contains only the empty directory entry; its original retrieval transport failure remains present. Successful retrieval contains exactly eleven unique, safe relative-path regular tar members, without links or special files, and every member matches the local output byte-for-byte. Regular output is 1,020,390 bytes; archive size is 1,034,240 bytes, both below the recorded 16,777,216-byte stream/output bound. Diagnosis and retrieval file hashes agree. The evidence consistently records retrieval without a helper retry.

Keeper 7a0f79ddb5f0dcb31df0a16391b29928eb701317e0dd2e71b886253ae5a8d33a, owned init/sleep PIDs/start ticks, image and isolation modes remain identical through preuse, diagnosis and retrieval. Owned PID1/observer namespace observations agree; no protected/provider/host namespace comparison was used. Stop/remove outputs name that exact keeper, absence reports no such object for it, cleanup transport exited 0, and timestamps order cleanup after retrieval.

No helper/tokenizer rerun, SSH/GPU/model/CI action, source change or frozen-file mutation was performed. These inputs and their actual receipt must still be materialized into a later reviewed model binding. All model/GPU admission gates remain separate; no model result, recovery, runtime equivalence or performance claim follows from this assessment.
