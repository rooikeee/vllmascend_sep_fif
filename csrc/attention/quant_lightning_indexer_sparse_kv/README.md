# QuantLightningIndexerSparseKv

Only this fused operator is an optimization and acceptance target. See
[build and test instructions](../../../README.request-shared-950.md).

The fixed 950 QLI computes score and TopK, then all AIVs enter a hardware
`SyncAll()` inside the same mixed kernel. Each request is owned by one AIV,
which uses a persistent reverse map, CAS miss deduplication, a free stack and
protected timestamp eviction across all MTP query rows. There is no standalone
planner launch and no per-row LRU/hash callback. TopK is published to GM before
the barrier; request control reuses the finished TopK UB slab.

The supported QLI inputs remain FP8 E4M3 `[T,32,128]` queries, paged FP8 keys,
BF16 weights, FP32 scales, K=2048 and sparse mode 3. Planner capacity is now
per request and must cover `max_queries*2048` with at most 16 query rows.
The mutable ABI changed: rebuild both the device package and Torch extension.
Device execution, synchronization, Graph replay and performance require 950;
910 can only host a compatible cross compiler and CPU tests.

## Vendored source

The indexer is copied from
`ops-transformer/attention/quant_lightning_indexer` at revision
`2fd074e77b07ecac2fa3f2daf7901b22c4646a1b`. Its v1 kernel, original TopK
chunks, and minimal transitive headers from `lightning_indexer` and
`lightning_indexer_v2` live under `vendor/`. Original CANN Open Software
License headers are preserved. The fused copy adds a request-planning phase,
single-owner invalid-row handling, and shared scratch allocation; scoring,
trunk selection, reduction, and TopK ordering are unchanged. The host
validation and template selection are restricted to the supported target;
the unused 32-bit TopK specialization and two unused headers are omitted. The
upstream license copied from
`ops-transformer/LICENSE` is included at `vendor/LICENSE` and applies to the
vendored source inventory below.

Vendored inventory:

- `vendor/LICENSE`
- `vendor/quant_lightning_indexer/op_kernel/quant_lightning_indexer_common.h`
- `vendor/quant_lightning_indexer/op_kernel/arch35/quant_lightning_indexer_kernel_arch35.h`
- `vendor/quant_lightning_indexer/op_kernel/arch35/quant_lightning_indexer_service_cube_arch35.h`
- `vendor/quant_lightning_indexer/op_kernel/arch35/quant_lightning_indexer_service_vector_arch35.h`
- `vendor/quant_lightning_indexer/op_kernel/arch35/vf/quant_lightning_indexer_topk.h`
- `vendor/quant_lightning_indexer/op_kernel/arch35/vf/quant_lightning_indexer_vector1.h`
- `vendor/lightning_indexer/op_kernel/lightning_indexer_common.h`
- `vendor/lightning_indexer/op_kernel/arch35/common/lightning_indexer_kernel_base_arch35.h`
- `vendor/lightning_indexer/op_kernel/arch35/vf/common/vf_topk_16_gather.h`
- `vendor/lightning_indexer_v2/op_kernel/arch35/common/lightning_indexer_v2_base_arch35.h`
- `vendor/lightning_indexer_v2/op_kernel/arch35/common/lightning_indexer_v2_kernel_base_arch35.h`
- `vendor/lightning_indexer_v2/op_kernel/arch35/common/vf/lightning_indexer_v2_topk_base.h`
- `vendor/lightning_indexer_v2/op_kernel/arch35/common/vf/lightning_indexer_v2_vector1_base.h`
