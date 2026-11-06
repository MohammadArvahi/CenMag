# CenMag

CenMag is a method for personalized federated QDoRA. It decomposes each client's
DoRA magnitude into a shared center and a persistent local residual. The backbone
is frozen and quantized with 4-bit NF4 and double quantization. The clients
share LoRA-A and retain their own LoRA-B and classifier parameters.

Each round aggregates the clients' LoRA-A and effective magnitudes. The
server reconstructs each residual relative to the new center and applies a
paired numerical recentering that preserves the effective magnitudes up to
floating-point tolerance. The local objective adds the squared residual norm
scaled by the regularization coefficient and divided by twice the number of
magnitude parameters.

## MNLI configuration

The [MNLI configuration](configs/mnli.json) specifies the model, optimizer,
data split, and evaluation settings. Training uses seed 1, stateless SGD,
and full participation from three clients. The `--rounds` argument sets the
total training duration.

## Environment and inputs

The supported environment is Linux x86-64 with CPython 3.12.14,
PyTorch 2.10.0+cu128, CUDA 12.8, Transformers 4.42.3, PEFT 0.12.0 and
bitsandbytes 0.50.0. GPU execution uses one visible CUDA device. The PyTorch build must support
the accelerator's architecture and the host must supply a compatible NVIDIA
driver.

The [input plan](manifests/direct_download_plan.json) pins the upstream
files and their SHA-256 digests. It includes
[RoBERTa-large](https://huggingface.co/FacebookAI/roberta-large/tree/722cf37b1afa9454edce342e7895e588b6ff1d59),
[GLUE/MNLI](https://huggingface.co/datasets/nyu-mll/glue/tree/bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c)
and the [GLUE metric](https://huggingface.co/spaces/evaluate-metric/glue/tree/8fcbb9b3ab802e237b12e334feff420eb25a9af1). The [environment lock](configs/local_reference.lock)
pins the complete dependency set and wheel hashes.

Run the following from the repository root on Linux. Use an unused
environment prefix and allow at least 12 GiB of scratch space for the
runtime installation, plus space for the model, dataset and token cache.

```sh
python3 online_prepare.py download --plan manifests/direct_download_plan.json \
  --root inputs --groups runtime model dataset metric
python3 online_prepare.py install --release . --root inputs --prefix .runtime
.runtime/env/bin/python cache_rebuild.py --release . --root inputs \
  --plan manifests/direct_download_plan.json
```

The cache builder uses the MNLI training and matched validation splits.
It allocates examples with the pinned label-Dirichlet splitter, partition
seed 12345 and concentration 0.5. Validation allocation uses each client's
training-label counts as its prior. The cache builder checks the hashes of
all six index arrays, the token, label and sample-ID tensors, and the
serialized cache before making it available to the runtime.

## Train and resume

Verify the environment and input hashes, then select exactly one CUDA device.

```sh
.runtime/env/bin/python cenmag.py check --inputs inputs
CUDA_VISIBLE_DEVICES=0 .runtime/env/bin/python cenmag.py start \
  --rounds 300 --out runs/experiment
```

`start` loads the pretrained backbone and initializes the adapters and
classifier with the configured seed. LoRA-B and the client magnitude
residuals start at zero. Choose any positive integer for `--rounds`.

To stop an active session safely, send one interrupt and allow the current
round to finish and its checkpoint to be written. The session also checks
for numerical failures, unstable validation behavior, and available time
and disk space before continuing.

Resume from the most recent session into a new output directory. Set
`--rounds` to the desired total, which must exceed the number of completed
rounds in the checkpoint. The examples use different targets to illustrate
extension.

```sh
.runtime/env/bin/python cenmag.py check --from runs/experiment
CUDA_VISIBLE_DEVICES=0 .runtime/env/bin/python cenmag.py resume \
  --from runs/experiment --rounds 700 --out runs/continued
```

Each completed round produces a complete checkpoint with the shared LoRA-A
and magnitude center, each client's residual, LoRA-B and classifier, the RNG
state, and the completed-round position. Recovery verifies the file hashes,
training configuration, inputs, execution environment, and contiguous round
history. It resumes at the next round. Use the same code, prepared inputs,
pinned packages, and GPU model when resuming. Keep the whole session directory
if you move it.

After pruning, each session retains the 10 most recent complete checkpoints
and the session's best checkpoint when it falls outside that window. The
two newest diagnostic bundles may contain up to three checkpoint copies
each. The metrics ledger covers every completed round.

## Validation accuracy and checks

The engine evaluates each client's personalized model on its assigned
`validation_matched` shard at every completed round. Sample-weighted
validation accuracy is the total number of correct predictions divided by the
total number of examples across the three shards. Each session records
`weighted_validation_accuracy` in `rounds.jsonl`. The metrics command reads
the complete checkpoint-verified history, including predecessor sessions.
The stop rules treat nonfinite evaluation outputs as failures.

```sh
.runtime/env/bin/python cenmag.py metrics --run runs/continued
CUDA_VISIBLE_DEVICES="" .runtime/env/bin/python -B tests/test_public_workflow.py
```

The CPU checks cover the scientific invariants, round targets, retention,
target extension, and rejection of corrupt recovery files.

## Attribution

See [NOTICE](NOTICE) and the [Apache 2.0 license](LICENSE) for the
FedSA-LoRA and FederatedScope code incorporated in the implementation.
The pretrained model and MNLI dataset are obtained separately from their
upstream publishers.

- Pengxin Guo et al. [Selective Aggregation for Low-Rank Adaptation in Federated Learning](https://github.com/Pengxin-Guo/FedSA-LoRA).
- Yuexiang Xie et al. [FederatedScope](https://github.com/alibaba/FederatedScope).
- Tim Dettmers et al. [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314).
- Shih-Yang Liu et al. [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353).
- Yinhan Liu et al. [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692).
- Adina Williams, Nikita Nangia and Samuel Bowman. [A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference](https://aclanthology.org/N18-1101/).
