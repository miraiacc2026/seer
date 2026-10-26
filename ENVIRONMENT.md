# SEER: Environment and Reproducibility

## 1. Scope

This document records the software environment inspected for the
SEER inference pipeline. It describes the observed execution environment
and the principal Python dependencies. It is not a complete lockfile for
every transitive dependency or a specification of the entire HPC system.

## 2. Execution Environment

| Component | Observed configuration |
|---|---|
| Container runtime | Apptainer 1.3.5 |
| Container software | NVIDIA PyTorch 24.09-py3; image location supplied by the caller |
| Python | 3.10.12 |
| PyTorch | 2.5.0a0+b465a5843b.nv24.9 |
| PyTorch CUDA build | CUDA 12.6 |
| GPU used for inspection | NVIDIA H100 80GB HBM3 |

The CUDA version reported by nvidia-smi describes driver compatibility;
it is distinct from the CUDA version against which PyTorch was built.

## 3. Python Dependencies

### 3.1 Principal inference and API dependencies

The following versions were observed in the custom Python package
directory, PKG_ENV.

| Package | Version |
|---|---:|
| transformers | 4.51.0 |
| tokenizers | 0.21.4 |
| safetensors | 0.7.0 |
| huggingface-hub | 0.36.2 |
| accelerate | 1.13.0 |
| numpy | 1.26.4 |
| sentencepiece | 0.2.0 |
| protobuf | 4.25.9 |
| tqdm | 4.67.3 |
| regex | 2026.4.4 |
| packaging | 26.2 |
| filelock | 3.29.0 |
| PyYAML | 6.0.3 |
| requests | 2.33.1 |
| httpx | 0.28.1 |
| openai | 2.33.0 |
| pydantic | 2.13.3 |
| psutil | 7.2.2 |
| peft | 0.11.1 |

### 3.2 Packages provided by the container

| Package | Observed version |
|---|---:|
| torch | 2.5.0a0+b465a5843b.nv24.9 |
| torchvision | 0.20.0a0 |
| Jinja2 | 3.1.4 |
| einops | 0.8.0 |
| flash-attn | 2.4.2 |
| triton | Importable; distribution version metadata unavailable |

The custom package directory is added to PYTHONPATH. It is not a
standalone virtual environment: Python and the GPU-enabled PyTorch
installation are provided by the container.

## 4. Installation

The supplied install_inference_env.sh script installs the pinned
principal Python packages into a custom package directory while
using the specified container image.

Load a compatible container runtime in your own environment. Export
`CONTAINER_IMAGE` to select an existing image and `PKG_ENV` to select a new
package directory; neither has a deployment-specific default. Then run:

    bash install_env.sh

`install_inference_env.sh` forwards to the same installer. Existing package
directories are never overwritten. The installer additionally includes
`pysbd==0.3.4`, required by candidate construction but absent from the supplied
version inspection. This added pin is not an observed version.

The script does not install a separate PyTorch or CUDA distribution.
It uses the GPU-enabled PyTorch supplied by the container.

The custom package directory must be supplied through PYTHONPATH when
running the inference scripts.

## 5. Runtime Configuration

The inference wrappers use the following configuration variables:

- CONTAINER_IMAGE / IMAGE: path to the container image, depending on the launcher.
- MYPROJ: caller-selected writable cache root; no default host location.
- PKG_ENV: custom Python package directory.
- WORKDIR: directory containing the inference Python scripts.
- MODEL and BASE_TOKENIZER: model checkpoint and tokenizer paths.
- SRC_JSON and OUT_DIR: input dataset and output directory.
- START_IDX and END_IDX: inclusive sample index range.
- BATCH_SIZE: inference batch size.

Model checkpoints, datasets, credentials, and HPC-specific storage
are external resources and are not included in this source bundle.

External input/output paths must be supplied through the launcher's environment
variables or the Python CLI arguments. Missing required paths fail explicitly.
Repository source paths are resolved relative to the scripts. Artifact filenames
and subdirectories beneath caller-selected roots are part of the pipeline's
interface, not a fixed host filesystem layout.

API endpoints are supplied through `OPENAI_BASE_URL` or `PROXY_BASE_URL`; proxy
credentials use `PROXY_API_KEY`, separately from `OPENAI_API_KEY`. Native Batch
operations additionally require `API_SUPPORTS_BATCH=1`. No endpoint hostname is
hardcoded or used to infer capabilities. Never commit credentials or runtime logs.

## 6. Reproducibility Notes

The versions above were observed in a working environment. The install
script pins the principal packages listed in Section 3.1; pip may
resolve transitive dependencies during installation. Consequently,
this script alone does not guarantee bit-for-bit reproduction of every
dependency.

For exact environment archival, record the final installed package
inventory, the container image checksum, the relevant model checkpoint
identifiers, and the inference parameters.

The environment described here is the inspected inference environment.
Training may require additional packages or a separately validated
configuration.
