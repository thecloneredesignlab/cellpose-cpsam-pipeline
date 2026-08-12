#!/usr/bin/env bash

# Freeze one full-file SIF digest at submission time, then let every Slurm
# worker verify the immutable file identity with metadata and the digest of
# this small receipt.  Worker verification deliberately never re-reads the
# multi-gigabyte SIF.

broad_phenotype_container_stat_fingerprint() {
  local image="$1"
  python3 -I - "$image" <<'PY'
import json
import os
import sys

path = os.path.realpath(sys.argv[1])
st = os.stat(path, follow_symlinks=True)
print(json.dumps(
    {
        "realpath": path,
        "dev": st.st_dev,
        "inode": st.st_ino,
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "ctime_ns": st.st_ctime_ns,
    },
    sort_keys=True,
    separators=(",", ":"),
))
PY
}

broad_phenotype_capture_container_identity() {
  local image="$1" expected_sha256="$2" identity_file="$3"
  local before after observed_sha256 identity_parent identity_tmp

  [[ "$image" == /* && -r "$image" ]] || {
    echo "Container SIF is not an absolute readable path: $image" >&2
    return 2
  }
  [[ "$expected_sha256" =~ ^[0-9a-f]{64}$ ]] || {
    echo "Expected SIF SHA-256 is invalid: $expected_sha256" >&2
    return 2
  }
  [[ "$identity_file" == /* ]] || {
    echo "HPC container identity file must be absolute: $identity_file" >&2
    return 2
  }
  [[ ! -e "$identity_file" && ! -L "$identity_file" ]] || {
    echo "Refusing to replace an existing HPC container identity file: $identity_file" >&2
    return 2
  }
  command -v python3 >/dev/null 2>&1 || {
    echo "python3 is required for SIF identity metadata capture" >&2
    return 127
  }
  command -v sha256sum >/dev/null 2>&1 || {
    echo "sha256sum is required for the one-time SIF digest" >&2
    return 127
  }

  before="$(broad_phenotype_container_stat_fingerprint "$image")" || return
  observed_sha256="$(sha256sum "$image")" || return
  observed_sha256="${observed_sha256%%[[:space:]]*}"
  after="$(broad_phenotype_container_stat_fingerprint "$image")" || return
  [[ "$before" == "$after" ]] || {
    echo "Container SIF metadata changed during one-time SHA-256 capture: $image" >&2
    return 2
  }
  [[ "$observed_sha256" == "$expected_sha256" ]] || {
    echo "SIF SHA-256 mismatch: expected=$expected_sha256 observed=$observed_sha256 image=$image" >&2
    return 2
  }

  identity_parent="$(dirname "$identity_file")"
  mkdir -p "$identity_parent"
  identity_tmp="$identity_parent/.hpc_container_identity.tmp.$$"
  [[ ! -e "$identity_tmp" ]] || {
    echo "Temporary HPC container identity path already exists: $identity_tmp" >&2
    return 2
  }
  if python3 -I - "$after" "$expected_sha256" "$identity_tmp" <<'PY'
import json
import os
import sys

stat_payload = json.loads(sys.argv[1])
payload = {
    "schema_version": "broad_phenotype_hpc_container_identity_v1",
    "container_realpath": stat_payload.pop("realpath"),
    "expected_sha256": sys.argv[2],
    "stat": stat_payload,
}
output = sys.argv[3]
with open(output, "x", encoding="utf-8", newline="\n") as handle:
    json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
  then
    :
  else
    local status=$?
    rm -f -- "$identity_tmp"
    return "$status"
  fi
  mv "$identity_tmp" "$identity_file" || {
    local status=$?
    rm -f -- "$identity_tmp"
    return "$status"
  }
}

broad_phenotype_verify_container_identity() {
  local image="$1" expected_sha256="$2" identity_file="$3" expected_identity_sha256="$4"
  local observed_identity_sha256

  [[ "$image" == /* && -r "$image" ]] || {
    echo "Container SIF is not an absolute readable path: $image" >&2
    return 2
  }
  [[ "$identity_file" == /* && -s "$identity_file" ]] || {
    echo "HPC_CONTAINER_IDENTITY_FILE must be an absolute nonempty file: $identity_file" >&2
    return 2
  }
  [[ "$expected_sha256" =~ ^[0-9a-f]{64}$ ]] || {
    echo "Expected SIF SHA-256 is invalid: $expected_sha256" >&2
    return 2
  }
  [[ "$expected_identity_sha256" =~ ^[0-9a-f]{64}$ ]] || {
    echo "HPC_CONTAINER_IDENTITY_FILE_SHA256 is invalid: $expected_identity_sha256" >&2
    return 2
  }

  observed_identity_sha256="$(sha256sum "$identity_file")" || return
  observed_identity_sha256="${observed_identity_sha256%%[[:space:]]*}"
  [[ "$observed_identity_sha256" == "$expected_identity_sha256" ]] || {
    echo "HPC container identity receipt SHA-256 mismatch: expected=$expected_identity_sha256 observed=$observed_identity_sha256" >&2
    return 2
  }

  python3 -I - "$image" "$expected_sha256" "$identity_file" <<'PY' || return
import json
import os
import sys

image, expected_sha256, identity_file = sys.argv[1:]
with open(identity_file, encoding="utf-8") as handle:
    payload = json.load(handle)
if payload.get("schema_version") != "broad_phenotype_hpc_container_identity_v1":
    raise SystemExit("HPC container identity schema mismatch")
if payload.get("expected_sha256") != expected_sha256:
    raise SystemExit("HPC container identity expected SHA-256 mismatch")
realpath = os.path.realpath(image)
if payload.get("container_realpath") != realpath:
    raise SystemExit(
        "HPC container identity realpath mismatch: "
        f"frozen={payload.get('container_realpath')} observed={realpath}"
    )
st = os.stat(realpath, follow_symlinks=True)
observed = {
    "dev": st.st_dev,
    "inode": st.st_ino,
    "size": st.st_size,
    "mtime_ns": st.st_mtime_ns,
    "ctime_ns": st.st_ctime_ns,
}
if payload.get("stat") != observed:
    raise SystemExit(
        "HPC container identity metadata mismatch: "
        f"frozen={payload.get('stat')} observed={observed}"
    )
PY

  BROAD_PHENOTYPE_CONTAINER_SHA256="$expected_sha256"
  BROAD_PHENOTYPE_CONTAINER_IDENTITY_FILE_SHA256="$observed_identity_sha256"
  export BROAD_PHENOTYPE_CONTAINER_SHA256 BROAD_PHENOTYPE_CONTAINER_IDENTITY_FILE_SHA256
}

broad_phenotype_worker_verify_container_identity() {
  local image="$1" expected_sha256="$2"
  : "${HPC_CONTAINER_IDENTITY_FILE:?HPC_CONTAINER_IDENTITY_FILE is required; workers may not fall back to hashing the SIF}"
  : "${HPC_CONTAINER_IDENTITY_FILE_SHA256:?HPC_CONTAINER_IDENTITY_FILE_SHA256 is required; workers may not fall back to hashing the SIF}"
  broad_phenotype_verify_container_identity \
    "$image" \
    "$expected_sha256" \
    "$HPC_CONTAINER_IDENTITY_FILE" \
    "$HPC_CONTAINER_IDENTITY_FILE_SHA256"
}
