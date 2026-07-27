#!/usr/bin/env bash

# Shared runtime for SIF-backed Slurm scripts. Supply HPC_CONTAINER_IMAGE at
# runtime or through a private, gitignored launcher configuration.

HPC_CONTAINER_RUNTIME_ROOT="${HPC_CONTAINER_RUNTIME_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
HPC_CONTAINER_IMAGE="${HPC_CONTAINER_IMAGE:-}"
HPC_CONTAINER_BINDS="${HPC_CONTAINER_BINDS:-}"
HPC_CONTAINER_FORWARD_PREFIXES="${HPC_CONTAINER_FORWARD_PREFIXES:-PYTHONNOUSERSITE,KMP_DUPLICATE_LIB_OK,MPLCONFIGDIR,CUDA_VISIBLE_DEVICES,REPORT_PLUGIN_ROOT}"
HPC_CONTAINER_GPU="${HPC_CONTAINER_GPU:-auto}"
HPC_CONTAINER_RUNTIME_ACTIVE=TRUE

case ":${PATH}:" in
  *":${HPC_CONTAINER_RUNTIME_ROOT}/bin:"*) ;;
  *) PATH="${HPC_CONTAINER_RUNTIME_ROOT}/bin:${PATH}" ;;
esac

export HPC_CONTAINER_RUNTIME_ROOT
export HPC_CONTAINER_IMAGE
export HPC_CONTAINER_BINDS
export HPC_CONTAINER_FORWARD_PREFIXES
export HPC_CONTAINER_GPU
export HPC_CONTAINER_RUNTIME_ACTIVE
export PATH

hpc_container_prepare() {
  command -v apptainer >/dev/null 2>&1 || {
    echo "apptainer is required for SIF-backed HPC execution." >&2
    return 127
  }
  [[ -n "${HPC_CONTAINER_IMAGE}" ]] || {
    echo "Set HPC_CONTAINER_IMAGE to the absolute HPC SIF path." >&2
    return 2
  }
  [[ "${HPC_CONTAINER_IMAGE}" == /* && -r "${HPC_CONTAINER_IMAGE}" ]] || {
    echo "Container SIF is not an absolute readable path: ${HPC_CONTAINER_IMAGE}" >&2
    return 2
  }
}

hpc_container_ignore_host_runtime() {
  local requested="${1:-}"
  if [[ -n "${requested}" && "${HPC_CONTAINER_RUNTIME_NOTICE_SHOWN:-FALSE}" != "TRUE" ]]; then
    echo "Container runtime active; ignoring host runtime request: ${requested}" >&2
    HPC_CONTAINER_RUNTIME_NOTICE_SHOWN=TRUE
    export HPC_CONTAINER_RUNTIME_NOTICE_SHOWN
  fi
  hpc_container_prepare
}

hpc_container_should_forward() {
  local name="$1"
  case "${name}" in
    *PASSWORD*|*PASSWD*|*TOKEN*|*SECRET*|*CREDENTIAL*|*AUTHORIZATION*|*COOKIE*|*PRIVATE_KEY*|*CLIENT_CERT*)
      return 1
      ;;
    SLURM_*|OMP_NUM_THREADS|OPENBLAS_NUM_THREADS|MKL_NUM_THREADS|VECLIB_MAXIMUM_THREADS)
      return 0
      ;;
  esac

  local prefix
  local old_ifs="${IFS}"
  IFS=","
  for prefix in ${HPC_CONTAINER_FORWARD_PREFIXES}; do
    if [[ -n "${prefix}" && "${name}" == "${prefix}"* ]]; then
      IFS="${old_ifs}"
      return 0
    fi
  done
  IFS="${old_ifs}"
  return 1
}

hpc_container_use_gpu() {
  case "${HPC_CONTAINER_GPU}" in
    1|TRUE|true|yes|YES)
      return 0
      ;;
    0|FALSE|false|no|NO)
      return 1
      ;;
    auto)
      [[ -n "${SLURM_JOB_GPUS:-${SLURM_STEP_GPUS:-}}" ]] && return 0
      [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ]] && return 0
      return 1
      ;;
    *)
      echo "HPC_CONTAINER_GPU must be auto, 1, or 0: ${HPC_CONTAINER_GPU}" >&2
      return 2
      ;;
  esac
}

hpc_apptainer_exec() {
  hpc_container_prepare

  local job_token="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}-${SLURM_ARRAY_TASK_ID:-0}-$$"
  local container_home="${TMPDIR:-/tmp}/hpc-container-home-${UID:-$(id -u)}-${job_token}"
  mkdir -p "${container_home}/cache"

  local -a command=(
    apptainer exec
    --cleanenv
    --home "${container_home}"
    --env "XDG_CACHE_HOME=${container_home}/cache"
    --env "HPC_CONTAINER_RUNTIME_ACTIVE=TRUE"
  )

  if hpc_container_use_gpu; then
    command+=(--nv)
  else
    local gpu_status=$?
    [[ "${gpu_status}" -eq 1 ]] || return "${gpu_status}"
  fi

  local name
  while IFS= read -r name; do
    if hpc_container_should_forward "${name}"; then
      command+=(--env "${name}=${!name}")
    fi
  done < <(compgen -e)

  local project_root="${HPC_PROJECT_ROOT:-${PWD}}"
  local host_home="${HOME:-}"
  if [[ -n "${host_home}" && "${project_root}" == "${host_home}" ]]; then
    echo "Refusing to bind the whole host home as the project root." >&2
    return 2
  fi
  if [[ -d "${project_root}" ]]; then
    command+=(--bind "${project_root}:${project_root}" --pwd "${project_root}")
  fi

  if [[ -n "${HPC_CONTAINER_BINDS:-}" ]]; then
    local bind_path
    local old_ifs="${IFS}"
    IFS=","
    for bind_path in ${HPC_CONTAINER_BINDS}; do
      [[ -n "${bind_path}" ]] || continue
      local bind_source="${bind_path%%:*}"
      case "${bind_path}" in
        */.ssh*|*/.gnupg*|*/.docker*|*/.aws*|*docker.sock*|*/Keychains*)
          echo "Refusing sensitive host bind: ${bind_path}" >&2
          IFS="${old_ifs}"
          return 2
          ;;
      esac
      if [[ "${bind_source}" == "/home" || "${bind_source}" == "/Users" || \
            ( -n "${host_home}" && "${bind_source}" == "${host_home}" ) ]]; then
        echo "Refusing to bind a whole host home directory: ${bind_source}" >&2
        IFS="${old_ifs}"
        return 2
      fi
      [[ -e "${bind_source}" ]] || {
        echo "Container bind source does not exist: ${bind_source}" >&2
        IFS="${old_ifs}"
        return 2
      }
      command+=(--bind "${bind_path}")
    done
    IFS="${old_ifs}"
  fi

  command+=("${HPC_CONTAINER_IMAGE}")
  command+=("$@")
  "${command[@]}"
}
