# -*- coding: utf-8 -*-
"""
Stage III-3 dynamic tensile fracture test for refined 005/010/015/020 multi-H GB coverage models.

Workstation all-GB slow-loading/thick-grip version.
This keeps the user's workstation paths, LAMMPS/MSMPI settings, and directory organization.

This version applies the same treatment as the updated 0H/1H slow-grip script:
  1) all four GB types use slow loading and thicker grips;
  2) GB-fracture location validation is enabled for sigma3/sigma5/sigma11/sigma17;
  3) non-GB fracture is rejected as NON_GB_FRACTURE_REJECTED;
  4) VS Code terminal output is brief by default: [RUN], [SKIP], [DONE], [FAILED], [ALL_DONE];
  5) CSV/summary record profile parameters and crack-location diagnostics.

Run requirement:
  python -m pip install scipy numpy
"""

import csv
import json
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


# ============================================================
# 0. 工作区、模型路径与势函数：refined 005/010/015/020cov
# ============================================================
WORKDIR = Path(r"C:\Users\Admin01\Desktop\stage3_fracture_response_refined_0_25")
MULTIH_MODEL_DIR = Path(r"C:\Users\Admin01\Desktop\multiH_GB_models_refined_0_25")

LAMMPS_EXE = Path(r"C:\Users\Admin01\AppData\Local\LAMMPS 64-bit 10Dec2025-MSMPI with Python\bin\lmp.exe")
POTENTIAL_FILE = Path(r"C:\Users\Admin01\Desktop\stage3_fracture_response_refined_0_25\potential\BOP1.poly")

SIGMAS_TO_RUN = ["sigma3", "sigma5", "sigma11", "sigma17"]
# refined 0-25% 覆盖度模型。000cov/025cov/050cov/100cov
# 已由 0H/1H 和 multiH 脚本处理，这里只跑 005-020cov。
COVERAGES_TO_RUN = ["005cov", "010cov", "015cov", "020cov"]

# 可写 "all"，也可只重跑指定工况，例如：
# CASES_TO_RUN = ["sigma3_GB_005cov", "sigma11_GB_020cov"]
CASES_TO_RUN = "all"

# ============================================================
# 1. Run mode controls
# ============================================================
# Options: "fresh", "rerun", "resume_crash", "extend", "skip_done"
RUN_MODE = "fresh"
RERUN_CASES = []
RESUME_CASES = []
EXTEND_CASES = "auto_not_fractured"  # or list of case names

BACKUP_OLD_OUTPUTS = True
CLEAN_OLD_OUTPUTS = False

# ============================================================
# 1b. CPU / MPI / OpenMP parallel controls
# ============================================================
# 128 logical CPU cores on the workstation. For 16 refined cases:
# MAX_WORKERS=16 and MPI_RANKS_PER_JOB=8 -> about 128 MPI ranks total.
TOTAL_CPU_CORES = 128
RESERVED_CPU_CORES = 8

USE_MPI = True
MPIEXEC_EXE = Path(r"C:\Program Files\Microsoft MPI\Bin\mpiexec.exe")
MPI_EXTRA_ARGS = []
REQUIRE_MATCHING_MPI = True
ALLOW_MPIEXEC_PATH_FALLBACK = False
MPI_RANKS_PER_JOB = 32

MAX_WORKERS = 4
OMP_THREADS = 1
USE_OMP_PACKAGE = False
WARN_ON_OVERSUBSCRIPTION = True

# ============================================================
# 2. Preprocessing before tensile loading
# ============================================================
DO_PRE_MINIMIZATION = True
DO_PRE_EQUILIBRATION = True

PRE_MIN_FIRE = True
PRE_MIN_CG = True
PRE_MIN_ETOL = "1.0e-10"
PRE_MIN_FTOL = "1.0e-10"
PRE_MIN_MAXITER = 12000
PRE_MIN_MAXEVAL = 60000
PRE_EQ_TIME_PS = 45.0


# ============================================================
# 3. Geometry / dynamics / loading
# ============================================================
VAC_PAD_A = 80.0
GRIP_THICKNESS_A = 14.0

TARGET_TEMP_K = 10.0
TDAMP_PS = 0.1
TIMESTEP_PS = 0.001

V_PULL_A_PER_PS = 0.05
STEPS_PER_CHUNK = 1000
N_CHUNKS_INITIAL = 420
THERMO_EVERY = 200
RANDOM_SEED = 49271


# ============================================================
# 4. Restart and extend-to-fracture controls
# ============================================================
ENABLE_RESTART = True
RESTART_DIR_NAME = "restarts"
RESTART_PREFIX = "restart_dynamic"

EXTEND_UNTIL_FRACTURE = True
EXTEND_CHUNKS_PER_ROUND = 50
MAX_TOTAL_CHUNKS = 900

STOP_AFTER_FRACTURE = True
POST_FRACTURE_EXTRA_CHUNKS = 10


# ============================================================
# 5. Fracture criterion
# ============================================================
USE_CONNECTIVITY_CRITERION = True
USE_STRESS_DROP_CRITERION = True

# If True, fracture requires connectivity loss AND stress drop.
# If False, connectivity loss alone can identify geometric fracture.
REQUIRE_STRESS_DROP_WITH_CONNECTIVITY = False

FRACTURE_STRESS_DROP_RATIO = 0.30
FRACTURE_HOLD_CHUNKS = 3

CONNECT_TYPES = [1, 2]       # Fe/C load-bearing network; H is excluded.
CONNECT_CUTOFF_A = 3.0
CHECK_CONNECT_EVERY_CHUNK = 1


# ============================================================
# 6. Output controls
# ============================================================
WRITE_FRAME0 = True
WRITE_PRE_EQUILIBRATED_DATA = True
WRITE_FINAL_DATA = True
DUMP_FIELDS = "id type x y z ix iy iz xu yu zu vx vy vz"

# Keep VS Code terminal output short: only show which cases start, finish, skip, or fail.
# Detailed per-chunk information is still written to CSV/log/stdout files.
TERMINAL_BRIEF = True
SHOW_STARTUP_CONFIG = False

# For strong GBs such as sigma3 and sigma11, reject a disconnected structure
# if the crack is not located near the intended grain-boundary core.
# This does not force the model to crack at the GB; it prevents non-GB fracture
# from being accepted as a valid GB-fracture result.
REQUIRE_GB_FRACTURE_SIGMAS = {"sigma3", "sigma5", "sigma11", "sigma17"}
GB_LOCATION_TOL_A = 8.0
MANUAL_GB_Y_RANGES = {
    "sigma3": [(90.0368, 92.4893)],
    "sigma5": [(85.6508, 91.8687)],
    "sigma11": [(92.9102, 98.3433)],
    "sigma17": [(90.0171, 96.9221)],
}


# ============================================================
# 6b. Case-wise profile controls
# ============================================================
# This version uses SLOW loading and THICKER grips for all four GBs
# (sigma3/sigma5/sigma11/sigma17).  Paths, CPU/MPI settings, source folders,
# and output folder organization are intentionally untouched.
#
# Important meaning:
#   v_pull_A_per_ps is the single-side grip velocity.
#   total opening rate = 2 * v_pull_A_per_ps.
#
# Default profile is already a slow, thick-grip profile.  CASE_OVERRIDES below
# keep all eight 0H/1H cases explicit, so the summary CSV records the exact
# profile used by each case.
DEFAULT_PROFILE = {
    "profile_note": "all-GB slow-loading baseline: thick grips + long pre-equilibration + GB fracture check",
    "vac_pad_A": VAC_PAD_A,
    "grip_thickness_A": 14.0,
    "target_temp_K": TARGET_TEMP_K,
    "tdamp_ps": TDAMP_PS,
    "timestep_ps": TIMESTEP_PS,
    "random_seed": RANDOM_SEED,
    "v_pull_A_per_ps": 0.05,
    "steps_per_chunk": STEPS_PER_CHUNK,
    "n_chunks_initial": 420,
    "thermo_every": THERMO_EVERY,
    "extend_chunks_per_round": EXTEND_CHUNKS_PER_ROUND,
    "max_total_chunks": 900,
    "post_fracture_extra_chunks": POST_FRACTURE_EXTRA_CHUNKS,
    "pre_eq_time_ps": 45.0,
    "pre_min_maxiter": 12000,
    "pre_min_maxeval": 60000,
    "require_gb_fracture": True,
    "gb_location_tol_A": GB_LOCATION_TOL_A,
}

CASE_OVERRIDES = {}

def _add_sigma_profile(sigma: str, cov: str, grip: float, vpull: float, preeq: float,
                       ninit: int, maxchunks: int, miniter: int, maxeval: int, tol: float,
                       note: str):
    CASE_OVERRIDES[f"{sigma}_GB_{cov}"] = {
        "profile_note": note,
        "grip_thickness_A": grip,
        "v_pull_A_per_ps": vpull,
        "pre_eq_time_ps": preeq,
        "pre_min_maxiter": miniter,
        "pre_min_maxeval": maxeval,
        "n_chunks_initial": ninit,
        "max_total_chunks": maxchunks,
        "require_gb_fracture": True,
        "gb_location_tol_A": tol,
    }

for _cov in COVERAGES_TO_RUN:
    _add_sigma_profile("sigma3", _cov, 14.0, 0.05, 45.0, 420, 900, 12000, 60000, 11.0,
                       f"sigma3-{_cov} slow: 14A grips, 0.05 A/ps single-side pull, GB fracture required")
    _add_sigma_profile("sigma5", _cov, 14.0, 0.05, 45.0, 420, 900, 12000, 60000, 11.0,
                       f"sigma5-{_cov} slow: 14A grips, 0.05 A/ps single-side pull, GB fracture required")
    _add_sigma_profile("sigma11", _cov, 16.0, 0.04, 55.0, 520, 1100, 15000, 80000, 12.0,
                       f"sigma11-{_cov} very slow: 16A grips, 0.04 A/ps single-side pull, GB fracture required")
    _add_sigma_profile("sigma17", _cov, 14.0, 0.05, 45.0, 420, 900, 12000, 60000, 12.0,
                       f"sigma17-{_cov} slow: 14A grips, 0.05 A/ps single-side pull, GB fracture required")


def get_case_profile(case_name: str) -> dict:
    profile = dict(DEFAULT_PROFILE)
    profile.update(CASE_OVERRIDES.get(case_name, {}))
    return profile


# ============================================================
# 7. Utilities
# ============================================================
def log(msg: str):
    print(msg, flush=True)


def now_stamp():
    return time.strftime("%Y%m%d_%H%M%S")


def decode_bytes(x):
    if x is None:
        return ""
    for enc in ("utf-8", "gb18030", "latin-1"):
        try:
            return x.decode(enc)
        except Exception:
            pass
    return x.decode("utf-8", errors="ignore")


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def safe_int(x):
    try:
        return int(float(x))
    except Exception:
        return None


def ensure_dirs():
    WORKDIR.mkdir(parents=True, exist_ok=True)
    (WORKDIR / "summaries").mkdir(parents=True, exist_ok=True)




def resolve_mpiexec():
    """Return an mpiexec executable path if USE_MPI is enabled.

    This workflow uses a LAMMPS binary whose path says "MSMPI".
    Therefore, the launcher must be Microsoft MPI's mpiexec.exe, not MPICH2's mpiexec.exe.
    """
    if not USE_MPI:
        return None

    if MPIEXEC_EXE:
        p = Path(MPIEXEC_EXE)
        if p.exists():
            return str(p)
        if not ALLOW_MPIEXEC_PATH_FALLBACK:
            return None
        found = shutil.which(str(MPIEXEC_EXE))
        if found:
            return found

    if ALLOW_MPIEXEC_PATH_FALLBACK:
        found = shutil.which("mpiexec")
        if found:
            return found
    return None


def validate_mpi_launcher():
    if not USE_MPI:
        return
    mpiexec = resolve_mpiexec()
    if mpiexec is None:
        raise FileNotFoundError(
            f"USE_MPI=True, but Microsoft MPI mpiexec.exe was not found at: {MPIEXEC_EXE}. "
            "Do not use C:\\Program Files\\MPICH2\\bin\\mpiexec.exe with this MSMPI LAMMPS build."
        )

    mpiexec_lower = str(mpiexec).lower()
    lmp_lower = str(LAMMPS_EXE).lower()
    if REQUIRE_MATCHING_MPI and "msmpi" in lmp_lower and "mpich" in mpiexec_lower:
        raise RuntimeError(
            "MPI launcher mismatch: your LAMMPS executable is an MSMPI build, but mpiexec resolved to MPICH2.\n"
            f"  LAMMPS_EXE = {LAMMPS_EXE}\n"
            f"  mpiexec    = {mpiexec}\n"
            "Fix MPIEXEC_EXE to C:\\Program Files\\Microsoft MPI\\Bin\\mpiexec.exe, "
            "or install/use a LAMMPS build compiled for MPICH2."
        )


def effective_lammps_slots(n_cases=None):
    active_workers = MAX_WORKERS if n_cases is None else min(MAX_WORKERS, max(1, int(n_cases)))
    mpi_ranks = MPI_RANKS_PER_JOB if USE_MPI else 1
    omp_threads = max(1, OMP_THREADS)
    return active_workers * mpi_ranks * omp_threads


def parallel_plan_text(n_cases=None):
    active_workers = MAX_WORKERS if n_cases is None else min(MAX_WORKERS, max(1, int(n_cases)))
    mpi_ranks = MPI_RANKS_PER_JOB if USE_MPI else 1
    omp_threads = max(1, OMP_THREADS)
    slots = active_workers * mpi_ranks * omp_threads
    mode = "MPI" if USE_MPI and omp_threads == 1 else ("MPI+OpenMP" if USE_MPI else ("OpenMP" if omp_threads > 1 else "serial-per-case"))
    return (
        f"mode={mode}, active_workers={active_workers}, "
        f"mpi_ranks_per_job={mpi_ranks}, omp_threads={omp_threads}, "
        f"estimated_active_cpu_slots={slots}"
    )


def check_cpu_plan(n_cases=None):
    slots = effective_lammps_slots(n_cases)
    target = max(1, TOTAL_CPU_CORES - RESERVED_CPU_CORES)
    if WARN_ON_OVERSUBSCRIPTION and slots > TOTAL_CPU_CORES:
        log(f"[WARN] CPU oversubscription: {slots} slots requested for {TOTAL_CPU_CORES} cores.")
        log("       Reduce MAX_WORKERS, MPI_RANKS_PER_JOB, or OMP_THREADS.")
    elif slots < max(1, int(0.75 * target)):
        log(f"[WARN] CPU underuse likely: only {slots} slots requested for about {target} usable cores.")
        log("       Increase MPI_RANKS_PER_JOB and/or MAX_WORKERS after a small benchmark.")


def check_paths():
    if not LAMMPS_EXE.exists():
        raise FileNotFoundError(f"LAMMPS executable not found: {LAMMPS_EXE}")
    validate_mpi_launcher()
    if not POTENTIAL_FILE.exists():
        raise FileNotFoundError(f"Potential file not found: {POTENTIAL_FILE}")
    if USE_CONNECTIVITY_CRITERION and not HAS_SCIPY:
        raise ImportError("Connectivity criterion requires scipy. Install with: python -m pip install scipy numpy")


def build_cases():
    cases = {}
    for sigma in SIGMAS_TO_RUN:
        for cov in COVERAGES_TO_RUN:
            case_name = f"{sigma}_GB_{cov}"
            source = MULTIH_MODEL_DIR / sigma / "minimized" / f"{sigma}_{cov}_min.lmp"
            outdir = WORKDIR / sigma / cov / "results"
            cases[case_name] = {"sigma": sigma, "coverage": cov, "source": source, "outdir": outdir}
    return cases


def select_cases(cases):
    if CASES_TO_RUN == "all":
        return cases
    return {name: cases[name] for name in CASES_TO_RUN}


def read_box_bounds_from_data(data_path: Path):
    xlo = xhi = ylo = yhi = zlo = zhi = None
    with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
        for _ in range(220):
            line = f.readline()
            if not line:
                break
            s = line.strip()
            if re.search(r"\bxlo\s+xhi\b", s):
                p = s.split(); xlo, xhi = float(p[0]), float(p[1])
            elif re.search(r"\bylo\s+yhi\b", s):
                p = s.split(); ylo, yhi = float(p[0]), float(p[1])
            elif re.search(r"\bzlo\s+zhi\b", s):
                p = s.split(); zlo, zhi = float(p[0]), float(p[1])
    if None in (xlo, xhi, ylo, yhi, zlo, zhi):
        raise RuntimeError(f"Failed to parse box bounds from: {data_path}")
    return {"xlo": xlo, "xhi": xhi, "ylo": ylo, "yhi": yhi, "zlo": zlo, "zhi": zhi,
            "lx": xhi - xlo, "ly": yhi - ylo, "lz": zhi - zlo}


def parse_lammps_data_atomic(data_path: Path):
    atoms = []
    box = read_box_bounds_from_data(data_path)
    in_atoms = False
    with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s.startswith("Atoms"):
                in_atoms = True
                continue
            if in_atoms:
                if not s:
                    continue
                if re.match(r"^[A-Za-z]", s):
                    break
                parts = s.split("#")[0].split()
                if len(parts) >= 5 and parts[0].isdigit():
                    atoms.append([int(parts[0]), int(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
    if not atoms:
        raise RuntimeError(f"No atoms parsed from {data_path}")
    return box, np.array(atoms, dtype=float)


def count_h_atoms(data_path: Path):
    _, atoms = parse_lammps_data_atomic(data_path)
    return int(np.sum(atoms[:, 1].astype(int) == 3))


def compute_grip_ids_from_source(source_data: Path, bounds: dict, profile: dict):
    _, atoms = parse_lammps_data_atomic(source_data)
    ybot_hi = bounds["ylo"] + profile['grip_thickness_A']
    ytop_lo = bounds["yhi"] - profile['grip_thickness_A']
    atom_ids = atoms[:, 0].astype(int)
    types = atoms[:, 1].astype(int)
    y = atoms[:, 3]
    load_mask = np.isin(types, CONNECT_TYPES)
    bottom_ids = set(atom_ids[(y <= ybot_hi) & load_mask].tolist())
    top_ids = set(atom_ids[(y >= ytop_lo) & load_mask].tolist())
    return top_ids, bottom_ids


def backup_existing_outdir(outdir: Path):
    if outdir.exists() and BACKUP_OLD_OUTPUTS:
        backup_dir = outdir.parent / f"{outdir.name}_backup_{now_stamp()}"
        shutil.move(str(outdir), str(backup_dir))
        if not TERMINAL_BRIEF:
            log(f"[BACKUP] {outdir} -> {backup_dir}")


def prepare_outdir_for_fresh(outdir: Path, force_clean: bool):
    if outdir.exists() and force_clean:
        if BACKUP_OLD_OUTPUTS:
            backup_existing_outdir(outdir)
        elif CLEAN_OLD_OUTPUTS:
            shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / RESTART_DIR_NAME).mkdir(parents=True, exist_ok=True)
    (outdir / "chunk_dumps").mkdir(parents=True, exist_ok=True)


def build_lammps_command(input_file: Path, log_file: Path):
    env = os.environ.copy()

    # Keep all thread-based libraries aligned with the requested OpenMP setting.
    # This prevents hidden oversubscription when running many cases.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[var] = str(max(1, OMP_THREADS))

    lammps_cmd = [str(LAMMPS_EXE), "-in", str(input_file), "-log", str(log_file)]

    # Only enable LAMMPS OpenMP package if you explicitly benchmarked it.
    # For this polymorphic BOP workflow, pure MPI is usually the safer first choice.
    if USE_OMP_PACKAGE:
        lammps_cmd.extend(["-pk", "omp", str(max(1, OMP_THREADS)), "-sf", "omp"])

    if USE_MPI:
        mpiexec = resolve_mpiexec()
        if mpiexec is None:
            raise FileNotFoundError("mpiexec not found while USE_MPI=True")
        cmd = [str(mpiexec)] + list(MPI_EXTRA_ARGS) + ["-n", str(MPI_RANKS_PER_JOB)] + lammps_cmd
    else:
        cmd = lammps_cmd

    return cmd, env


def run_lammps(input_file: Path, log_file: Path, workdir: Path):
    cmd, env = build_lammps_command(input_file, log_file)
    proc = subprocess.run(cmd, cwd=workdir, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False)
    stdout_text = decode_bytes(proc.stdout)
    stderr_text = decode_bytes(proc.stderr)
    stdout_file = workdir / f"{input_file.stem}_stdout.txt"
    cmd_text = subprocess.list2cmdline([str(x) for x in cmd])
    stdout_file.write_text(
        "===== COMMAND =====\n" + cmd_text +
        "\n\n===== STDOUT =====\n" + stdout_text +
        "\n\n===== STDERR =====\n" + stderr_text,
        encoding="utf-8", errors="ignore"
    )
    return proc.returncode, stdout_text, stderr_text


def restart_file(outdir: Path, chunk: int):
    return outdir / RESTART_DIR_NAME / f"{RESTART_PREFIX}_chunk_{chunk:04d}.bin"


def latest_restart(outdir: Path):
    rdir = outdir / RESTART_DIR_NAME
    if not rdir.exists():
        return None, None
    candidates = []
    for fp in rdir.glob(f"{RESTART_PREFIX}_chunk_*.bin"):
        m = re.search(r"_chunk_(\d+)\.bin$", fp.name)
        if m:
            candidates.append((int(m.group(1)), fp))
    if not candidates:
        return None, None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1]


def parse_curve_lines(stdout_text: str):
    """Parse __CURVE__ lines from LAMMPS screen/log text.

    Some LAMMPS builds write print output to the log file but not to the
    captured subprocess stdout. Some also prepend MPI/rank text. This parser
    accepts a __CURVE__ record anywhere on a line, but ignores echoed input
    lines that still contain ${...} variables.
    """
    rows = []
    for raw_line in stdout_text.splitlines():
        line = raw_line.strip().strip('"')
        idx = line.find("__CURVE__,")
        if idx < 0 or "${" in line:
            continue
        line = line[idx:].strip().strip('"')
        parts = [p.strip().strip('"') for p in line.split(",")]
        if len(parts) != 9:
            continue
        row = {
            "chunk": safe_int(parts[1]),
            "time_ps": safe_float(parts[2]),
            "opening_A": safe_float(parts[3]),
            "strain_nominal": safe_float(parts[4]),
            "sigma_raw_gpa": safe_float(parts[5]),
            "pe_eV": safe_float(parts[6]),
            "fy_top_eVA": safe_float(parts[7]),
            "fy_bot_eVA": safe_float(parts[8]),
        }
        # Do not accept echoed/unexpanded/bad records as real curve rows.
        if row["chunk"] is None or row["time_ps"] is None or row["sigma_raw_gpa"] is None:
            continue
        rows.append(row)
    return rows


def read_text_if_exists(path: Path):
    if path is None or not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def parse_curve_from_lammps_outputs(stdout_text: str, stderr_text: str, log_file: Path):
    """Look for __CURVE__ records in stdout first, then stderr and the LAMMPS log."""
    combined = "\n".join([stdout_text or "", stderr_text or "", read_text_if_exists(log_file)])
    return parse_curve_lines(combined)


# ============================================================
# 8. LAMMPS input generators
# ============================================================
def lmp_common_pair():
    return f"""pair_style      polymorphic
pair_coeff      * * {POTENTIAL_FILE.as_posix()} Fe C H

neighbor        2.0 bin
neigh_modify    delay 0 every 1 check yes
"""


def make_prep_input(source_data: Path, prep_restart: Path, traj_file: Path, analysis_dump: Path, preeq_data: Path, bounds: dict, profile: dict):
    ybot_hi = bounds["ylo"] + profile['grip_thickness_A']
    ytop_lo = bounds["yhi"] - profile['grip_thickness_A']
    area0 = bounds["lx"] * bounds["lz"]
    pre_eq_steps = int(round(profile['pre_eq_time_ps'] / profile['timestep_ps']))

    min_text = ""
    if DO_PRE_MINIMIZATION:
        min_text += """fix             hold_bottom bottom setforce 0.0 0.0 0.0
fix             hold_top    top    setforce 0.0 0.0 0.0
"""
        if PRE_MIN_FIRE:
            min_text += f"""min_style       fire
minimize        {PRE_MIN_ETOL} {PRE_MIN_FTOL} {profile['pre_min_maxiter']} {profile['pre_min_maxeval']}
"""
        if PRE_MIN_CG:
            min_text += f"""min_style       cg
minimize        {PRE_MIN_ETOL} {PRE_MIN_FTOL} {profile['pre_min_maxiter']} {profile['pre_min_maxeval']}
"""
        min_text += """unfix           hold_bottom
unfix           hold_top
"""

    eq_text = f"""velocity        mobile create {profile['target_temp_K']:.6f} {profile['random_seed']} mom yes rot no dist gaussian
velocity        top    set 0.0 0.0 0.0
velocity        bottom set 0.0 0.0 0.0
"""
    if DO_PRE_EQUILIBRATION and pre_eq_steps > 0:
        eq_text += f"""fix             pre_nvt mobile nvt temp {profile['target_temp_K']:.6f} {profile['target_temp_K']:.6f} {profile['tdamp_ps']:.6f}
fix_modify      pre_nvt temp temp_mobile
run             {pre_eq_steps}
unfix           pre_nvt
"""

    frame0_dump = ""
    if WRITE_FRAME0:
        frame0_dump = f"""write_dump      all custom {traj_file.as_posix()} {DUMP_FIELDS} modify sort id
write_dump      all custom {analysis_dump.as_posix()} {DUMP_FIELDS} modify sort id
"""

    preeq_write = f"write_data      {preeq_data.as_posix()}\n" if WRITE_PRE_EQUILIBRATED_DATA else ""

    return f"""units           metal
dimension       3
boundary        p f p
atom_style      atomic
newton          on

read_data       {source_data.as_posix()}

{lmp_common_pair()}
thermo          {THERMO_EVERY}
thermo_style    custom step temp pe lx ly lz pyy fnorm fmax atoms

variable        gauge0 equal {bounds['ly']:.10f}
variable        area0  equal {area0:.10f}
variable        ybot_hi equal {ybot_hi:.10f}
variable        ytop_lo equal {ytop_lo:.10f}

change_box      all y delta -{profile['vac_pad_A']:.6f} {profile['vac_pad_A']:.6f} units box

region          bottom_grip block INF INF INF v_ybot_hi INF INF units box
region          top_grip    block INF INF v_ytop_lo INF INF INF units box
group           bottom region bottom_grip
group           top    region top_grip
group           mobile subtract all bottom top

timestep        {profile['timestep_ps']:.10f}

# Temperature control for mobile atoms only; subtract COM drift of mobile group.
compute         temp_mobile mobile temp/com
thermo_modify   temp temp_mobile

{min_text}
{eq_text}
reset_timestep  0

compute         fy_top top    reduce sum fy
compute         fy_bot bottom reduce sum fy
variable        ftop_now equal abs(c_fy_top)
variable        fbot_now equal abs(c_fy_bot)
variable        freact_now equal 0.5*(v_ftop_now + v_fbot_now)
variable        sigma_now equal v_freact_now/v_area0*160.21766208
variable        pe_now equal pe
variable        time_ps_now equal step*dt
variable        opening_now equal 0.0
variable        strain_now equal 0.0

run             0
print           "__CURVE__,0,${{time_ps_now}},${{opening_now}},${{strain_now}},${{sigma_now}},${{pe_now}},${{ftop_now}},${{fbot_now}}"

{frame0_dump}{preeq_write}write_restart   {prep_restart.as_posix()}
"""


def make_chunk_input(prev_restart: Path, next_restart: Path, traj_file: Path, analysis_dump: Path, bounds: dict, chunk_target: int, profile: dict):
    area0 = bounds["lx"] * bounds["lz"]
    return f"""units           metal
dimension       3
boundary        p f p
atom_style      atomic
newton          on

read_restart    {prev_restart.as_posix()}

{lmp_common_pair()}
thermo          {THERMO_EVERY}
thermo_style    custom step temp pe lx ly lz pyy fnorm fmax atoms

variable        gauge0 equal {bounds['ly']:.10f}
variable        area0  equal {area0:.10f}

timestep        {profile['timestep_ps']:.10f}

compute         temp_mobile mobile temp/com
thermo_modify   temp temp_mobile

fix             int_mobile mobile nvt temp {profile['target_temp_K']:.6f} {profile['target_temp_K']:.6f} {profile['tdamp_ps']:.6f}
fix_modify      int_mobile temp temp_mobile
fix             move_top   top    move linear 0.0  {profile['v_pull_A_per_ps']:.10f} 0.0 units box
fix             move_bot   bottom move linear 0.0 -{profile['v_pull_A_per_ps']:.10f} 0.0 units box

compute         fy_top top    reduce sum fy
compute         fy_bot bottom reduce sum fy
variable        ftop_now equal abs(c_fy_top)
variable        fbot_now equal abs(c_fy_bot)
variable        freact_now equal 0.5*(v_ftop_now + v_fbot_now)
variable        sigma_now equal v_freact_now/v_area0*160.21766208
variable        pe_now equal pe
variable        time_ps_now equal step*dt
variable        opening_now equal 2.0*{profile['v_pull_A_per_ps']:.10f}*v_time_ps_now
variable        strain_now equal v_opening_now/v_gauge0

run             {profile['steps_per_chunk']}

print           "__CURVE__,{chunk_target},${{time_ps_now}},${{opening_now}},${{strain_now}},${{sigma_now}},${{pe_now}},${{ftop_now}},${{fbot_now}}"

write_dump      all custom {traj_file.as_posix()} {DUMP_FIELDS} modify append yes sort id
write_dump      all custom {analysis_dump.as_posix()} {DUMP_FIELDS} modify sort id
write_restart   {next_restart.as_posix()}
"""


# ============================================================
# 9. Dump parsing and connectivity analysis
# ============================================================
def parse_single_frame_dump(dump_file: Path):
    with open(dump_file, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    natoms = None; box = None; fields = None; atom_start = None
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith("ITEM: NUMBER OF ATOMS"):
            natoms = int(lines[i + 1].strip()); i += 2
        elif s.startswith("ITEM: BOX BOUNDS"):
            xlo, xhi = map(float, lines[i + 1].split()[:2])
            ylo, yhi = map(float, lines[i + 2].split()[:2])
            zlo, zhi = map(float, lines[i + 3].split()[:2])
            box = {"xlo": xlo, "xhi": xhi, "ylo": ylo, "yhi": yhi, "zlo": zlo, "zhi": zhi,
                   "lx": xhi - xlo, "ly": yhi - ylo, "lz": zhi - zlo}
            i += 4
        elif s.startswith("ITEM: ATOMS"):
            fields = s.split()[2:]
            atom_start = i + 1
            break
        else:
            i += 1

    if natoms is None or fields is None or atom_start is None:
        raise RuntimeError(f"Failed to parse dump: {dump_file}")

    data = {name: [] for name in fields}
    for line in lines[atom_start: atom_start + natoms]:
        parts = line.split()
        for j, name in enumerate(fields):
            data[name].append(parts[j])

    atom_ids = np.array(data["id"], dtype=int)
    types = np.array(data["type"], dtype=int)
    if all(k in data for k in ("xu", "yu", "zu")):
        coords = np.column_stack([np.array(data["xu"], dtype=float), np.array(data["yu"], dtype=float), np.array(data["zu"], dtype=float)])
    else:
        coords = np.column_stack([np.array(data["x"], dtype=float), np.array(data["y"], dtype=float), np.array(data["z"], dtype=float)])
    return {"box": box, "id": atom_ids, "type": types, "coords": coords}


class DSU:
    def __init__(self, n):
        self.p = np.arange(n, dtype=np.int64)
        self.sz = np.ones(n, dtype=np.int64)
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.sz[ra] < self.sz[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        self.sz[ra] += self.sz[rb]


def wrap_xz(coords, box):
    out = coords.copy()
    out[:, 0] = (out[:, 0] - box["xlo"]) % box["lx"] + box["xlo"]
    out[:, 2] = (out[:, 2] - box["zlo"]) % box["lz"] + box["zlo"]
    return out


def gb_center_from_ranges(sigma: str):
    ranges = MANUAL_GB_Y_RANGES.get(sigma, [])
    if not ranges:
        return None
    y0, y1 = ranges[0]
    return 0.5 * (y0 + y1)


def is_y_in_gb_core(yval, sigma: str, tol_A: float):
    if yval is None or yval == "":
        return False
    try:
        y = float(yval)
    except Exception:
        return False
    ranges = MANUAL_GB_Y_RANGES.get(sigma, [])
    for y0, y1 in ranges:
        if (y0 - tol_A) <= y <= (y1 + tol_A):
            return True
    return False


def dominant_root(root_values):
    u, c = np.unique(root_values, return_counts=True)
    return int(u[np.argmax(c)])


def analyze_connectivity_from_dump(dump_file: Path, top_ids: set, bottom_ids: set, sigma: str, profile: dict):
    d = parse_single_frame_dump(dump_file)
    atom_ids, types, coords, box = d["id"], d["type"], d["coords"], d["box"]
    mask = np.isin(types, CONNECT_TYPES)
    ids_l, coords_l = atom_ids[mask], coords[mask]
    n = len(ids_l)

    base = {
        "top_bottom_connected": False,
        "n_components": 0,
        "largest_component_size": 0,
        "second_component_size": 0,
        "crack_y_center_A": "",
        "crack_gap_A": "",
        "fracture_in_gb_core": False,
        "fracture_location_note": "not_analyzed",
    }

    if n == 0:
        base["fracture_location_note"] = "no_load_atoms"
        return base

    id_to_local = {int(aid): i for i, aid in enumerate(ids_l.tolist())}
    top_local = [id_to_local[i] for i in top_ids if i in id_to_local]
    bottom_local = [id_to_local[i] for i in bottom_ids if i in id_to_local]
    if not top_local or not bottom_local:
        base["fracture_location_note"] = "missing_top_or_bottom_grip_atoms"
        return base

    coords_base = wrap_xz(coords_l, box)
    shifts = [(sx, 0.0, sz) for sx in (-box["lx"], 0.0, box["lx"]) for sz in (-box["lz"], 0.0, box["lz"])]
    aug_pos = []
    aug_orig = []
    for sh in shifts:
        aug_pos.append(coords_base + np.array(sh))
        aug_orig.append(np.arange(n, dtype=np.int64))
    aug_pos = np.vstack(aug_pos)
    aug_orig = np.concatenate(aug_orig)

    tree = cKDTree(aug_pos)
    neighs = tree.query_ball_point(coords_base, r=CONNECT_CUTOFF_A)
    dsu = DSU(n)
    for i, neigh in enumerate(neighs):
        for jj in neigh:
            j = int(aug_orig[jj])
            if j > i:
                dsu.union(i, j)

    roots = np.array([dsu.find(i) for i in range(n)], dtype=np.int64)
    connected = bool(set(roots[top_local].tolist()).intersection(set(roots[bottom_local].tolist())))
    _, counts = np.unique(roots, return_counts=True)
    counts = np.sort(counts)[::-1]

    out = {
        "top_bottom_connected": connected,
        "n_components": int(len(counts)),
        "largest_component_size": int(counts[0]) if len(counts) else 0,
        "second_component_size": int(counts[1]) if len(counts) > 1 else 0,
        "crack_y_center_A": "",
        "crack_gap_A": "",
        "fracture_in_gb_core": False,
        "fracture_location_note": "connected" if connected else "disconnected",
    }

    if connected:
        return out

    try:
        top_root = dominant_root(roots[top_local])
        bottom_root = dominant_root(roots[bottom_local])
        y_top_comp = coords_l[roots == top_root, 1]
        y_bottom_comp = coords_l[roots == bottom_root, 1]
        if len(y_top_comp) and len(y_bottom_comp):
            bottom_ymax = float(np.max(y_bottom_comp))
            top_ymin = float(np.min(y_top_comp))
            crack_y = 0.5 * (bottom_ymax + top_ymin)
            crack_gap = top_ymin - bottom_ymax
            in_gb = is_y_in_gb_core(crack_y, sigma, profile.get("gb_location_tol_A", GB_LOCATION_TOL_A))
            out["crack_y_center_A"] = crack_y
            out["crack_gap_A"] = crack_gap
            out["fracture_in_gb_core"] = bool(in_gb)
            out["fracture_location_note"] = "gb_core" if in_gb else "outside_gb_core"
        else:
            out["fracture_location_note"] = "disconnected_but_crack_y_unknown"
    except Exception as e:
        out["fracture_location_note"] = f"crack_location_failed:{type(e).__name__}"

    return out


# ============================================================
# 10. CSV state
# ============================================================
CURVE_FIELDS = [
    "chunk", "time_ps", "opening_A", "strain_nominal",
    "sigma_raw_gpa", "sigma_zeroed_gpa", "sigma0_gpa", "sigma_peak_gpa",
    "pe_eV", "fy_top_eVA", "fy_bot_eVA", "force_imbalance_ratio",
    "top_bottom_connected", "n_components", "largest_component_size", "second_component_size",
    "crack_y_center_A", "crack_gap_A", "fracture_in_gb_core", "fracture_location_note",
    "fracture_flag", "fracture_reason"
]


def init_curve_csv(csv_file: Path):
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=CURVE_FIELDS).writeheader()


def append_curve_row(csv_file: Path, row: dict):
    with open(csv_file, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CURVE_FIELDS)
        writer.writerow({k: row.get(k, "") for k in CURVE_FIELDS})


def read_curve_csv(csv_file: Path):
    if not csv_file.exists():
        return []
    rows = []
    with open(csv_file, "r", encoding="utf-8", errors="ignore") as f:
        for r in csv.DictReader(f):
            rr = dict(r)
            for k in ["chunk", "n_components", "largest_component_size", "second_component_size"]:
                rr[k] = safe_int(rr.get(k))
            for k in ["time_ps", "opening_A", "strain_nominal", "sigma_raw_gpa", "sigma_zeroed_gpa", "sigma0_gpa", "sigma_peak_gpa", "pe_eV", "fy_top_eVA", "fy_bot_eVA", "force_imbalance_ratio", "crack_y_center_A", "crack_gap_A"]:
                rr[k] = safe_float(rr.get(k))
            rr["top_bottom_connected"] = str(rr.get("top_bottom_connected")).lower() == "true"
            rr["fracture_in_gb_core"] = str(rr.get("fracture_in_gb_core")).lower() == "true"
            rr["fracture_flag"] = str(rr.get("fracture_flag")).lower() == "true"
            rows.append(rr)
    return rows


def get_last_curve_state(rows):
    if not rows:
        return {"last_chunk": 0, "sigma0_gpa": 0.0, "sigma_peak_gpa": 0.0, "fracture_detected": False}
    return {
        "last_chunk": max([r["chunk"] for r in rows if r.get("chunk") is not None] or [0]),
        "sigma0_gpa": rows[0].get("sigma0_gpa") or 0.0,
        "sigma_peak_gpa": max([r.get("sigma_zeroed_gpa") for r in rows if r.get("sigma_zeroed_gpa") is not None] or [0.0]),
        "fracture_detected": any(r.get("fracture_flag") for r in rows),
    }


def process_raw_curve_row(raw: dict, sigma0_gpa: float, sigma_peak_gpa: float):
    sigma_raw = raw.get("sigma_raw_gpa")
    sigma_zeroed = None if sigma_raw is None else sigma_raw - sigma0_gpa
    if sigma_zeroed is not None:
        sigma_peak_gpa = max(sigma_peak_gpa, sigma_zeroed)
    fy_top, fy_bot = raw.get("fy_top_eVA"), raw.get("fy_bot_eVA")
    force_imbalance = None
    if fy_top is not None and fy_bot is not None:
        denom = max(0.5 * (abs(fy_top) + abs(fy_bot)), 1.0e-12)
        force_imbalance = abs(abs(fy_top) - abs(fy_bot)) / denom
    out = dict(raw)
    out["sigma_zeroed_gpa"] = sigma_zeroed
    out["sigma0_gpa"] = sigma0_gpa
    out["sigma_peak_gpa"] = sigma_peak_gpa
    out["force_imbalance_ratio"] = force_imbalance
    return out, sigma_peak_gpa


def update_fracture_state(row, hold_counter, sigma_name: str, profile: dict):
    connected = row.get("top_bottom_connected")
    sigma = row.get("sigma_zeroed_gpa")
    peak = row.get("sigma_peak_gpa")
    conn_condition = (connected is False)
    stress_condition = sigma is not None and peak is not None and peak > 1.0e-9 and sigma < FRACTURE_STRESS_DROP_RATIO * peak

    if USE_CONNECTIVITY_CRITERION and USE_STRESS_DROP_CRITERION and REQUIRE_STRESS_DROP_WITH_CONNECTIVITY:
        current, reason = conn_condition and stress_condition, "connectivity_lost_and_stress_drop"
    elif USE_CONNECTIVITY_CRITERION:
        current, reason = conn_condition, "top_bottom_connectivity_lost"
    elif USE_STRESS_DROP_CRITERION:
        current, reason = stress_condition, "stress_drop"
    else:
        current, reason = False, ""

    # Special check for sigma3/sigma11: if the sample disconnects outside the
    # intended GB core, reject this result instead of accepting it as GB fracture.
    if current and profile.get("require_gb_fracture", False) and row.get("fracture_in_gb_core") is not True:
        return False, 0, "non_gb_fracture_rejected"

    hold_counter = hold_counter + 1 if current else 0
    return hold_counter >= FRACTURE_HOLD_CHUNKS, hold_counter, reason if hold_counter >= FRACTURE_HOLD_CHUNKS else ""


# ============================================================
# 11. Case runner
# ============================================================
def make_case_files(case_name: str, outdir: Path):
    return {
        "inp_prep": outdir / f"{case_name}_prep.in",
        "log_prep": outdir / f"{case_name}_prep.log",
        "curve_csv": outdir / "stress_strain_dynamic.csv",
        "summary_json": outdir / "run_summary.json",
        "traj_file": outdir / f"{case_name}_dynamic_trajectory.lammpstrj",
        "preeq_data": outdir / "pre_equilibrated.lmp",
        "final_data": outdir / "final_dynamic.lmp",
        "chunk_dump_dir": outdir / "chunk_dumps",
    }


def should_run_case(case_name, outdir):
    summary_json = make_case_files(case_name, outdir)["summary_json"]
    if RUN_MODE == "skip_done":
        return (not summary_json.exists()), "SKIPPED_DONE" if summary_json.exists() else "RUN"
    if RUN_MODE == "rerun":
        return (case_name in RERUN_CASES or not RERUN_CASES), "RERUN"
    if RUN_MODE == "resume_crash":
        return (case_name in RESUME_CASES or not RESUME_CASES), "RESUME_CRASH"
    if RUN_MODE == "extend":
        if EXTEND_CASES == "auto_not_fractured":
            if not summary_json.exists():
                return False, "NO_SUMMARY_TO_EXTEND"
            try:
                js = json.loads(summary_json.read_text(encoding="utf-8"))
                profile = get_case_profile(case_name)
                need = js.get("fracture_detected") is False and js.get("actual_total_chunks", 0) < profile["max_total_chunks"]
                return need, "EXTEND" if need else "SKIPPED_NO_EXTEND_NEEDED"
            except Exception:
                return False, "BAD_SUMMARY"
        return (case_name in EXTEND_CASES), "EXTEND"
    return True, "FRESH"


def run_prep(case_name, source, outdir, files, bounds, profile):
    inp_text = make_prep_input(source, restart_file(outdir, 0), files["traj_file"], files["chunk_dump_dir"] / "chunk_0000.dump", files["preeq_data"], bounds, profile)
    files["inp_prep"].write_text(inp_text, encoding="utf-8", errors="ignore")
    ret, stdout, stderr = run_lammps(files["inp_prep"], files["log_prep"], outdir)
    if ret != 0:
        raise RuntimeError(f"PREP LAMMPS failed for {case_name}. See {files['log_prep']} and {outdir / (files['inp_prep'].stem + '_stdout.txt')}")
    rows = parse_curve_from_lammps_outputs(stdout, stderr, files["log_prep"])
    if not rows:
        raise RuntimeError(f"No prep curve row parsed for {case_name}. Check {files['log_prep']} and {outdir / (files['inp_prep'].stem + '_stdout.txt')}")
    sigma0 = rows[-1]["sigma_raw_gpa"] if rows[-1]["sigma_raw_gpa"] is not None else 0.0
    return rows[-1], sigma0


def run_one_chunk(case_name, outdir, files, bounds, chunk_target, profile):
    prev_restart = restart_file(outdir, chunk_target - 1)
    next_restart = restart_file(outdir, chunk_target)
    if not prev_restart.exists():
        raise FileNotFoundError(f"Previous restart not found: {prev_restart}")
    inp_file = outdir / f"{case_name}_chunk_{chunk_target:04d}.in"
    log_file = outdir / f"{case_name}_chunk_{chunk_target:04d}.log"
    dump_file = files["chunk_dump_dir"] / f"chunk_{chunk_target:04d}.dump"
    inp_file.write_text(make_chunk_input(prev_restart, next_restart, files["traj_file"], dump_file, bounds, chunk_target, profile), encoding="utf-8", errors="ignore")
    ret, stdout, stderr = run_lammps(inp_file, log_file, outdir)
    if ret != 0:
        raise RuntimeError(f"Chunk {chunk_target} LAMMPS failed for {case_name}. See {log_file} and {outdir / (inp_file.stem + '_stdout.txt')}")
    rows = parse_curve_from_lammps_outputs(stdout, stderr, log_file)
    if not rows:
        raise RuntimeError(f"No curve row parsed at chunk {chunk_target} for {case_name}. Check {log_file} and {outdir / (inp_file.stem + '_stdout.txt')}")
    return rows[-1], dump_file


def run_case(case_name, info):
    sigma, cov, source, outdir = info["sigma"], info["coverage"], info["source"], info["outdir"]
    profile = get_case_profile(case_name)
    if not source.exists():
        return {"model": case_name, "sigma": sigma, "coverage": cov, "status": "FAILED_SOURCE_NOT_FOUND", "source": str(source)}

    run_it, mode_reason = should_run_case(case_name, outdir)
    if not run_it:
        log(f"[SKIP] {case_name} -> {mode_reason}")
        return {"model": case_name, "sigma": sigma, "coverage": cov, "N_H": count_h_atoms(source), "status": mode_reason, "source": str(source)}

    force_fresh = RUN_MODE in ("fresh", "rerun") or mode_reason in ("FRESH", "RERUN")
    prepare_outdir_for_fresh(outdir, force_clean=force_fresh)
    files = make_case_files(case_name, outdir)
    bounds = read_box_bounds_from_data(source)
    top_ids, bottom_ids = compute_grip_ids_from_source(source, bounds, profile)
    nh = count_h_atoms(source)

    log(f"[RUN] {case_name} | mode={RUN_MODE}/{mode_reason} | N_H={nh}")
    if not TERMINAL_BRIEF:
        log(f"      source={source}")

    if force_fresh:
        init_curve_csv(files["curve_csv"])
        raw0, sigma0 = run_prep(case_name, source, outdir, files, bounds, profile)
        sigma_peak = 0.0
        row0, sigma_peak = process_raw_curve_row(raw0, sigma0, sigma_peak)
        conn = analyze_connectivity_from_dump(files["chunk_dump_dir"] / "chunk_0000.dump", top_ids, bottom_ids, sigma, profile) if USE_CONNECTIVITY_CRITERION else {}
        row0.update(conn)
        row0["fracture_flag"] = False
        row0["fracture_reason"] = ""
        append_curve_row(files["curve_csv"], row0)
        start_chunk = 0
        target_chunk = profile['n_chunks_initial']
    else:
        latest_chunk, latest_fp = latest_restart(outdir)
        if latest_fp is None:
            raise RuntimeError(f"No restart found for non-fresh run: {case_name}")
        rows_old = read_curve_csv(files["curve_csv"])
        state = get_last_curve_state(rows_old)
        sigma0 = state["sigma0_gpa"]
        sigma_peak = state["sigma_peak_gpa"]
        start_chunk = latest_chunk
        target_chunk = profile['n_chunks_initial'] if RUN_MODE == "resume_crash" else min(start_chunk + profile['extend_chunks_per_round'], profile['max_total_chunks'])

    current_chunk = start_chunk
    hold_counter = 0
    fracture_detected = False
    non_gb_rejected = False
    max_loop_target = target_chunk

    while current_chunk < max_loop_target:
        next_chunk = current_chunk + 1
        raw, dump_file = run_one_chunk(case_name, outdir, files, bounds, next_chunk, profile)
        row, sigma_peak = process_raw_curve_row(raw, sigma0, sigma_peak)
        conn = analyze_connectivity_from_dump(dump_file, top_ids, bottom_ids, sigma, profile) if USE_CONNECTIVITY_CRITERION and next_chunk % CHECK_CONNECT_EVERY_CHUNK == 0 else {}
        row.update(conn)
        if not fracture_detected:
            fractured_now, hold_counter, reason = update_fracture_state(row, hold_counter, sigma, profile)
            if reason == "non_gb_fracture_rejected":
                row["fracture_flag"] = False
                row["fracture_reason"] = reason
                append_curve_row(files["curve_csv"], row)
                current_chunk = next_chunk
                non_gb_rejected = True
                break
            row["fracture_flag"] = fractured_now
            row["fracture_reason"] = reason
            if fractured_now:
                fracture_detected = True
                if STOP_AFTER_FRACTURE:
                    max_loop_target = min(max_loop_target, next_chunk + profile['post_fracture_extra_chunks'])
        else:
            row["fracture_flag"] = True
            row["fracture_reason"] = "post_fracture"
        append_curve_row(files["curve_csv"], row)
        current_chunk = next_chunk
        if not TERMINAL_BRIEF:
            sig = row.get("sigma_zeroed_gpa")
            sigtxt = "nan" if sig is None else f"{sig:.4f}"
            log(f"      chunk {current_chunk:04d}: strain={row.get('strain_nominal'):.5f}, sigma0ed={sigtxt} GPa, connected={row.get('top_bottom_connected')}, fracture={row.get('fracture_flag')}")

    # Automatic extend until fracture or max chunk
    rows_current = read_curve_csv(files["curve_csv"])
    state = get_last_curve_state(rows_current)
    while EXTEND_UNTIL_FRACTURE and (not non_gb_rejected) and not state["fracture_detected"] and current_chunk < profile['max_total_chunks'] and RUN_MODE in ("fresh", "rerun", "extend"):
        ext_target = min(current_chunk + profile['extend_chunks_per_round'], profile['max_total_chunks'])
        if not TERMINAL_BRIEF:
            log(f"[EXTEND] {case_name}: no fracture by chunk {current_chunk}; extend to {ext_target}")
        while current_chunk < ext_target:
            next_chunk = current_chunk + 1
            raw, dump_file = run_one_chunk(case_name, outdir, files, bounds, next_chunk, profile)
            row, sigma_peak = process_raw_curve_row(raw, sigma0, sigma_peak)
            row.update(analyze_connectivity_from_dump(dump_file, top_ids, bottom_ids, sigma, profile) if USE_CONNECTIVITY_CRITERION else {})
            fractured_now, hold_counter, reason = update_fracture_state(row, hold_counter, sigma, profile)
            if reason == "non_gb_fracture_rejected":
                row["fracture_flag"] = False
                row["fracture_reason"] = reason
                append_curve_row(files["curve_csv"], row)
                current_chunk = next_chunk
                non_gb_rejected = True
                break
            row["fracture_flag"] = fractured_now
            row["fracture_reason"] = reason
            append_curve_row(files["curve_csv"], row)
            current_chunk = next_chunk
            if fractured_now:
                if STOP_AFTER_FRACTURE:
                    post_target = min(current_chunk + profile['post_fracture_extra_chunks'], profile['max_total_chunks'])
                    while current_chunk < post_target:
                        next_chunk = current_chunk + 1
                        raw, dump_file = run_one_chunk(case_name, outdir, files, bounds, next_chunk, profile)
                        row, sigma_peak = process_raw_curve_row(raw, sigma0, sigma_peak)
                        row.update(analyze_connectivity_from_dump(dump_file, top_ids, bottom_ids, sigma, profile) if USE_CONNECTIVITY_CRITERION else {})
                        row["fracture_flag"] = True
                        row["fracture_reason"] = "post_fracture"
                        append_curve_row(files["curve_csv"], row)
                        current_chunk = next_chunk
                break
        if non_gb_rejected:
            break
        rows_current = read_curve_csv(files["curve_csv"])
        state = get_last_curve_state(rows_current)

    # Write final data from latest restart
    if WRITE_FINAL_DATA:
        latest_chunk, latest_fp = latest_restart(outdir)
        if latest_fp:
            final_in = outdir / f"{case_name}_write_final.in"
            final_log = outdir / f"{case_name}_write_final.log"
            final_in.write_text(f"""units metal
dimension 3
boundary p f p
atom_style atomic
read_restart {latest_fp.as_posix()}
{lmp_common_pair()}
write_data {files['final_data'].as_posix()}
""", encoding="utf-8")
            run_lammps(final_in, final_log, outdir)

    rows_final = read_curve_csv(files["curve_csv"])
    state = get_last_curve_state(rows_final)
    if non_gb_rejected:
        status = "NON_GB_FRACTURE_REJECTED"
    elif state["fracture_detected"]:
        status = "EXTENDED_FRACTURED" if current_chunk > profile['n_chunks_initial'] else "COMPLETED_FRACTURED"
    else:
        status = "EXTENDED_NOT_FRACTURED_WITHIN_LIMIT" if current_chunk >= profile['max_total_chunks'] else "COMPLETED_NOT_FRACTURED"
    return finalize_case_summary(case_name, sigma, cov, nh, status, outdir, files, rows_final, source, profile)


def finalize_case_summary(case_name, sigma, cov, nh, status, outdir, files, rows, source, profile):
    valid = [r for r in rows if r.get("sigma_zeroed_gpa") is not None and r.get("strain_nominal") is not None]
    if valid:
        peak_row = max(valid, key=lambda r: r.get("sigma_zeroed_gpa", -1e99))
        peak_stress = peak_row.get("sigma_zeroed_gpa")
        peak_strain = peak_row.get("strain_nominal")
        peak_time = peak_row.get("time_ps")
    else:
        peak_stress = peak_strain = peak_time = None
    frac_rows = [r for r in rows if r.get("fracture_flag")]
    reject_rows = [r for r in rows if r.get("fracture_reason") == "non_gb_fracture_rejected"]
    if frac_rows:
        fr = frac_rows[0]
        fracture_detected = True
        fracture_chunk = fr.get("chunk")
        fracture_time = fr.get("time_ps")
        fracture_strain = fr.get("strain_nominal")
        fracture_stress = fr.get("sigma_zeroed_gpa")
        fracture_reason = fr.get("fracture_reason")
    elif reject_rows:
        fr = reject_rows[0]
        fracture_detected = False
        fracture_chunk = fr.get("chunk")
        fracture_time = fr.get("time_ps")
        fracture_strain = fr.get("strain_nominal")
        fracture_stress = fr.get("sigma_zeroed_gpa")
        fracture_reason = fr.get("fracture_reason")
    else:
        fr = {}
        fracture_detected = False
        fracture_chunk = fracture_time = fracture_strain = fracture_stress = fracture_reason = None
    actual_total_chunks = max([r.get("chunk") for r in rows if r.get("chunk") is not None] or [0])
    latest_chunk, latest_fp = latest_restart(outdir)
    summary = {
        "model": case_name, "sigma": sigma, "coverage": cov, "N_H": nh,
        "profile_note": profile.get("profile_note", ""),
        "grip_thickness_A": profile.get("grip_thickness_A"),
        "v_pull_A_per_ps": profile.get("v_pull_A_per_ps"),
        "pre_eq_time_ps": profile.get("pre_eq_time_ps"),
        "n_chunks_initial": profile.get("n_chunks_initial"),
        "max_total_chunks": profile.get("max_total_chunks"),
        "require_gb_fracture": profile.get("require_gb_fracture", False),
        "status": status, "source": str(source), "actual_total_chunks": actual_total_chunks,
        "peak_stress_GPa": peak_stress, "peak_strain": peak_strain, "peak_time_ps": peak_time,
        "fracture_detected": fracture_detected, "fracture_chunk": fracture_chunk,
        "fracture_time_ps": fracture_time, "fracture_strain": fracture_strain,
        "fracture_stress_GPa": fracture_stress, "fracture_reason": fracture_reason,
        "crack_y_center_A": fr.get("crack_y_center_A"),
        "crack_gap_A": fr.get("crack_gap_A"),
        "fracture_in_gb_core": fr.get("fracture_in_gb_core"),
        "fracture_location_note": fr.get("fracture_location_note"),
        "latest_restart": str(latest_fp) if latest_fp else "",
        "curve_csv": str(files["curve_csv"]), "trajectory_file": str(files["traj_file"]),
        "final_data": str(files["final_data"]),
    }
    files["summary_json"].write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"[DONE] {case_name} -> {status}, fracture={fracture_detected}, chunks={actual_total_chunks}")
    return summary


def write_summary_csv_with_fallback(summary_csv: Path, fields, results):
    """Write the summary CSV; if the target is locked, write a timestamped fallback."""
    def _write(path):
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in results:
                writer.writerow({k: r.get(k, "") for k in fields})

    try:
        _write(summary_csv)
        return summary_csv
    except PermissionError:
        fallback = summary_csv.with_name(f"{summary_csv.stem}_{now_stamp()}{summary_csv.suffix}")
        _write(fallback)
        log(f"[WARN] Could not write summary because target is locked or not writable: {summary_csv}")
        log(f"[WARN] Wrote timestamped summary instead: {fallback}")
        return fallback


# ============================================================
# 12. Main
# ============================================================
def main():
    ensure_dirs()
    check_paths()
    cases_all = build_cases()
    cases = select_cases(cases_all)
    check_cpu_plan(len(cases))

    if SHOW_STARTUP_CONFIG or not TERMINAL_BRIEF:
        log("=" * 88)
        log("Stage III-3 refined 005/010/015/020 dynamic tensile fracture test: all-GB slow grip")
        log(f"WORKDIR              = {WORKDIR}")
        log(f"MULTIH_MODEL_DIR      = {MULTIH_MODEL_DIR}")
        log(f"LAMMPS_EXE            = {LAMMPS_EXE}")
        log(f"POTENTIAL_FILE        = {POTENTIAL_FILE}")
        log(f"RUN_MODE             = {RUN_MODE}")
        log(f"SIGMAS_TO_RUN         = {SIGMAS_TO_RUN}")
        log(f"COVERAGES_TO_RUN      = {COVERAGES_TO_RUN}")
        log(f"CASES_TO_RUN          = {list(cases.keys())}")
        log(f"TOTAL_CPU_CORES       = {TOTAL_CPU_CORES}")
        log(f"RESERVED_CPU_CORES    = {RESERVED_CPU_CORES}")
        log(f"USE_MPI               = {USE_MPI}")
        log(f"MPIEXEC_RESOLVED      = {resolve_mpiexec() if USE_MPI else ''}")
        log(f"MPI_EXTRA_ARGS        = {MPI_EXTRA_ARGS}")
        log(f"MPI_RANKS_PER_JOB     = {MPI_RANKS_PER_JOB if USE_MPI else 1}")
        log(f"MAX_WORKERS           = {MAX_WORKERS}")
        log(f"OMP_THREADS           = {OMP_THREADS}")
        log(f"USE_OMP_PACKAGE       = {USE_OMP_PACKAGE}")
        log(f"PARALLEL_PLAN         = {parallel_plan_text(len(cases))}")
        log(f"TARGET_TEMP_K         = {TARGET_TEMP_K}")
        log(f"DEFAULT_PRE_EQ_TIME_PS = {DEFAULT_PROFILE['pre_eq_time_ps']}")
        log(f"DEFAULT_V_PULL_A_PER_PS = {DEFAULT_PROFILE['v_pull_A_per_ps']}")
        log(f"DEFAULT_N_CHUNKS_INITIAL = {DEFAULT_PROFILE['n_chunks_initial']}")
        log(f"EXTEND_UNTIL_FRACTURE = {EXTEND_UNTIL_FRACTURE}")
        log(f"DEFAULT_MAX_TOTAL_CHUNKS = {DEFAULT_PROFILE['max_total_chunks']}")
        log(f"CONNECT_CUTOFF_A      = {CONNECT_CUTOFF_A}")
        log(f"REQUIRE_GB_FRACTURE_SIGMAS = {sorted(REQUIRE_GB_FRACTURE_SIGMAS)}")
        log("=" * 88)

    results = []
    if MAX_WORKERS == 1:
        for case_name, info in cases.items():
            try:
                results.append(run_case(case_name, info))
            except Exception as e:
                log(f"[FAILED] {case_name}: {e}")
                results.append({"model": case_name, "sigma": info.get("sigma"), "coverage": info.get("coverage"), "status": f"FAILED: {type(e).__name__}: {e}", "source": str(info.get("source"))})
    else:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            fut_map = {ex.submit(run_case, name, info): (name, info) for name, info in cases.items()}
            for fut in as_completed(fut_map):
                name, info = fut_map[fut]
                try:
                    results.append(fut.result())
                except Exception as e:
                    log(f"[FAILED] {name}: {e}")
                    results.append({"model": name, "sigma": info.get("sigma"), "coverage": info.get("coverage"), "status": f"FAILED: {type(e).__name__}: {e}", "source": str(info.get("source"))})

    summary_csv = WORKDIR / "summaries" / "phase3_refined_0_25_dynamic_summary_allGB_slow_grip.csv"
    fields = ["model", "sigma", "coverage", "N_H", "profile_note", "grip_thickness_A", "v_pull_A_per_ps", "pre_eq_time_ps", "n_chunks_initial", "max_total_chunks", "require_gb_fracture", "status", "source", "actual_total_chunks", "peak_stress_GPa", "peak_strain", "peak_time_ps", "fracture_detected", "fracture_chunk", "fracture_time_ps", "fracture_strain", "fracture_stress_GPa", "fracture_reason", "crack_y_center_A", "crack_gap_A", "fracture_in_gb_core", "fracture_location_note", "latest_restart", "curve_csv", "trajectory_file", "final_data"]
    written_summary_csv = write_summary_csv_with_fallback(summary_csv, fields, results)
    log(f"[ALL_DONE] Summary written to: {written_summary_csv}")


if __name__ == "__main__":
    main()
