"""
run_equilibrium_tensile_all_cases.py

Stage III-1 tensile-response simulation for 0H/1H and refined/multi-H GB models on the workstation.

Design:
  - Each case reads its source model from:
      <WORKDIR>/phase1_y_tension_sources/<case>/<case>_source.lmp
  - Coverage levels:
      0H, 1H, 005cov, 010cov, 015cov, 020cov, 025cov, 050cov, 100cov
  - Loading protocol:
      quasi-static y-direction tensile deformation by changing box length,
      followed by energy minimization at every strain step.
  - Outputs per case:
      strain_stress.csv
      <case>_tensile_trajectory.lammpstrj
      final_tensile.lmp
      run_summary.json
  - Summary over all cases:
      D:\stage3_tensile_refined_multiH\summaries\stage3_1_quasistatic_0_100H_summary.csv

Important:
  Stage III-1 is not a fracture-to-failure protocol. The maximum stress in the CSV
  is the maximum corrected engineering stress within the prescribed strain window,
  not necessarily the fracture stress.
"""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import os
import re
import shutil
import subprocess
import time

import numpy as np


# ============================================================
# 0. Paths and cases
# ============================================================

WORKDIR = Path(r"C:\Users\Admin01\Desktop\stage3_tensile_refined_multiH")

LAMMPS_EXE = Path(
    r"C:\Users\Admin01\AppData\Local\LAMMPS 64-bit 10Dec2025-MSMPI with Python\bin\lmp.exe"
)

POTENTIAL_FILE = WORKDIR / "potential" / "BOP1.poly"

SIGMAS_TO_RUN = ["sigma3", "sigma5", "sigma11", "sigma17"]
# Exact coverage levels required.
COVERAGES_TO_RUN = ["0H", "1H", "005cov", "010cov", "015cov", "020cov", "025cov", "050cov", "100cov"]
# "all" or list, e.g. ["sigma5_GB_005cov", "sigma5_GB_010cov"]
CASES_TO_RUN = "all"


# ============================================================
# 1. Run controls
# ============================================================

RUN_MODE = "fresh"
# Options:
#   "fresh"     : run selected cases, backup old result folder if it already exists
#   "rerun"     : force rerun cases in RERUN_CASES, or all selected if RERUN_CASES=[]
#   "skip_done" : skip if strain_stress.csv and run_summary.json already exist
#   "input_only": only write LAMMPS input files, do not run LAMMPSfresh

RERUN_CASES = []

BACKUP_OLD_OUTPUTS = True
CLEAN_OLD_OUTPUTS = False

# ============================================================
# 1A. CPU / MPI / OpenMP parallel controls for 128-core workstation
# ============================================================
# This workstation LAMMPS executable is an MSMPI build, so launch each case
# with Microsoft MPI mpiexec.exe, matching the Stage III-3 workstation scripts.
#
# Recommended full-throughput setting for 36 quasi-static cases:
#   16 concurrent cases x 8 MPI ranks/case = 128 LAMMPS ranks total.
# If memory becomes tight during minimization, reduce MAX_WORKERS to 8 first.

TOTAL_CPU_CORES = 128
RESERVED_CPU_CORES = 8

USE_MPI = True
MPIEXEC_EXE = Path(r"C:\Program Files\Microsoft MPI\Bin\mpiexec.exe")
MPI_EXTRA_ARGS = []  # If local-host connection/firewall issues occur, try ["-localonly"].
REQUIRE_MATCHING_MPI = True
ALLOW_MPIEXEC_PATH_FALLBACK = False

MAX_WORKERS = 4
MPI_RANKS_PER_JOB = 64
OMP_THREADS = 1
USE_OMP_PACKAGE = False
WARN_ON_OVERSUBSCRIPTION = True


# ============================================================
# 2. Tensile protocol
# ============================================================

# Same conceptual protocol as the previous Stage III-1:
# finite y-strain window, quasi-static deformation, minimization at each frame.
N_STRAIN_STEPS = 266
MAX_ENGINEERING_STRAIN = 0.6

# If True, each frame uses exact target strain:
#   ylo/yhi are reset to y-center +/- 0.5*Ly0*(1+strain)
# This avoids cumulative strain error.
USE_EXACT_TARGET_STRAIN = True

# Optional initial minimization before recording frame 0.
DO_INITIAL_MINIMIZATION = True

# Per-frame minimization.
DO_MINIMIZE_EACH_STEP = True

MIN_STYLE_1 = "fire"
MIN_ETOL_1 = "1.0e-8"
MIN_FTOL_1 = "1.0e-8"
MIN_MAXITER_1 = 2000
MIN_MAXEVAL_1 = 8000

MIN_STYLE_2 = "cg"
MIN_ETOL_2 = "1.0e-10"
MIN_FTOL_2 = "1.0e-10"
MIN_MAXITER_2 = 4000
MIN_MAXEVAL_2 = 12000

THERMO_EVERY = 100


# ============================================================
# 3. Post-processing parameters
# ============================================================

# Fit initial slope over corrected stress-strain data below this strain.
ELASTIC_FIT_STRAIN_MAX = 0.02

# If too few points below the threshold, use the first N nonzero strain points.
ELASTIC_FIT_MIN_POINTS = 5
ELASTIC_FIT_FALLBACK_POINTS = 10


# ============================================================
# 4. Atom types and masses
# ============================================================

FE_TYPE = 1
C_TYPE = 2
H_TYPE = 3

MASS_FE = 55.845
MASS_C = 12.011
MASS_H = 1.008


# ============================================================
# 5. Utilities
# ============================================================

def log(msg: str):
    print(msg, flush=True)


def stamp():
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
    for sub in [
        "scripts",
        "summaries",
        "figs",
        "logs",
        "phase1_y_tension_sources",
        "phase1_y_tension_inputs",
        "phase1_y_tension_results",
    ]:
        (WORKDIR / sub).mkdir(parents=True, exist_ok=True)


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
    if RUN_MODE != "input_only":
        if not LAMMPS_EXE.exists():
            raise FileNotFoundError(f"LAMMPS executable not found: {LAMMPS_EXE}")
        validate_mpi_launcher()
        if not POTENTIAL_FILE.exists():
            raise FileNotFoundError(f"Potential file not found: {POTENTIAL_FILE}")


def case_name(sigma: str, cov: str):
    return f"{sigma}_GB_{cov}"


def build_cases():
    cases = {}
    for sigma in SIGMAS_TO_RUN:
        for cov in COVERAGES_TO_RUN:
            case = case_name(sigma, cov)
            source = WORKDIR / "phase1_y_tension_sources" / case / f"{case}_source.lmp"
            input_dir = WORKDIR / "phase1_y_tension_inputs" / case
            result_dir = WORKDIR / "phase1_y_tension_results" / case
            cases[case] = {
                "case": case,
                "sigma": sigma,
                "coverage": cov,
                "source": source,
                "input_dir": input_dir,
                "result_dir": result_dir,
            }
    return cases


def select_cases(cases):
    if CASES_TO_RUN == "all":
        return cases
    selected = {}
    for c in CASES_TO_RUN:
        if c not in cases:
            raise KeyError(f"Unknown case name: {c}")
        selected[c] = cases[c]
    return selected


def read_box_bounds_from_data(data_path: Path):
    xlo = xhi = ylo = yhi = zlo = zhi = None
    with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
        for _ in range(300):
            line = f.readline()
            if not line:
                break
            s = line.strip()
            if re.search(r"\bxlo\s+xhi\b", s):
                p = s.split()
                xlo, xhi = float(p[0]), float(p[1])
            elif re.search(r"\bylo\s+yhi\b", s):
                p = s.split()
                ylo, yhi = float(p[0]), float(p[1])
            elif re.search(r"\bzlo\s+zhi\b", s):
                p = s.split()
                zlo, zhi = float(p[0]), float(p[1])
    if None in (xlo, xhi, ylo, yhi, zlo, zhi):
        raise RuntimeError(f"Failed to parse box bounds from: {data_path}")

    return {
        "xlo": xlo, "xhi": xhi,
        "ylo": ylo, "yhi": yhi,
        "zlo": zlo, "zhi": zhi,
        "lx": xhi - xlo,
        "ly": yhi - ylo,
        "lz": zhi - zlo,
    }


def parse_lammps_data_atomic(data_path: Path):
    box = read_box_bounds_from_data(data_path)
    atoms = []
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
                    atoms.append([
                        int(parts[0]), int(parts[1]),
                        float(parts[2]), float(parts[3]), float(parts[4])
                    ])

    if not atoms:
        raise RuntimeError(f"No atoms parsed from: {data_path}")

    return box, np.array(atoms, dtype=float)


def count_atom_types(data_path: Path):
    _, atoms = parse_lammps_data_atomic(data_path)
    types = atoms[:, 1].astype(int)
    n_fe = int(np.sum(types == FE_TYPE))
    n_c = int(np.sum(types == C_TYPE))
    n_h = int(np.sum(types == H_TYPE))
    return n_fe, n_c, n_h


def global_appm(n_h, n_fe):
    if n_fe + n_h <= 0:
        return 0.0
    return n_h / (n_fe + n_h) * 1.0e6


def global_wppm(n_h, n_fe):
    if n_fe <= 0:
        return 0.0
    return (n_h * MASS_H) / (n_fe * MASS_FE) * 1.0e6


def prepare_case_dirs(result_dir: Path, input_dir: Path, force_fresh: bool):
    input_dir.mkdir(parents=True, exist_ok=True)

    if result_dir.exists() and force_fresh:
        if BACKUP_OLD_OUTPUTS:
            backup = result_dir.parent / f"{result_dir.name}_backup_{stamp()}"
            shutil.move(str(result_dir), str(backup))
            log(f"[BACKUP] {result_dir} -> {backup}")
        elif CLEAN_OLD_OUTPUTS:
            shutil.rmtree(result_dir)

    result_dir.mkdir(parents=True, exist_ok=True)


def build_lammps_command(input_file: Path, log_file: Path):
    env = os.environ.copy()

    # Keep thread-based libraries aligned with the requested OpenMP setting.
    # This prevents hidden oversubscription when running many cases.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[var] = str(max(1, OMP_THREADS))

    lammps_cmd = [
        str(LAMMPS_EXE),
        "-in", str(input_file),
        "-log", str(log_file),
    ]

    # For the MSMPI LAMMPS workstation build, pure MPI is usually safer than OpenMP.
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

    proc = subprocess.run(
        cmd,
        cwd=workdir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
    )

    stdout_text = decode_bytes(proc.stdout)
    stderr_text = decode_bytes(proc.stderr)

    stdout_file = workdir / f"{input_file.stem}_stdout.txt"
    cmd_text = subprocess.list2cmdline([str(x) for x in cmd])
    stdout_file.write_text(
        "===== COMMAND =====\n" + cmd_text +
        "\n\n===== STDOUT =====\n" + stdout_text +
        "\n\n===== STDERR =====\n" + stderr_text,
        encoding="utf-8",
        errors="ignore"
    )

    return proc.returncode, stdout_text, stderr_text


# ============================================================
# 6. LAMMPS input generator
# ============================================================

def pair_block():
    return f"""pair_style      polymorphic
pair_coeff      * * {POTENTIAL_FILE.as_posix()} Fe C H

neighbor        2.0 bin
neigh_modify    delay 0 every 1 check yes
"""


def per_step_relax_block():
    if not DO_MINIMIZE_EACH_STEP:
        return "run             0\n"

    return f"""min_style       {MIN_STYLE_1}
minimize        {MIN_ETOL_1} {MIN_FTOL_1} {MIN_MAXITER_1} {MIN_MAXEVAL_1}

min_style       {MIN_STYLE_2}
minimize        {MIN_ETOL_2} {MIN_FTOL_2} {MIN_MAXITER_2} {MIN_MAXEVAL_2}
"""


def initial_relax_block():
    if not DO_INITIAL_MINIMIZATION:
        return "run             0\n"

    return f"""min_style       {MIN_STYLE_1}
minimize        {MIN_ETOL_1} {MIN_FTOL_1} {MIN_MAXITER_1} {MIN_MAXEVAL_1}

min_style       {MIN_STYLE_2}
minimize        {MIN_ETOL_2} {MIN_FTOL_2} {MIN_MAXITER_2} {MIN_MAXEVAL_2}
"""


def make_lammps_input(case, source, result_dir, bounds):
    ylo0 = bounds["ylo"]
    yhi0 = bounds["yhi"]
    ly0 = bounds["ly"]
    yc0 = 0.5 * (ylo0 + yhi0)

    strain_inc = MAX_ENGINEERING_STRAIN / N_STRAIN_STEPS

    csv_file = result_dir / "strain_stress.csv"
    traj_file = result_dir / f"{case}_tensile_trajectory.lammpstrj"
    final_data = result_dir / "final_tensile.lmp"

    exact_change_box = f"""variable        strain_target equal v_i*{strain_inc:.12f}
variable        ylo_t equal {yc0:.12f}-0.5*{ly0:.12f}*(1.0+v_strain_target)
variable        yhi_t equal {yc0:.12f}+0.5*{ly0:.12f}*(1.0+v_strain_target)
change_box      all y final ${{ylo_t}} ${{yhi_t}} remap units box
"""

    incremental_change_box = f"""change_box      all y scale {1.0 + strain_inc:.12f} remap units box
"""

    change_box_text = exact_change_box if USE_EXACT_TARGET_STRAIN else incremental_change_box

    text = f"""units           metal
dimension       3
boundary        p p p
atom_style      atomic
newton          on

read_data       {source.as_posix()}

{pair_block()}
thermo          {THERMO_EVERY}
thermo_style    custom step temp pe pyy pxx pzz lx ly lz fnorm fmax atoms

reset_timestep  0

# Initial relaxation before frame 0
{initial_relax_block()}
run             0

# Frozen reference geometry from the original model
variable        ly0_const equal {ly0:.12f}

# Current variables
variable        strain_now equal (ly-v_ly0_const)/v_ly0_const
variable        stress_raw equal -pyy*1.0e-4
variable        pe_now equal pe
variable        ly_now equal ly
variable        pyy_now equal pyy

# Freeze initial stress after initial relaxation.
variable        sigma0 equal ${{stress_raw}}
variable        stress_corr equal v_stress_raw-v_sigma0

print           "step,strain,stress_raw_gpa,stress_corrected_gpa,pe_eV,ly_A,pyy_bar" file {csv_file.as_posix()} screen no
print           "0,${{strain_now}},${{stress_raw}},${{stress_corr}},${{pe_now}},${{ly_now}},${{pyy_now}}" append {csv_file.as_posix()} screen no

write_dump      all custom {traj_file.as_posix()} id type x y z ix iy iz xu yu zu modify sort id

variable        i loop {N_STRAIN_STEPS}
label           loop_tension

{change_box_text}

{per_step_relax_block()}
run             0

variable        strain_now equal (ly-v_ly0_const)/v_ly0_const
variable        stress_raw equal -pyy*1.0e-4
variable        stress_corr equal v_stress_raw-v_sigma0
variable        pe_now equal pe
variable        ly_now equal ly
variable        pyy_now equal pyy

print           "${{i}},${{strain_now}},${{stress_raw}},${{stress_corr}},${{pe_now}},${{ly_now}},${{pyy_now}}" append {csv_file.as_posix()} screen no
write_dump      all custom {traj_file.as_posix()} id type x y z ix iy iz xu yu zu modify append yes sort id

next            i
jump            SELF loop_tension

write_data      {final_data.as_posix()}
"""
    return text


# ============================================================
# 7. Result parsing and summary
# ============================================================

def read_strain_stress_csv(csv_file: Path):
    rows = []
    if not csv_file.exists():
        return rows

    with open(csv_file, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for r in reader:
            row = {
                "step": safe_int(r.get("step")),
                "strain": safe_float(r.get("strain")),
                "stress_raw_gpa": safe_float(r.get("stress_raw_gpa")),
                "stress_corrected_gpa": safe_float(r.get("stress_corrected_gpa")),
                "pe_eV": safe_float(r.get("pe_eV")),
                "ly_A": safe_float(r.get("ly_A")),
                "pyy_bar": safe_float(r.get("pyy_bar")),
            }
            rows.append(row)
    return rows


def fit_initial_slope(rows):
    data = [
        (r["strain"], r["stress_corrected_gpa"])
        for r in rows
        if r.get("strain") is not None
        and r.get("stress_corrected_gpa") is not None
        and r["strain"] > 0
    ]

    if not data:
        return None, 0

    fit_data = [(e, s) for e, s in data if e <= ELASTIC_FIT_STRAIN_MAX]

    if len(fit_data) < ELASTIC_FIT_MIN_POINTS:
        fit_data = data[:ELASTIC_FIT_FALLBACK_POINTS]

    if len(fit_data) < 2:
        return None, len(fit_data)

    eps = np.array([x[0] for x in fit_data], dtype=float)
    sig = np.array([x[1] for x in fit_data], dtype=float)

    # slope in GPa because stress is GPa and strain is dimensionless.
    slope, intercept = np.polyfit(eps, sig, 1)
    return float(slope), len(fit_data)


def summarize_curve(rows):
    if not rows:
        return {
            "n_frames": 0,
            "max_strain": None,
            "max_stress_GPa": None,
            "max_stress_strain": None,
            "final_strain": None,
            "final_stress_GPa": None,
            "initial_slope_GPa": None,
            "initial_slope_fit_points": 0,
        }

    valid = [
        r for r in rows
        if r.get("strain") is not None
        and r.get("stress_corrected_gpa") is not None
    ]

    if not valid:
        return {
            "n_frames": len(rows),
            "max_strain": None,
            "max_stress_GPa": None,
            "max_stress_strain": None,
            "final_strain": None,
            "final_stress_GPa": None,
            "initial_slope_GPa": None,
            "initial_slope_fit_points": 0,
        }

    max_row = max(valid, key=lambda r: r["stress_corrected_gpa"])
    final_row = valid[-1]
    slope, nfit = fit_initial_slope(valid)

    return {
        "n_frames": len(rows),
        "max_strain": max(r["strain"] for r in valid),
        "max_stress_GPa": max_row["stress_corrected_gpa"],
        "max_stress_strain": max_row["strain"],
        "final_strain": final_row["strain"],
        "final_stress_GPa": final_row["stress_corrected_gpa"],
        "initial_slope_GPa": slope,
        "initial_slope_fit_points": nfit,
    }


# ============================================================
# 8. Case runner
# ============================================================

def run_case(case, info):
    source = info["source"]
    input_dir = info["input_dir"]
    result_dir = info["result_dir"]
    sigma = info["sigma"]
    coverage = info["coverage"]

    if not source.exists():
        return {
            "case_name": case,
            "sigma": sigma,
            "coverage": coverage,
            "status": "FAILED_SOURCE_NOT_FOUND",
            "source": str(source),
        }

    csv_file = result_dir / "strain_stress.csv"
    summary_json = result_dir / "run_summary.json"

    if RUN_MODE == "skip_done" and csv_file.exists() and summary_json.exists():
        try:
            old = json.loads(summary_json.read_text(encoding="utf-8"))
            old["status"] = "SKIPPED_DONE"
            return old
        except Exception:
            return {
                "case_name": case,
                "sigma": sigma,
                "coverage": coverage,
                "status": "SKIPPED_DONE",
                "source": str(source),
            }

    force_fresh = RUN_MODE in ("fresh", "rerun", "input_only")
    if RUN_MODE == "rerun" and RERUN_CASES:
        force_fresh = case in RERUN_CASES

    prepare_case_dirs(result_dir, input_dir, force_fresh=force_fresh)

    n_fe, n_c, n_h = count_atom_types(source)
    appm = global_appm(n_h, n_fe)
    wppm = global_wppm(n_h, n_fe)

    bounds = read_box_bounds_from_data(source)

    input_file = input_dir / f"{case}_tensile.in"
    log_file = input_dir / f"{case}_tensile.log"

    lmp_text = make_lammps_input(case, source, result_dir, bounds)
    input_file.write_text(lmp_text, encoding="utf-8", errors="ignore")

    log(f"[CASE] {case}")
    log(f"       source = {source}")
    log(f"       N_H    = {n_h}, equiv = {wppm:.2f} wppm, {appm:.1f} appm")

    if RUN_MODE == "input_only":
        status = "INPUT_WRITTEN"
        rows = []
        curve_summary = summarize_curve(rows)
        retcode = None
    else:
        retcode, stdout, stderr = run_lammps(input_file, log_file, input_dir)
        if retcode == 0 and csv_file.exists():
            status = "OK"
        elif retcode == 0:
            status = "FAILED_NO_CSV"
        else:
            status = f"FAILED_RET_{retcode}"

        rows = read_strain_stress_csv(csv_file)
        curve_summary = summarize_curve(rows)

    result = {
        "case_name": case,
        "sigma": sigma,
        "coverage": coverage,
        "N_Fe": n_fe,
        "N_C": n_c,
        "N_H": n_h,
        "global_H_appm_equiv": appm,
        "global_H_wppm_equiv": wppm,
        "status": status,
        "return_code": retcode,
        "source": str(source),
        "input_file": str(input_file),
        "log_file": str(log_file),
        "curve_csv": str(csv_file),
        "trajectory_file": str(result_dir / f"{case}_tensile_trajectory.lammpstrj"),
        "final_data": str(result_dir / "final_tensile.lmp"),
        "N_STRAIN_STEPS": N_STRAIN_STEPS,
        "MAX_ENGINEERING_STRAIN": MAX_ENGINEERING_STRAIN,
        **curve_summary,
    }

    summary_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    log(
        f"[DONE] {case} -> {status}; "
        f"max_stress={result.get('max_stress_GPa')} GPa, "
        f"at strain={result.get('max_stress_strain')}"
    )

    return result


# ============================================================
# 9. Main
# ============================================================

def main():
    ensure_dirs()
    check_paths()

    cases_all = build_cases()
    cases = select_cases(cases_all)
    check_cpu_plan(len(cases))

    log("=" * 88)
    log("Stage III-1 quasi-static tensile-response simulation: 0H/1H + 005-100H coverage")
    log(f"WORKDIR                = {WORKDIR}")
    log(f"RUN_MODE               = {RUN_MODE}")
    log(f"SIGMAS_TO_RUN           = {SIGMAS_TO_RUN}")
    log(f"COVERAGES_TO_RUN        = {COVERAGES_TO_RUN}")
    log(f"CASES_TO_RUN            = {list(cases.keys())}")
    log(f"TOTAL_CPU_CORES         = {TOTAL_CPU_CORES}")
    log(f"RESERVED_CPU_CORES      = {RESERVED_CPU_CORES}")
    log(f"USE_MPI                 = {USE_MPI}")
    log(f"MPIEXEC_RESOLVED        = {resolve_mpiexec() if USE_MPI else ''}")
    log(f"MPI_EXTRA_ARGS          = {MPI_EXTRA_ARGS}")
    log(f"MPI_RANKS_PER_JOB       = {MPI_RANKS_PER_JOB if USE_MPI else 1}")
    log(f"MAX_WORKERS             = {MAX_WORKERS}")
    log(f"OMP_THREADS             = {OMP_THREADS}")
    log(f"USE_OMP_PACKAGE         = {USE_OMP_PACKAGE}")
    log(f"PARALLEL_PLAN           = {parallel_plan_text(len(cases))}")
    log(f"N_STRAIN_STEPS          = {N_STRAIN_STEPS}")
    log(f"MAX_ENGINEERING_STRAIN  = {MAX_ENGINEERING_STRAIN}")
    log(f"DO_INITIAL_MINIMIZATION = {DO_INITIAL_MINIMIZATION}")
    log(f"DO_MINIMIZE_EACH_STEP   = {DO_MINIMIZE_EACH_STEP}")
    log("=" * 88)

    results = []

    if MAX_WORKERS == 1:
        for case, info in cases.items():
            try:
                res = run_case(case, info)
            except Exception as e:
                log(f"[FAILED] {case}: {type(e).__name__}: {e}")
                res = {
                    "case_name": case,
                    "sigma": info.get("sigma"),
                    "coverage": info.get("coverage"),
                    "status": f"FAILED_EXCEPTION_{type(e).__name__}: {e}",
                    "source": str(info.get("source")),
                }
            results.append(res)
    else:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            future_map = {ex.submit(run_case, case, info): (case, info) for case, info in cases.items()}
            for fut in as_completed(future_map):
                case, info = future_map[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    log(f"[FAILED] {case}: {type(e).__name__}: {e}")
                    res = {
                        "case_name": case,
                        "sigma": info.get("sigma"),
                        "coverage": info.get("coverage"),
                        "status": f"FAILED_EXCEPTION_{type(e).__name__}: {e}",
                        "source": str(info.get("source")),
                    }
                results.append(res)

    summary_csv = WORKDIR / "summaries" / "stage3_1_quasistatic_0_100H_summary.csv"
    fieldnames = [
        "case_name", "sigma", "coverage",
        "N_Fe", "N_C", "N_H",
        "global_H_appm_equiv", "global_H_wppm_equiv",
        "status", "return_code",
        "n_frames", "max_strain",
        "max_stress_GPa", "max_stress_strain",
        "final_strain", "final_stress_GPa",
        "initial_slope_GPa", "initial_slope_fit_points",
        "N_STRAIN_STEPS", "MAX_ENGINEERING_STRAIN",
        "source", "input_file", "log_file", "curve_csv",
        "trajectory_file", "final_data",
    ]

    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    log("\nDone.")
    log(f"Summary written to: {summary_csv}")


if __name__ == "__main__":
    main()
