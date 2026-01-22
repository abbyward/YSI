import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

print("\033[0;35mA CSV file with columns containing a plate name, IDs for each well, chemistry, and concentration are required")
print("Remove dashes and underscores from all inputs except the file path")
print("The column headers, well IDs, duration, starting volume, and average initial and final counts will be needed.\033[0m")


# -----------------------------
# Helpers
# -----------------------------
def norm_str(x: str) -> str:
    return str(x).strip().lower().replace("_", "").replace("-", "").replace(" ", "")

def norm_header(x: str) -> str:
    return str(x).strip().lower().replace(" ", "_")

def base_name(name: str) -> str:
    # split on "_rep#" or " rep#" patterns (case-insensitive)
    m = re.split(r"[_\s]*rep\s*\d+$", name.strip(), flags=re.IGNORECASE)
    return m[0] if m else name

def get_well_ids(prompt, plate_map, df, plate_col, well_col):
    wells = []
    print(f"Enter {prompt} as 'plate_number, well_id' (e.g., 1, p96d05). Press Enter to finish.")
    while True:
        entry = input("> ").strip()
        if entry == "":
            break
        try:
            plate_num, well = [x.strip() for x in entry.split(",")]
            if plate_num not in plate_map:
                print(f"⚠️ Plate number {plate_num} not defined! Try again.")
                continue
            plate = plate_map[plate_num]
            well_norm = norm_str(well)

            df_subset = df[df[plate_col] == plate]
            valid_wells = df_subset[well_col].unique()
            if well_norm not in valid_wells:
                print(f"⚠️ Well {well_norm} not found in plate {plate}! Try again.")
                continue

            wells.append((plate, well_norm))
        except Exception:
            print("⚠️ Format must be: plate_number, well_id (example: 1, p96d05)")
    return wells

def average_concentration(df, wells, plate_column, well_column, conc_column, calibration_factor=None):
    """Mean concentration for a set of (plate, well). Optionally multiply by calibration_factor."""
    if not wells:
        return None

    subset = pd.DataFrame({
        "plate": df[plate_column].astype(str).map(norm_str),
        "well":  df[well_column].astype(str).map(norm_str),
        "conc":  pd.to_numeric(df[conc_column], errors="coerce")
    })

    wells_set = set(wells)
    mask = subset.apply(lambda row: (row["plate"], row["well"]) in wells_set, axis=1)
    matched = subset[mask]

    if matched.empty:
        return None

    mean_conc = matched["conc"].mean()
    return mean_conc * calibration_factor if calibration_factor else mean_conc

def group_mean_sem(values_by_name: dict):
    """
    values_by_name: {raw_condition_name: value}
    Groups by base_name (rep stripping), returns group_names, means, sems.
    """
    raw_names = list(values_by_name.keys())
    raw_vals  = np.array(list(values_by_name.values()), dtype=float)

    grouped = {}
    for n, v in zip(raw_names, raw_vals):
        b = base_name(n)
        grouped.setdefault(b, []).append(v)

    names = list(grouped.keys())
    vals_by_group = [np.array(grouped[n], dtype=float) for n in names]
    means = np.array([v.mean() for v in vals_by_group], dtype=float)
    sems  = np.array([v.std(ddof=1)/np.sqrt(len(v)) if len(v) > 1 else 0.0 for v in vals_by_group], dtype=float)
    return names, means, sems

def plot_one_chemistry(chem: str, values_for_plot: dict, results_df: pd.DataFrame):
    """
    values_for_plot[chem] = {line_name: metric}
    results_df contains paired columns; we filter table columns to show just this chem + shared metadata.
    """
    # aggregate reps
    names, means, sems = group_mean_sem(values_for_plot.get(chem, {}))
    finite_mask = np.isfinite(means)
    if not np.any(finite_mask):
        print(f"⚠️ No finite values to plot for {chem}. Skipping figure.")
        return

    names = [n for n, ok in zip(names, finite_mask) if ok]
    means = means[finite_mask]
    sems  = sems[finite_mask]

    # style
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 14,
        "axes.linewidth": 2.8,
        "xtick.major.width": 2.8,
        "ytick.major.width": 2.8,
        "xtick.major.size": 8,
        "ytick.major.size": 8,
    })

    x = np.arange(len(names))

    fig, ax = plt.subplots(2, 1, figsize=(10, 9), gridspec_kw={"height_ratios": [2.2, 1.3]})

    ax[0].bar(
        x,
        means,
        yerr=sems if np.any(sems > 0) else None,
        capsize=7,
        width=0.65,
        facecolor="white",
        edgecolor="black",
        linewidth=3.2,
        error_kw=dict(lw=2.2)
    )

    ax[0].set_ylabel("Δ / AUC (fmol/cell/hr)", fontsize=18, fontweight="bold")
    ax[0].set_title(f"{chem.upper()}", fontsize=22, fontweight="bold", pad=12)

    ax[0].set_xticks(x)
    ax[0].set_xticklabels(names, rotation=45, ha="right", fontsize=14)

    # y-lims include negatives and 0
    y_min = float(np.nanmin(means - sems))
    y_max = float(np.nanmax(means + sems))
    pad = 0.15 * (y_max - y_min) if (y_max - y_min) > 0 else 1.0
    lower = min(0.0, y_min) - pad
    upper = max(0.0, y_max) + pad
    ax[0].set_ylim(lower, upper)
    ax[0].axhline(0, linewidth=2)

    ax[0].spines["top"].set_visible(False)
    ax[0].spines["right"].set_visible(False)

    # Table panel: show shared + chem-specific columns
    keep_cols = [
        "condition", "initial_count", "baseline_vol_L", "final_vol_L",
        f"{chem}_initial_conc", f"{chem}_final_conc",
        f"{chem}_initial_amount_mol", f"{chem}_final_amount_mol",
        f"{chem}_metric_fmol_per_cell_hr"
    ]
    keep_cols = [c for c in keep_cols if c in results_df.columns]
    table_data = results_df[keep_cols].round(6)

    ax[1].axis("off")
    tbl = ax[1].table(
        cellText=table_data.values,
        colLabels=table_data.columns,
        loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1, 1.25)

    plt.tight_layout()
    plt.show()


# -----------------------------
# Read CSV + normalize
# -----------------------------
file_path = input("Enter path to your CSV file: ").strip()
df_raw = pd.read_csv(file_path)
df_raw.columns = df_raw.columns.map(norm_header)

chem_col  = norm_header(input("Enter the header of your chemistry column: "))
plate_col = norm_header(input("Enter the header of your Plate Name column: "))
well_col  = norm_header(input("Enter the header of your Well IDs column: "))
conc_col  = norm_header(input("Enter the header of your Concentration column: "))

required_cols = [chem_col, plate_col, well_col, conc_col]
missing = [c for c in required_cols if c not in df_raw.columns]
if missing:
    raise ValueError(f"⚠️ Missing required column(s) in CSV: {', '.join(missing)}")

df_raw[chem_col]  = df_raw[chem_col].astype(str).map(norm_str)
df_raw[plate_col] = df_raw[plate_col].astype(str).map(norm_str)
df_raw[well_col]  = df_raw[well_col].astype(str).map(norm_str)

available_chems = sorted(df_raw[chem_col].unique())

default_pair = "glucose,lactate"
pair_in = input(f"Enter the chemistries to be analyzed (default {default_pair}): ").strip()
pair_in = pair_in if pair_in else default_pair
chem_a, chem_b = [norm_str(x) for x in pair_in.split(",")]

if chem_a not in available_chems or chem_b not in available_chems:
    raise ValueError(f"⚠️ One or both paired chemistries not found. You entered: {chem_a}, {chem_b}")

df_a = df_raw[df_raw[chem_col] == chem_a].reset_index(drop=True)
df_b = df_raw[df_raw[chem_col] == chem_b].reset_index(drop=True)

if df_a.empty or df_b.empty:
    raise ValueError("⚠️ One of the paired chemistry subsets is empty after filtering.")


# -----------------------------
# Plate map ONCE (shared)
# -----------------------------
print("\n\033[0;35mPlate map\033[0m")
num_plates = int(input("How many plates are there in this experiment? ").strip())

plate_map = {}
for i in range(1, num_plates + 1):
    pname = norm_str(input(f"Enter the name of plate {i}: "))
    if pname not in df_a[plate_col].unique():
        raise ValueError(f"⚠️ Plate name not found for {chem_a}: {pname}")
    if pname not in df_b[plate_col].unique():
        raise ValueError(f"⚠️ Plate name not found for {chem_b}: {pname}")
    plate_map[str(i)] = pname

start_volume_L = float(input("Enter the initial sample volume in L: ").strip())


# -----------------------------
# Per-plate calibration + evaporation volumes (shared)
# -----------------------------
print("\n\033[0;35mCalibration + evaporation wells \033[0m")

plate_factors = {}
for plate_num, plate_name in plate_map.items():
    print(f"\n\033[0;32m--- Plate {plate_num} ({plate_name}) ---\033[0m")

    calibration_wells = get_well_ids(
        f"Plate {plate_num} calibration wells; Enter blank to type factors manually)",
        plate_map, df_a, plate_col, well_col
    )

    if len(calibration_wells) == 0:
        print("\033[0;33mNo calibration wells entered.\033[0m")
        calib_a = float(input(f"Enter pre-calculated calibration factor for {chem_a} (plate {plate_num}): ").strip())
        calib_b = float(input(f"Enter pre-calculated calibration factor for {chem_b} (plate {plate_num}): ").strip())
    else:
        mean_calib_a_raw = average_concentration(df_a, calibration_wells, plate_col, well_col, conc_col)
        mean_calib_b_raw = average_concentration(df_b, calibration_wells, plate_col, well_col, conc_col)
        if mean_calib_a_raw is None or mean_calib_b_raw is None:
            raise ValueError(f"⚠️ Calibration wells did not match rows for plate {plate_num} in one/both chemistries.")

        # Required rule:
        # glucose -> 5.0/mean ; lactate -> 10.0/mean
        def default_target(chem):
            if chem == "glucose":
                return 5.0
            if chem == "lactate":
                return 10.0
            return None

        target_a = default_target(chem_a)
        target_b = default_target(chem_b)

        if target_a is None:
            t = float(input(f"Enter calibration target for {chem_a} (plate {plate_num}) (e.g. 5 or 10): ").strip())
            calib_a = t / mean_calib_a_raw
        else:
            calib_a = target_a / mean_calib_a_raw

        if target_b is None:
            t = float(input(f"Enter calibration target for {chem_b} (plate {plate_num}) (e.g. 5 or 10): ").strip())
            calib_b = t / mean_calib_b_raw
        else:
            calib_b = target_b / mean_calib_b_raw

    # Evaporation wells (entered once)
    pre_eva_wells      = get_well_ids(f"Plate {plate_num} pre-evap wells (C1)",      plate_map, df_a, plate_col, well_col)
    baseline_eva_wells = get_well_ids(f"Plate {plate_num} baseline-evap wells (C2)", plate_map, df_a, plate_col, well_col)
    final_eva_wells    = get_well_ids(f"Plate {plate_num} final-evap wells (C3)",    plate_map, df_a, plate_col, well_col)

    # volume correction uses ratios; calibration cancels -> use raw from chem_a
    avg_pre_raw   = average_concentration(df_a, pre_eva_wells,      plate_col, well_col, conc_col, calibration_factor=None)
    avg_base_raw  = average_concentration(df_a, baseline_eva_wells, plate_col, well_col, conc_col, calibration_factor=None)
    avg_final_raw = average_concentration(df_a, final_eva_wells,    plate_col, well_col, conc_col, calibration_factor=None)

    if None in (avg_pre_raw, avg_base_raw, avg_final_raw):
        raise ValueError(f"⚠️ Missing evaporation data for plate {plate_num} (using {chem_a} rows).")

    baseline_vol_L = (start_volume_L * avg_pre_raw) / avg_base_raw
    final_vol_L    = (baseline_vol_L * avg_base_raw) / avg_final_raw

    plate_factors[plate_name] = {
        "baseline_vol_L": baseline_vol_L,
        "final_vol_L": final_vol_L,
        "calibration": {chem_a: calib_a, chem_b: calib_b}
    }


# -----------------------------
# Condition loop (entered ONCE, compute BOTH)
# -----------------------------
print("\n\033[0;35mNow enter conditions\033[0m")

rows_out = []
values_for_plot = {chem_a: {}, chem_b: {}}

while True:
    line_name = input("\nEnter a cell line / condition name (or press Enter to finish): ").strip()
    if line_name == "":
        break

    try:
        init_wells  = get_well_ids(f"{line_name} wells for hour 0 concentration", plate_map, df_a, plate_col, well_col)
        final_wells = get_well_ids(f"{line_name} wells for final concentration",  plate_map, df_a, plate_col, well_col)

        duration_hrs  = float(input(f"Enter duration (hrs) for {line_name}: ").strip())
        initial_count = float(input(f"Enter the initial cell count for {line_name}: ").strip())

        condition_plates = {plate for (plate, well) in init_wells + final_wells}
        if len(condition_plates) != 1:
            print(f"⚠️ {line_name} spans multiple plates ({condition_plates}), not supported. Skipping.")
            continue
        plate_name = next(iter(condition_plates))

        baseline_vol_L = plate_factors[plate_name]["baseline_vol_L"]
        final_vol_L    = plate_factors[plate_name]["final_vol_L"]

        print(f"\n\033[0;35mDoublings per day questions\033[0m")
        doublings = float(input(f"Enter doublings per day for {line_name} (enter 0 if unknown): ").strip())
        if doublings == 0:
            final_count = float(input(f"Enter the final cell count for {line_name}: ").strip())
            doublings = (np.log(final_count / initial_count) / np.log(2)) / (duration_hrs / 24.0)

        if doublings == 0:
            print(f"⚠️ Doublings per day is zero for {line_name}, can't compute AUC. Skipping.")
            continue

        doublings_per_day = doublings / 24

        # keep your original formula
        auc = (initial_count / (doublings_per_day * np.log(2))) * (2 ** (doublings_per_day * 24) - 1)

        out_row = {
            "condition": line_name,
            "initial_count": initial_count,
            "doublings_per_day": doublings_per_day,
            "baseline_vol_L": baseline_vol_L,
            "final_vol_L": final_vol_L,
        }

        for chem, df_chem in [(chem_a, df_a), (chem_b, df_b)]:
            calib = plate_factors[plate_name]["calibration"][chem]

            init_conc  = average_concentration(df_chem, init_wells,  plate_col, well_col, conc_col, calibration_factor=calib)
            final_conc = average_concentration(df_chem, final_wells, plate_col, well_col, conc_col, calibration_factor=calib)

            if init_conc is None or final_conc is None:
                print(f"⚠️ Missing concentration data for {line_name} in {chem}. Skipping this chem.")
                out_row[f"{chem}_metric_fmol_per_cell_hr"] = np.nan
                continue

            init_amt_mol  = init_conc  * baseline_vol_L
            final_amt_mol = final_conc * final_vol_L

            delta_mol = final_amt_mol - init_amt_mol
            delta_over_auc = delta_mol / auc if auc != 0 else np.nan
            metric = delta_over_auc * 1e12  # mol -> fmol

            out_row[f"{chem} initial amount"] = init_amt_mol
            out_row[f"{chem} final amount"]   = final_amt_mol
            out_row[f"{chem} fmol per cell hr"] = metric

            values_for_plot[chem][line_name] = metric

        rows_out.append(out_row)

    except Exception as e:
        print(f"⚠️ Error processing {line_name}: {e}. Skipping.")


# -----------------------------
# Output tables + TWO figures
# -----------------------------
if not rows_out:
    print("\n⚠️ No conditions processed.")
    raise SystemExit

results_df = pd.DataFrame(rows_out)

print("\n\033[0;35m===== Paired Results Table (both chemistries) =====\033[0m")
print(results_df)

# Figure 1: chem_a (glucose)
plot_one_chemistry(chem_a, values_for_plot, results_df)

# Figure 2: chem_b (lactate)
plot_one_chemistry(chem_b, values_for_plot, results_df)

print("\n\033[0;35mDone.\033[0m")
