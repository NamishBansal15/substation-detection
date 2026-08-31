import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm # Alias font_manager as fm for brevity

# ════════════════════════════════════════════════════════════════
# GLOBAL ACADEMIC STYLE CONFIGURATION
# ════════════════════════════════════════════════════════════════

TITLE_SIZE = 22
LABEL_SIZE = 20
TICK_SIZE = 20
LEGEND_SIZE = 14

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": TICK_SIZE,
    "axes.titlesize": TITLE_SIZE,
    "axes.labelsize": LABEL_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": LEGEND_SIZE,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.titlesize": TITLE_SIZE + 3
})

# ════════════════════════════════════════════════════════════════
# GLOBAL ACADEMIC STYLE CONFIGURATION
# ════════════════════════════════════════════════════════════════
TITLE_SIZE = 22 # Increased from 15
LABEL_SIZE = 20 # Increased from 13
TICK_SIZE = 20 # Increased from 11
LEGEND_SIZE = 14 # Increased from 12
FONT_FAMILY = "Times New Roman"

# 3. Verify if 'Times New Roman' is now available in Matplotlib's font list
available_font_names = {f.name for f in fm.fontManager.ttflist}
if FONT_FAMILY in available_font_names:
    print(f" '{FONT_FAMILY}' font found in available fonts.")
    # Set 'Times New Roman' as the primary font
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = [FONT_FAMILY] + plt.rcParams['font.serif']
else:
    print(f" WARNING: '{FONT_FAMILY}' font NOT found in available fonts. Fallback to default Matplotlib font will occur.")
    print("Listing top 5 available serif fonts for debugging:")
    serif_fonts = sorted(list({f.name for f in fm.fontManager.ttflist if 'serif' in f.name.lower()}))
    print(serif_fonts[:5])

plt.rcParams.update({
    "font.size": TICK_SIZE,
    "axes.titlesize": TITLE_SIZE,
    "axes.labelsize": LABEL_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": LEGEND_SIZE,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.titlesize": TITLE_SIZE + 3
})

# ════════════════════════════════════════════════════════════════
# MASTER DATASET: EXCLUSIVELY BOOTSTRAP FAMILIES & BASELINES
# ════════════════════════════════════════════════════════════════
provided_data = [
    # RF-DETR Models (Updated rfdetr-base values)
    {"model": "rfdetr-large", "architecture": "RF-DETR", "final_mAP50": 0.8808, "final_mAP50_95": 0.6321, "ap50_alt_energy": 0.9512, "ap50_circuit_breaker": 0.7585, "ap50_control": 1.0000, "ap50_power_lines": 1.0000, "ap50_reactor": 0.8551, "ap50_transformer": 0.7203},
    {"model": "rfdetr-base", "architecture": "RF-DETR", "final_mAP50": 0.8257, "final_mAP50_95": 0.5245, "ap50_alt_energy": 0.8587, "ap50_circuit_breaker": 0.6911, "ap50_control": 0.9980, "ap50_power_lines": 1.0000, "ap50_reactor": 0.8453, "ap50_transformer": 0.5608},
    # Cascade R-CNN Baselines
    {"model": "cascade_r50", "architecture": "Cascade R-CNN", "final_mAP50": 0.3506, "final_mAP50_95": 0.2187, "ap50_alt_energy": 0.2571, "ap50_circuit_breaker": 0.2409, "ap50_control": 0.6242, "ap50_power_lines": 0.5927, "ap50_reactor": 0.1976, "ap50_transformer": 0.1912},
    {"model": "cascade_r101", "architecture": "Cascade R-CNN", "final_mAP50": 0.3352, "final_mAP50_95": 0.1742, "ap50_alt_energy": 0.1643, "ap50_circuit_breaker": 0.3018, "ap50_control": 0.5897, "ap50_power_lines": 0.5139, "ap50_reactor": 0.2381, "ap50_transformer": 0.2030},
    # YOLO Families
    {"model": "yolov8n", "architecture": "YOLOv8", "final_mAP50": 0.6024, "final_mAP50_95": 0.4016, "ap50_alt_energy": 0.6132, "ap50_circuit_breaker": 0.5841, "ap50_control": 0.7604, "ap50_power_lines": 0.7356, "ap50_reactor": 0.5061, "ap50_transformer": 0.4150},
    {"model": "yolov8s", "architecture": "YOLOv8", "final_mAP50": 0.6227, "final_mAP50_95": 0.4200, "ap50_alt_energy": 0.6400, "ap50_circuit_breaker": 0.6263, "ap50_control": 0.7392, "ap50_power_lines": 1.0000, "ap50_reactor": 0.5208, "ap50_transformer": 0.4732},
    {"model": "yolov8m", "architecture": "YOLOv8", "final_mAP50": 0.6198, "final_mAP50_95": 0.4184, "ap50_alt_energy": 0.6432, "ap50_circuit_breaker": 0.6189, "ap50_control": 0.7622, "ap50_power_lines": 1.0000, "ap50_reactor": 0.5017, "ap50_transformer": 0.4694},
    {"model": "yolov8l", "architecture": "YOLOv8", "final_mAP50": 0.4925, "final_mAP50_95": 0.3029, "ap50_alt_energy": 0.4922, "ap50_circuit_breaker": 0.4919, "ap50_control": 0.7346, "ap50_power_lines": 1.0000, "ap50_reactor": 0.3647, "ap50_transformer": 0.1994},
    {"model": "yolo11n", "architecture": "YOLOv11", "final_mAP50": 0.6054, "final_mAP50_95": 0.4007, "ap50_alt_energy": 0.6494, "ap50_circuit_breaker": 0.5736, "ap50_control": 0.7387, "ap50_power_lines": 1.0000, "ap50_reactor": 0.5385, "ap50_transformer": 0.4257},
    {"model": "yolo11s", "architecture": "YOLOv11", "final_mAP50": 0.6030, "final_mAP50_95": 0.3991, "ap50_alt_energy": 0.6469, "ap50_circuit_breaker": 0.6017, "ap50_control": 0.7603, "ap50_power_lines": 1.0000, "ap50_reactor": 0.5019, "ap50_transformer": 0.3865},
    {"model": "yolo11m", "architecture": "YOLOv11", "final_mAP50": 0.5441, "final_mAP50_95": 0.3414, "ap50_alt_energy": 0.5619, "ap50_circuit_breaker": 0.5209, "ap50_control": 0.7528, "ap50_power_lines": 1.0000, "ap50_reactor": 0.4375, "ap50_transformer": 0.2889},
    {"model": "yolo11l", "architecture": "YOLOv11", "final_mAP50": 0.4521, "final_mAP50_95": 0.2629, "ap50_alt_energy": 0.4628, "ap50_circuit_breaker": 0.4516, "ap50_control": 0.6382, "ap50_power_lines": 1.0000, "ap50_reactor": 0.3000, "ap50_transformer": 0.1867},
    {"model": "yolo12n", "architecture": "YOLOv12", "final_mAP50": 0.5956, "final_mAP50_95": 0.3919, "ap50_alt_energy": 0.6316, "ap50_circuit_breaker": 0.5723, "ap50_control": 0.7592, "ap50_power_lines": 1.0000, "ap50_reactor": 0.4938, "ap50_transformer": 0.4108},
    {"model": "yolo12s", "architecture": "YOLOv12", "final_mAP50": 0.5547, "final_mAP50_95": 0.3568, "ap50_alt_energy": 0.6025, "ap50_circuit_breaker": 0.5528, "ap50_control": 0.7251, "ap50_power_lines": 1.0000, "ap50_reactor": 0.4547, "ap50_transformer": 0.3108},
    {"model": "yolo12m", "architecture": "YOLOv12", "final_mAP50": 0.4856, "final_mAP50_95": 0.2950, "ap50_alt_energy": 0.5606, "ap50_circuit_breaker": 0.4575, "ap50_control": 0.7135, "ap50_power_lines": 1.0000, "ap50_reactor": 0.3344, "ap50_transformer": 0.1755},
    {"model": "yolo12l", "architecture": "YOLOv12", "final_mAP50": 0.4224, "final_mAP50_95": 0.2434, "ap50_alt_energy": 0.3803, "ap50_circuit_breaker": 0.4018, "ap50_control": 0.6913, "ap50_power_lines": 1.0000, "ap50_reactor": 0.2665, "ap50_transformer": 0.1375}
]

# ════════════════════════════════════════════════════════════════
# MASTER BOOTSTRAP 95% CI DICTIONARY (Updated rfdetr-base CIs)
# ════════════════════════════════════════════════════════════════
ci_data = {
    "rfdetr-base": {"mAP50": [0.7572, 0.8840], "mAP50_95": [0.4779, 0.5720], "Alt Energy": [0.6710, 0.9745], "Circuit Breaker": [0.4697, 0.8822], "Control": [0.9897, 1.0000], "Power Lines": [1.0000, 1.0000], "Reactor": [0.7200, 0.9365], "Transformer": [0.3138, 0.7628]},
    "rfdetr-large": {"mAP50": [0.8136, 0.9382], "mAP50_95": [0.5847, 0.6791], "Alt Energy": [0.8399, 1.0000], "Circuit Breaker": [0.5342, 0.9620], "Control": [1.0000, 1.0000], "Power Lines": [1.0000, 1.0000], "Reactor": [0.7243, 0.9515], "Transformer": [0.4928, 0.8917]},
    "cascade_r50": {"mAP50": [0.2865, 0.4183], "mAP50_95": [0.1752, 0.2663], "Alt Energy": [0.1165, 0.4258], "Circuit Breaker": [0.1340, 0.3631], "Control": [0.4183, 0.8723], "Power Lines": [0.4443, 0.7547], "Reactor": [0.0910, 0.3184], "Transformer": [0.0539, 0.3418]},
    "cascade_r101": {"mAP50": [0.2725, 0.3956], "mAP50_95": [0.1377, 0.2144], "Alt Energy": [0.0816, 0.2614], "Circuit Breaker": [0.1619, 0.4951], "Control": [0.3926, 0.8444], "Power Lines": [0.3707, 0.6758], "Reactor": [0.1226, 0.3654], "Transformer": [0.0971, 0.3077]},
    "yolov8n": {"mAP50": [0.5246, 0.6814], "mAP50_95": [0.3466, 0.4576], "Alt Energy": [0.4205, 0.8206], "Circuit Breaker": [0.3833, 0.8016], "Control": [0.5320, 0.9485], "Power Lines": [0.5769, 0.8816], "Reactor": [0.3504, 0.6548], "Transformer": [0.2523, 0.5991]},
    "yolov8s": {"mAP50": [0.5413, 0.7009], "mAP50_95": [0.3625, 0.4773], "Alt Energy": [0.4429, 0.8500], "Circuit Breaker": [0.4386, 0.8384], "Control": [0.5000, 0.9478], "Power Lines": [0.5845, 0.8810], "Reactor": [0.3445, 0.7024], "Transformer": [0.3139, 0.6632]},
    "yolov8m": {"mAP50": [0.5402, 0.6933], "mAP50_95": [0.3602, 0.4720], "Alt Energy": [0.4410, 0.8484], "Circuit Breaker": [0.4356, 0.8253], "Control": [0.5427, 0.9626], "Power Lines": [0.5827, 0.8805], "Reactor": [0.3389, 0.6651], "Transformer": [0.3179, 0.6361]},
    "yolov8l": {"mAP50": [0.4306, 0.5549], "mAP50_95": [0.2617, 0.3456], "Alt Energy": [0.3366, 0.6777], "Circuit Breaker": [0.3176, 0.6978], "Control": [0.5576, 0.9220], "Power Lines": [0.5271, 0.8210], "Reactor": [0.2352, 0.5046], "Transformer": [0.1015, 0.3161]},
    "yolo11n": {"mAP50": [0.5226, 0.6886], "mAP50_95": [0.3423, 0.4590], "Alt Energy": [0.4390, 0.8604], "Circuit Breaker": [0.3560, 0.8100], "Control": [0.5158, 0.9538], "Power Lines": [0.5548, 0.8540], "Reactor": [0.3813, 0.7023], "Transformer": [0.2829, 0.5833]},
    "yolo11s": {"mAP50": [0.5256, 0.6775], "mAP50_95": [0.3444, 0.4508], "Alt Energy": [0.4389, 0.8438], "Circuit Breaker": [0.4034, 0.8298], "Control": [0.5624, 0.9463], "Power Lines": [0.5740, 0.8810], "Reactor": [0.3487, 0.6595], "Transformer": [0.2329, 0.5528]},
    "yolo11m": {"mAP50": [0.4771, 0.6098], "mAP50_95": [0.2982, 0.3857], "Alt Energy": [0.3935, 0.7457], "Circuit Breaker": [0.3529, 0.7347], "Control": [0.5732, 0.9341], "Power Lines": [0.5533, 0.8632], "Reactor": [0.2857, 0.5915], "Transformer": [0.1649, 0.4184]},
    "yolo11l": {"mAP50": [0.3845, 0.5216], "mAP50_95": [0.2210, 0.3107], "Alt Energy": [0.2978, 0.6489], "Circuit Breaker": [0.2733, 0.6533], "Control": [0.4391, 0.8831], "Power Lines": [1.0000, 1.0000], "Reactor": [0.1829, 0.4197], "Transformer": [0.0943, 0.2789]},
    "yolo12n": {"mAP50": [0.5171, 0.6722], "mAP50_95": [0.3368, 0.4438], "Alt Energy": [0.4253, 0.8221], "Circuit Breaker": [0.3628, 0.8117], "Control": [0.5508, 0.9605], "Power Lines": [1.0000, 1.0000], "Reactor": [0.3388, 0.6584], "Transformer": [0.2644, 0.5727]},
    "yolo12s": {"mAP50": [0.4836, 0.6284], "mAP50_95": [0.3099, 0.4072], "Alt Energy": [0.4120, 0.7984], "Circuit Breaker": [0.3642, 0.7911], "Control": [0.5369, 0.9444], "Power Lines": [1.0000, 1.0000], "Reactor": [0.3043, 0.6076], "Transformer": [0.1807, 0.4548]},
    "yolo12m": {"mAP50": [0.4191, 0.5503], "mAP50_95": [0.2511, 0.3371], "Alt Energy": [0.3754, 0.7515], "Circuit Breaker": [0.2876, 0.6664], "Control": [0.5398, 0.8950], "Power Lines": [1.0000, 1.0000], "Reactor": [0.1900, 0.4865], "Transformer": [0.0803, 0.2789]},
    "yolo12l": {"mAP50": [0.3613, 0.4809], "mAP50_95": [0.2016, 0.2864], "Alt Energy": [0.2160, 0.5852], "Circuit Breaker": [0.2545, 0.5873], "Control": [0.5072, 0.9077], "Power Lines": [1.0000, 1.0000], "Reactor": [0.1516, 0.3731], "Transformer": [0.0683, 0.2028]}
}

CLASSES = ["Alt Energy", "Circuit Breaker", "Reactor", "Transformer"]
class_keys = [f"ap50_{c.lower().replace(' ','_')}" for c in CLASSES]

df = pd.DataFrame(provided_data)

# Scale decimal coordinates to clean percentage metrics
metric_cols = ["final_mAP50", "final_mAP50_95"] + class_keys
for col in metric_cols:
    df[col] = df[col] * 100.0

# Hierarchy sorting index updated
df["arch_order"] = df["architecture"].map({"RF-DETR": 0, "Cascade R-CNN": 1, "YOLOv8": 2, "YOLOv11": 3, "YOLOv12": 4})
df = df.sort_values(["arch_order", "final_mAP50"], ascending=[True, False]).reset_index(drop=True)

# Aesthetic palette update
ARCH_COLORS = {
    "RF-DETR":       "#FF7F0E",
    "Cascade R-CNN": "#7F7F7F",
    "YOLOv8":        "#2CA02C",
    "YOLOv11":       "#9467BD",
    "YOLOv12":       "#D62728"
}
model_colors = [ARCH_COLORS[arch] for arch in df["architecture"]]

# ════════════════════════════════════════════════════════════════
# CANVAS MASTER GRID SYSTEM (3x2 Matrix)
# ════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 20)) # Adjusted figsize to make figure bigger
gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.3) # Changed to 3 rows for 4 classes
y_pos = np.arange(len(df))

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])

# PANEL A: Macro Performance Evaluation (mAP@50)
bars_a = ax_a.barh(y_pos, df["final_mAP50"], color=model_colors, alpha=0.85, edgecolor='black', linewidth=0.6)
ax_a.set_yticks(y_pos)
ax_a.set_yticklabels(df["model"], fontsize=TICK_SIZE)
ax_a.invert_yaxis()
ax_a.set_xlabel("Score (%)", fontsize=LABEL_SIZE)
ax_a.set_title("(a) Overall Performance (mAP@50)", weight="bold", pad=12, x=-0.05, ha='left')
ax_a.set_xlim(0, 118)
ax_a.grid(axis='x', linestyle='--', alpha=0.3)

# PANEL B: Localization Precision Thresholds (mAP@50:95)
bars_b = ax_b.barh(y_pos, df["final_mAP50_95"], color=model_colors, alpha=0.85, edgecolor='black', linewidth=0.6)
ax_b.set_yticks(y_pos)
ax_b.set_yticklabels([])
ax_b.invert_yaxis()
ax_b.set_xlabel("Score (%)", fontsize=LABEL_SIZE)
ax_b.set_title("(b) Overall Performance (mAP@50:95)", weight="bold", pad=12)
ax_b.set_xlim(0, 118)
ax_b.grid(axis='x', linestyle='--', alpha=0.3)

# ════════════════════════════════════════════════════════════════
# ERROR BAR EXTRACTOR & TEXT ALIGNMENT LABELS (PANELS A & B)
# ════════════════════════════════════════════════════════════════
for idx, row in df.iterrows():
    m_name = row["model"]

    # --- Panel A Labels ---
    if m_name in ci_data:
        ci_a = np.array(ci_data[m_name]["mAP50"]) * 100.0
        ax_a.errorbar(
            x=[(ci_a[0] + ci_a[1]) / 2], y=[idx], xerr=[[(ci_a[1] - ci_a[0]) / 2], [(ci_a[1] - ci_a[0]) / 2]],
            fmt='none', ecolor='#222222', capsize=4, elinewidth=1.3, capthick=1.3, zorder=4
        )
        x_pos_text_a = ci_a[1]
    else:
        x_pos_text_a = row["final_mAP50"]

    ax_a.text(x_pos_text_a + 1.0, idx, f"{row['final_mAP50']:.1f}%", va='center', ha='left', fontsize=TICK_SIZE-1)

    # --- Panel B Labels ---
    if m_name in ci_data:
        ci_b = np.array(ci_data[m_name]["mAP50_95"]) * 100.0
        ax_b.errorbar(
            x=[(ci_b[0] + ci_b[1]) / 2], y=[idx], xerr=[[(ci_b[1] - ci_b[0]) / 2], [(ci_b[1] - ci_b[0]) / 2]],
            fmt='none', ecolor='#222222', capsize=4, elinewidth=1.3, capthick=1.3, zorder=4
        )
        x_pos_text_b = ci_b[1]
    else:
        x_pos_text_b = row["final_mAP50_95"]

    ax_b.text(x_pos_text_b + 1.0, idx, f"{row['final_mAP50_95']:.1f}%", va='center', ha='left', fontsize=TICK_SIZE-1)

# ════════════════════════════════════════════════════════════════
# CLASS EVALUATION BLOCKS (Panels c to f)
# ════════════════════════════════════════════════════════════════
grid_positions = [(1, 0), (1, 1), (2, 0), (2, 1)] # Updated for 4 classes
panel_letters = ["c", "d", "e", "f"]

for idx, class_name in enumerate(CLASSES):
    r_pos, c_pos = grid_positions[idx]
    ax_c = fig.add_subplot(gs[r_pos, c_pos])
    key = class_keys[idx]
    letter = panel_letters[idx]

    bars_c = ax_c.barh(y_pos, df[key], color=model_colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax_c.set_title(f"({letter}) {class_name} (mAP@50)", weight="bold", fontsize=TITLE_SIZE, pad=12)
    ax_c.set_xlim(0, 118)
    ax_c.invert_yaxis()
    ax_c.grid(axis='x', linestyle='--', alpha=0.25)
    ax_c.set_xlabel("Score (%)", fontsize=LABEL_SIZE)

    # Ensure y-axis labels are shown only for the first column of class plots in each row if applicable
    # For a 3x2 grid, y-labels should appear on the first column (c_pos == 0)
    if c_pos == 0:
        ax_c.set_yticks(y_pos)
        ax_c.set_yticklabels(df["model"], fontsize=TICK_SIZE)
    else:
        ax_c.set_yticks([])

    # Inject specific per-class confidence intervals and position labels right of CIs
    for row_idx, row in df.iterrows():
        m_name = row["model"]

        if m_name in ci_data and class_name in ci_data[m_name]:
            ci_val = np.array(ci_data[m_name][class_name]) * 100.0
            ax_c.errorbar(
                x=[(ci_val[0] + ci_val[1]) / 2], y=[row_idx],
                xerr=[[(ci_val[1] - ci_val[0]) / 2], [(ci_val[1] - ci_val[0]) / 2]],
                fmt='none', ecolor='#222222', capsize=3.5, elinewidth=1.1, capthick=1.1, zorder=4
            )
            x_pos_text_class = ci_val[1]
        else:
            x_pos_text_class = row[key]

        ax_c.text(x_pos_text_class + 1.0, row_idx, f"{row[key]:.1f}%", va='center', ha='left', fontsize=TICK_SIZE-1)

# ════════════════════════════════════════════════════════════════
# COMPACT LEGEND (moved inside ax_b)
# ════════════════════════════════════════════════════════════════
legend_handles = [plt.Rectangle((0,0), 1, 1, facecolor=color, edgecolor='black', alpha=0.85) for color in ARCH_COLORS.values()]

ax_b.legend(
    legend_handles,
    ARCH_COLORS.keys(),
    loc="lower right", # Changed location to lower right
    bbox_to_anchor=(1.0, 0), # Adjusted position to bottom-right outside, moved up from -0.25
    ncol=1, # Changed to vertical legend
    frameon=True, # Make legend frame visible
    facecolor='white', # Set background to white
    edgecolor='black', # Set edge color to black
    fontsize=LEGEND_SIZE + 2, # Increased legend fontsize
    title="Model Types"
)

# Removed plt.subplots_adjust(bottom=0.11) as legend is now internal.
plt.savefig("master_dashboard_all_models_ci.png", dpi=600, bbox_inches="tight")
plt.show()
print("\n✅ Canvas successfully updated with corrected rfdetr-base validation data blocks!")

# ════════════════════════════════════════════════════════════════
# Summary Tables
# ════════════════════════════════════════════════════════════════

print("\n--- Overall Model Performance Summary (mAP@50 and mAP@50:95) ---")
summary_df = df[['model', 'architecture', 'final_mAP50', 'final_mAP50_95']].sort_values(by='final_mAP50', ascending=False).reset_index(drop=True)
summary_df['final_mAP50'] = summary_df['final_mAP50'].map('{:.2f}%'.format)
summary_df['final_mAP50_95'] = summary_df['final_mAP50_95'].map('{:.2f}%'.format)
display(summary_df)

print("\n--- Average AP50 per Class Across All Models ---")
class_avg_data = {}
for class_name_display in CLASSES:
    column_name = f"ap50_{class_name_display.lower().replace(' ', '_')}"
    if column_name in df.columns:
        class_avg_data[class_name_display] = df[column_name].mean()
class_avg_df = pd.DataFrame(class_avg_data.items(), columns=['Class', 'Average AP50'])
class_avg_df['Average AP50'] = class_avg_df['Average AP50'].map('{:.2f}%'.format)
class_avg_df = class_avg_df.sort_values(by='Average AP50', ascending=False).reset_index(drop=True)
display(class_avg_df)