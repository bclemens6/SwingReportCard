import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches

# --- CONFIGURATION ---
st.set_page_config(page_title="FG Swing Tool", layout="wide", initial_sidebar_state="expanded")

# Force Light Mode & FanGraphs Header
st.markdown("""
    <style>
        [data-testid="stAppViewContainer"] {background-color: #ffffff;}
        [data-testid="stHeader"] {background-color: #ffffff;}
        .stAppHeader {background-color: #50ae26 !important;}
    </style>
""", unsafe_allow_html=True)

FG_GREEN = '#50ae26'
FG_BLUE = '#1f77b4'
FG_DARK = '#333333'
FG_GRAY = '#A9A9A9'
FG_LIGHT_GRAY = '#D3D3D3'

# --- DATA LOADING ---
@st.cache_data
def load_data():
    # Force reload by not caching heavily if you are iterating fast
    metrics = pd.read_csv('production_metrics.csv')
    raw = pd.read_csv('production_raw_data.csv')
    return metrics, raw

try:
    metrics_df, raw_df = load_data()
    # Use .get to fail gracefully if columns missing
    LG_AVG_GAP = metrics_df.get('lg_avg_gap', pd.Series([0])).iloc[0]
    LG_AVG_WIDTH = metrics_df.get('lg_avg_width', pd.Series([0])).iloc[0]
    LG_90_EV = metrics_df.get('lg_90_ev', pd.Series([0])).iloc[0]
except:
    st.error("Data files missing. Run builder.py locally first.")
    st.stop()

st.sidebar.header("Swing Efficiency Tool")
# Filter for players with enough data
valid_players = metrics_df[metrics_df['observations'] >= 50]['player_name'].unique()
player_list = sorted(valid_players)
default_idx = list(player_list).index("Aaron Judge") if "Aaron Judge" in player_list else 0
target_player = st.sidebar.selectbox("Search Player", player_list, index=default_idx)

# --- PLOTTING LOGIC ---
def render_report_card(player):
    p_met = metrics_df[metrics_df['player_name'] == player].iloc[0]
    p_raw = raw_df[raw_df['player_name'] == player]
    
    # Layout: 16:8.5 (Widescreen)
    fig = plt.figure(figsize=(16, 8.5)) 
    gs = gridspec.GridSpec(3, 3, height_ratios=[0.15, 1.5, 0.8])
    
    # 1. HEADER
    ax_h = fig.add_subplot(gs[0, :])
    ax_h.axis('off')
    ax_h.text(0, 0.5, player.upper(), fontsize=28, fontweight='bold', color=FG_DARK)
    ax_h.text(0, 0.1, "2025 Contact Efficiency Report", fontsize=12, color=FG_GREEN)
    ax_h.add_patch(patches.Rectangle((0, 0.85), 1, 0.1, color=FG_GREEN, transform=ax_h.transAxes))

    # 2. MAIN PLOT (Dual Axis)
    ax1 = fig.add_subplot(gs[1, 0:2])
    ax2 = ax1.twinx()
    
    ax1.scatter(p_raw['intercept_ball_minus_batter_pos_y_inches'], p_raw['launch_speed'], color=FG_GRAY, alpha=0.2, s=30)
    if 'woba_value' in p_raw.columns:
        ax2.scatter(p_raw['intercept_ball_minus_batter_pos_y_inches'], p_raw['woba_value'], color=FG_BLUE, alpha=0.2, s=20)

    x_range = np.linspace(0, 60, 100)

    # Draw EV Curve
    if p_met.get('ev_status') == 'Valid':
        a, b, c = p_met.get('ev_a'), p_met.get('ev_b'), p_met.get('ev_c')
        if pd.notna(a):
            y_ev = a*x_range**2 + b*x_range + c
            ax1.plot(x_range, y_ev, color=FG_GREEN, linewidth=4)
            ax1.axvline(p_met.get('ev_vertex'), color=FG_GREEN, linestyle=':')
    
    # Draw wOBA (Plateau vs Curve)
    # SAFELY ACCESS KEYS using .get()
    woba_plat_min = p_met.get('woba_plateau_min')
    woba_plat_max = p_met.get('woba_plateau_max')
    woba_plat_y   = p_met.get('woba_plateau_y')
    
    if pd.notna(woba_plat_min):
        # Draw Plateau Line
        ax2.plot([woba_plat_min, woba_plat_max], 
                 [woba_plat_y, woba_plat_y], 
                 color=FG_BLUE, linestyle='--', linewidth=4)
        
        mid_x = (woba_plat_min + woba_plat_max) / 2
        ax2.text(mid_x, woba_plat_y + 0.05, "High Production Zone", 
                 color=FG_BLUE, ha='center', fontsize=9, fontweight='bold', 
                 bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=2))
        
        if pd.notna(p_met.get('woba_vertex')):
            ax2.axvline(p_met.get('woba_vertex'), color=FG_BLUE, linestyle='-.')
            
    elif p_met.get('woba_status') == 'Valid' and pd.notna(p_met.get('woba_vertex')):
        # Fallback for valid curves
        ax2.axvline(p_met.get('woba_vertex'), color=FG_BLUE, linestyle='--')

    ax1.set_xlim(0, 60); ax1.set_ylim(bottom=60)
    ax2.set_ylim(0, 2.0)
    ax1.set_xlabel("Contact Point", fontweight='bold'); ax1.set_ylabel("EV (MPH)", fontweight='bold', color=FG_GREEN)
    ax2.set_ylabel("wOBA", fontweight='bold', color=FG_BLUE)
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # 3. METRICS SIDEBAR
    ax_met = fig.add_subplot(gs[1, 2])
    ax_met.axis('off')
    gap = p_met.get('Inefficiency_Gap')
    
    if pd.notna(gap):
        gap_txt = f"{gap:+.1f}\""
        color = FG_GREEN if abs(gap)<5 else FG_DARK
        ax_met.text(0.5, 0.9, "EFFICIENCY GAP", ha='center', fontsize=14, fontweight='bold', color=FG_GRAY)
        ax_met.text(0.5, 0.75, gap_txt, ha='center', fontsize=40, fontweight='bold', color=color)
        ax_met.text(0.5, 0.65, f"(Lg: {LG_AVG_GAP:+.1f})", ha='center', color=FG_GRAY)
    else:
        ax_met.text(0.5, 0.8, "GAP: N/A", ha='center', fontsize=20, color=FG_GRAY)

    ax_met.text(0.1, 0.5, "EV Peak", fontsize=12, fontweight='bold', color=FG_GREEN)
    ev_v = p_met.get('ev_vertex')
    ev_disp = f"{ev_v:.1f}\"" if pd.notna(ev_v) else "Plateau"
    ax_met.text(0.1, 0.42, ev_disp, fontsize=24, fontweight='bold')
    
    ax_met.text(0.1, 0.3, "wOBA Peak", fontsize=12, fontweight='bold', color=FG_BLUE)
    wb_v = p_met.get('woba_vertex')
    woba_disp = f"{wb_v:.1f}\"" if pd.notna(wb_v) else "N/A"
    ax_met.text(0.1, 0.22, woba_disp, fontsize=24, fontweight='bold')

    # 4. BOTTOM CHARTS
    # Horizontal Coverage
    ax_cov = fig.add_subplot(gs[2, 0])
    if 'plate_x' in p_raw.columns:
        inner = p_raw[p_raw['plate_x'].between(-0.8, -0.25)]['launch_speed'].mean()
        middle = p_raw[p_raw['plate_x'].between(-0.25, 0.25)]['launch_speed'].mean()
        outer = p_raw[p_raw['plate_x'].between(0.25, 0.8)]['launch_speed'].mean()
        evs = [0 if pd.isna(x) else x for x in [inner, middle, outer]]
        bars = ax_cov.bar(['In', 'Mid', 'Out'], evs, color=[FG_GREEN if e==max(evs) else FG_GRAY for e in evs], edgecolor=FG_DARK)
        ax_cov.set_ylim(bottom=70 if max(evs)>80 else 0)
        ax_cov.set_title("Horizontal Plate Coverage (Avg EV)", fontweight='bold', loc='left')
        for bar, ev in zip(bars, evs):
            if ev > 0: ax_cov.text(bar.get_x()+bar.get_width()/2, ev-5, f"{ev:.1f}", ha='center', color='white', fontweight='bold')

    # Swing Shape Report
    ax_shape = fig.add_subplot(gs[2, 1:])
    ax_shape.axis('off')
    ax_shape.add_patch(patches.Rectangle((0, 0.85), 1, 0.15, color=FG_GREEN, transform=ax_shape.transAxes))
    ax_shape.text(0.05, 0.925, "SWING SHAPE (90% EV Window)", transform=ax_shape.transAxes, fontsize=12, fontweight='bold', color='white', va='center')
    
    width = p_met.get('ev_width')
    width = width if pd.notna(width) else 0
    width_disp = f"{width:.1f}\"" if width > 0 else "N/A"
    
    ax_shape.text(0.25, 0.65, width_disp, ha='center', fontsize=30, fontweight='bold', color=FG_DARK, transform=ax_shape.transAxes)
    ax_shape.text(0.25, 0.55, "Power Window", ha='center', fontsize=10, color=FG_GRAY, transform=ax_shape.transAxes)
    ax_shape.text(0.25, 0.48, f"(Lg: {LG_AVG_WIDTH:.1f})", ha='center', fontsize=9, color=FG_LIGHT_GRAY, transform=ax_shape.transAxes)

    ev90 = p_met.get('ev_90')
    ev90_disp = f"{ev90:.1f}" if pd.notna(ev90) else "N/A"
    ax_shape.text(0.75, 0.65, ev90_disp, ha='center', fontsize=30, fontweight='bold', color=FG_DARK, transform=ax_shape.transAxes)
    ax_shape.text(0.75, 0.55, "90th% EV", ha='center', fontsize=10, color=FG_GRAY, transform=ax_shape.transAxes)
    ax_shape.text(0.75, 0.48, f"(Lg: {LG_90_EV:.1f})", ha='center', fontsize=9, color=FG_LIGHT_GRAY, transform=ax_shape.transAxes)

    # The Visual Bar Logic (Restored!)
    if p_met.get('ev_status') == 'Valid':
        a, b, c = p_met.get('ev_a'), p_met.get('ev_b'), p_met.get('ev_c')
        if pd.notna(a):
            # Recalculate coordinates for the bar
            peak_x = -b / (2 * a)
            peak_y = a * peak_x**2 + b * peak_x + c
            target_y = 0.90 * peak_y
            disc = b**2 - 4*a*(c - target_y)
            
            if disc >= 0:
                x1 = (-b + np.sqrt(disc)) / (2 * a)
                x2 = (-b - np.sqrt(disc)) / (2 * a)
                back, front = min(x1, x2), max(x1, x2)
                
                # Draw Bar
                plot_min, plot_max = 0, 60
                def scale_x(val): return 0.1 + (0.8 * (max(plot_min, min(plot_max, val)) - plot_min) / (plot_max - plot_min))
                
                # Base Line
                ax_shape.add_patch(patches.Rectangle((0.1, 0.29), 0.8, 0.02, color=FG_GRAY, alpha=0.3, transform=ax_shape.transAxes))
                
                # Player Bar
                start_x = scale_x(back)
                end_x = scale_x(front)
                if end_x - start_x > 0:
                    ax_shape.add_patch(patches.Rectangle((start_x, 0.27), end_x - start_x, 0.06, color=FG_GREEN, alpha=0.8, transform=ax_shape.transAxes))
                    ax_shape.text(start_x, 0.22, f"{back:.1f}\"", ha='center', fontsize=9, fontweight='bold', transform=ax_shape.transAxes)
                    ax_shape.text(end_x, 0.22, f"{front:.1f}\"", ha='center', fontsize=9, fontweight='bold', transform=ax_shape.transAxes)

                # Ghost Bracket (League Avg)
                # Center the league avg bracket on the player's window
                player_center = (start_x + end_x) / 2
                lg_width_scaled = (LG_AVG_WIDTH / (plot_max - plot_min)) * 0.8
                lg_start = player_center - (lg_width_scaled / 2)
                
                ax_shape.plot([lg_start, lg_start + lg_width_scaled], [0.15, 0.15], color=FG_LIGHT_GRAY, linewidth=2, transform=ax_shape.transAxes)
                ax_shape.text(player_center, 0.10, f"Lg Avg Width ({LG_AVG_WIDTH:.1f}\")", ha='center', fontsize=8, color=FG_LIGHT_GRAY, transform=ax_shape.transAxes)

    # Render
    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)

# --- MAIN ---
render_report_card(target_player)