"""
eda_plots.py — Phase 3: Exploratory Data Analysis
Generates 6 production-grade EDA charts answering specific business questions.
All charts saved to reports/figures/
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from loguru import logger

MEDIA_COLS = ['TV', 'Digital', 'Sponsorship', 'Content_Marketing',
              'Online_Marketing', 'Affiliates', 'SEM']

CHANNEL_COLORS = ['#1a73e8','#34a853','#fbbc04','#ea4335',
                  '#9c27b0','#00897b','#ff6d00']

COLORS = {
    'primary':  '#1a73e8',
    'secondary':'#34a853',
    'warning':  '#fbbc04',
    'danger':   '#ea4335',
    'bg':       '#f8f9fa',
    'grid':     '#e0e0e0',
}

plt.rcParams.update({
    'font.family':        'DejaVu Sans',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          True,
    'grid.alpha':         0.4,
    'figure.facecolor':   'white',
    'axes.facecolor':     '#f8f9fa',
})

FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_revenue_trend(df: pd.DataFrame, special: pd.DataFrame) -> str:
    """BQ1: What is the revenue trend and when do promotions fire?"""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(df['Date'], df['total_gmv']/1e6, alpha=0.15, color=COLORS['primary'])
    ax.plot(df['Date'], df['total_gmv']/1e6, color=COLORS['primary'],
            linewidth=2.5, marker='o', markersize=7, label='Total GMV (₹M)')

    sale_months = special['Date'].dt.to_period('M').unique()
    for _, row in df.iterrows():
        if row['Date'].to_period('M') in sale_months:
            ax.axvline(row['Date'], color=COLORS['warning'], alpha=0.4, linewidth=8)

    min_idx = df['total_gmv'].idxmin()
    ax.annotate(f"⚠ Min: ₹{df.loc[min_idx,'total_gmv']/1e6:.1f}M",
                xy=(df.loc[min_idx,'Date'], df.loc[min_idx,'total_gmv']/1e6),
                xytext=(30, 20), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color=COLORS['danger']),
                fontsize=9, color=COLORS['danger'])

    ax.set_title('Monthly Revenue (GMV) Trend — Jul 2015 to Jun 2016\n'
                 'Yellow bands = months with promotional events',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel('Revenue (₹ Millions)', fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'₹{x:.0f}M'))
    ax.legend(fontsize=9)
    plt.xticks(rotation=45)
    plt.tight_layout()
    path = str(FIGURES_DIR / "01_revenue_trend.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {path}")
    return path


def plot_media_spend(df: pd.DataFrame) -> str:
    """BQ2: How is media budget distributed across channels?"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    df_media = df[['Date'] + MEDIA_COLS].copy()
    for col in MEDIA_COLS:
        df_media[col] = pd.to_numeric(df_media[col], errors='coerce').fillna(0) / 1e6

    bottom = np.zeros(len(df_media))
    for i, col in enumerate(MEDIA_COLS):
        axes[0].bar(df_media['Date'], df_media[col], bottom=bottom,
                    label=col, color=CHANNEL_COLORS[i], alpha=0.85, width=20)
        bottom += df_media[col].values

    axes[0].set_title('Monthly Media Spend by Channel', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Spend (₹ Millions)', fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'₹{x:.0f}M'))
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)

    totals = df_media[MEDIA_COLS].sum()
    axes[1].pie(totals, labels=MEDIA_COLS, colors=CHANNEL_COLORS,
                autopct='%1.1f%%', startangle=90, pctdistance=0.75)
    axes[1].set_title('Total Spend Share by Channel', fontsize=12, fontweight='bold')

    plt.suptitle('Media Investment Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = str(FIGURES_DIR / "02_media_spend.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {path}")
    return path


def plot_correlation_heatmap(df: pd.DataFrame) -> str:
    """BQ3: Which channels correlate strongest with revenue?"""
    fig, ax = plt.subplots(figsize=(12, 9))
    corr_cols = MEDIA_COLS + ['total_gmv', 'NPS', 'total_Discount']
    corr_matrix = df[corr_cols].apply(pd.to_numeric, errors='coerce').corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                cmap='RdYlGn', center=0, vmin=-1, vmax=1,
                square=True, ax=ax, annot_kws={'size': 9})
    ax.set_title('Correlation Matrix — Media Channels vs Revenue',
                 fontsize=13, fontweight='bold', pad=15)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    path = str(FIGURES_DIR / "03_correlation_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {path}")
    return path


def plot_channel_scatter(df: pd.DataFrame) -> str:
    """BQ4: What is the spend-to-revenue relationship per channel?"""
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()
    for i, col in enumerate(MEDIA_COLS):
        x = pd.to_numeric(df[col], errors='coerce').fillna(0) / 1e6
        y = df['total_gmv'] / 1e6
        corr = x.corr(y)
        axes[i].scatter(x, y, color=CHANNEL_COLORS[i], s=80, alpha=0.8)
        if x.std() > 0:
            z = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            axes[i].plot(x_line, np.poly1d(z)(x_line), '--',
                         color=CHANNEL_COLORS[i], alpha=0.6)
        axes[i].set_title(f'{col}\nr = {corr:.2f}', fontsize=10, fontweight='bold')
        axes[i].set_xlabel('Spend (₹M)', fontsize=8)
        axes[i].set_ylabel('GMV (₹M)', fontsize=8)
        border = COLORS['secondary'] if abs(corr) > 0.7 else \
                 COLORS['warning'] if abs(corr) > 0.4 else COLORS['danger']
        for spine in axes[i].spines.values():
            spine.set_edgecolor(border)
            spine.set_linewidth(2)
    axes[-1].set_visible(False)
    plt.suptitle('Channel Spend vs Revenue — Scatter + Trend\n'
                 'Border: Green=strong (r>0.7), Yellow=moderate, Red=weak',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = str(FIGURES_DIR / "04_channel_scatter.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {path}")
    return path


def plot_nps_vs_gmv(df: pd.DataFrame) -> str:
    """BQ5: Does NPS lead or lag revenue?"""
    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax2 = ax1.twinx()
    ax1.plot(df['Date'], df['total_gmv']/1e6, color=COLORS['primary'],
             linewidth=2.5, marker='o', markersize=7, label='GMV (₹M)')
    ax1.fill_between(df['Date'], df['total_gmv']/1e6, alpha=0.1, color=COLORS['primary'])
    ax2.plot(df['Date'], df['NPS'], color=COLORS['danger'],
             linewidth=2.5, marker='s', markersize=7, linestyle='--', label='NPS')
    ax1.set_ylabel('Revenue GMV (₹M)', color=COLORS['primary'], fontsize=10)
    ax2.set_ylabel('NPS Score', color=COLORS['danger'], fontsize=10)
    ax2.set_ylim(0, 100)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], fontsize=9)
    ax1.set_title('Revenue vs NPS Over Time — Lead/Lag Analysis',
                  fontsize=13, fontweight='bold', pad=15)
    plt.xticks(rotation=45)
    plt.tight_layout()
    path = str(FIGURES_DIR / "05_nps_vs_gmv.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {path}")
    return path


def plot_category_revenue(df: pd.DataFrame) -> str:
    """BQ6: Which product categories drive revenue?"""
    rev_cols   = ['Revenue_Camera','Revenue_CameraAccessory',
                  'Revenue_EntertainmentSmall','Revenue_GameCDDVD','Revenue_GamingHardware']
    rev_labels = ['Camera','Cam Accessory','Entertainment','Game CD/DVD','Gaming HW']
    cat_colors = ['#1a73e8','#34a853','#fbbc04','#ea4335','#9c27b0']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    bottom = np.zeros(len(df))
    for i, (col, label) in enumerate(zip(rev_cols, rev_labels)):
        vals = df[col].fillna(0) / 1e6
        axes[0].bar(df['Date'], vals, bottom=bottom,
                    label=label, color=cat_colors[i], alpha=0.85, width=20)
        bottom += vals.values
    axes[0].set_title('Monthly Revenue by Product Category', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Revenue (₹ Millions)', fontsize=10)
    axes[0].legend(fontsize=8)
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)

    totals = [df[col].sum()/1e6 for col in rev_cols]
    axes[1].pie(totals, labels=rev_labels, colors=cat_colors,
                autopct='%1.1f%%', startangle=90, pctdistance=0.75)
    axes[1].set_title('Total Revenue Share by Category', fontsize=12, fontweight='bold')
    plt.suptitle('Product Category Revenue Analysis',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = str(FIGURES_DIR / "06_category_revenue.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {path}")
    return path


def run_full_eda(df, special):
    """Run all 6 EDA charts. Returns list of saved file paths."""
    logger.info("Starting full EDA...")
    paths = [
        plot_revenue_trend(df, special),
        plot_media_spend(df),
        plot_correlation_heatmap(df),
        plot_channel_scatter(df),
        plot_nps_vs_gmv(df),
        plot_category_revenue(df),
    ]
    logger.info(f"EDA complete. {len(paths)} charts saved to {FIGURES_DIR}")
    return paths
