#!/usr/bin/env python3
"""
Create a comprehensive figure showing habitable planets and their correlations to host stars.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

def load_and_prepare_data():
    """Load the habitability scores and original Kepler data."""
    print("Loading habitability scores...")
    habitability_df = pd.read_csv('/Users/armin/Documents/xplorer/data/processed/cumulative_2025.10.04_15.04.09_habitability_scores.csv')
    
    print("Loading original Kepler data...")
    # Load original data to get stellar parameters
    original_df = pd.read_csv('/Users/armin/Documents/xplorer/data/processed/cumulative_2025.10.04_15.04.09.csv', skiprows=144)
    
    # Merge the datasets
    merged_df = habitability_df.merge(original_df, left_index=True, right_index=True, how='left')
    
    return merged_df

def create_habitable_stars_figure(df):
    """Create comprehensive figure showing habitable planets and stellar correlations."""
    
    # Filter for habitable planets (moderate to high habitability)
    habitable = df[df['habitability_score'] >= 0.4].copy()
    print(f"Found {len(habitable)} habitable planets")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Define colors for habitability categories
    colors = {'High': '#2E8B57', 'Moderate': '#FFD700', 'Low': '#FF6347', 'Very Low': '#8B0000'}
    
    # 1. Stellar Temperature vs Planet Temperature (Top Left)
    ax1 = plt.subplot(3, 4, 1)
    scatter = ax1.scatter(df['koi_steff'], df['temperature_k'], 
                          c=df['habitability_score'], cmap='viridis', 
                          alpha=0.6, s=20)
    ax1.scatter(habitable['koi_steff'], habitable['temperature_k'], 
                c='red', s=100, alpha=0.8, edgecolors='black', linewidth=2, label='Habitable')
    ax1.set_xlabel('Stellar Temperature (K)')
    ax1.set_ylabel('Planet Temperature (K)')
    ax1.set_title('Stellar vs Planet Temperature')
    ax1.legend()
    plt.colorbar(scatter, ax=ax1, label='Habitability Score')
    
    # 2. Stellar Radius vs Planet Radius (Top Center)
    ax2 = plt.subplot(3, 4, 2)
    scatter2 = ax2.scatter(df['koi_srad'], df['radius_earth'], 
                          c=df['habitability_score'], cmap='viridis', 
                          alpha=0.6, s=20)
    ax2.scatter(habitable['koi_srad'], habitable['radius_earth'], 
                c='red', s=100, alpha=0.8, edgecolors='black', linewidth=2, label='Habitable')
    ax2.set_xlabel('Stellar Radius (R☉)')
    ax2.set_ylabel('Planet Radius (R⊕)')
    ax2.set_title('Stellar vs Planet Radius')
    ax2.legend()
    plt.colorbar(scatter2, ax=ax2, label='Habitability Score')
    
    # 3. Stellar Mass vs Planet Mass (Top Right)
    ax3 = plt.subplot(3, 4, 3)
    scatter3 = ax3.scatter(df['koi_smass'], df['estimated_mass_earth'], 
                          c=df['habitability_score'], cmap='viridis', 
                          alpha=0.6, s=20)
    ax3.scatter(habitable['koi_smass'], habitable['estimated_mass_earth'], 
                c='red', s=100, alpha=0.8, edgecolors='black', linewidth=2, label='Habitable')
    ax3.set_xlabel('Stellar Mass (M☉)')
    ax3.set_ylabel('Planet Mass (M⊕)')
    ax3.set_title('Stellar vs Planet Mass')
    ax3.legend()
    plt.colorbar(scatter3, ax=ax3, label='Habitability Score')
    
    # 4. Insolation vs Stellar Luminosity (Top Far Right)
    ax4 = plt.subplot(3, 4, 4)
    # Calculate stellar luminosity from radius and temperature
    stellar_luminosity = (df['koi_srad'] ** 2) * ((df['koi_steff'] / 5778) ** 4)
    scatter4 = ax4.scatter(stellar_luminosity, df['insolation_flux'], 
                          c=df['habitability_score'], cmap='viridis', 
                          alpha=0.6, s=20)
    ax4.scatter(stellar_luminosity[habitable.index], habitable['insolation_flux'], 
                c='red', s=100, alpha=0.8, edgecolors='black', linewidth=2, label='Habitable')
    ax4.set_xlabel('Stellar Luminosity (L☉)')
    ax4.set_ylabel('Insolation Flux (Earth flux)')
    ax4.set_title('Stellar Luminosity vs Insolation')
    ax4.legend()
    plt.colorbar(scatter4, ax=ax4, label='Habitability Score')
    
    # 5. Habitable Zone Position (Middle Left)
    ax5 = plt.subplot(3, 4, 5)
    # Create HZ boundaries based on stellar temperature
    stellar_temp = df['koi_steff']
    hz_inner = 0.95 * (stellar_temp / 5778) ** 2
    hz_outer = 1.37 * (stellar_temp / 5778) ** 2
    
    # Plot all planets
    ax5.scatter(df['semi_major_axis_au'], df['koi_steff'], 
                c=df['habitability_score'], cmap='viridis', alpha=0.6, s=20)
    ax5.scatter(habitable['semi_major_axis_au'], habitable['koi_steff'], 
                c='red', s=100, alpha=0.8, edgecolors='black', linewidth=2, label='Habitable')
    
    # Add HZ boundaries
    temp_range = np.linspace(3000, 7000, 100)
    hz_inner_line = 0.95 * (temp_range / 5778) ** 2
    hz_outer_line = 1.37 * (temp_range / 5778) ** 2
    ax5.plot(hz_inner_line, temp_range, 'g--', alpha=0.7, label='Conservative HZ')
    ax5.plot(hz_outer_line, temp_range, 'g--', alpha=0.7)
    ax5.fill_betweenx(temp_range, hz_inner_line, hz_outer_line, alpha=0.2, color='green', label='HZ')
    
    ax5.set_xlabel('Semi-major Axis (AU)')
    ax5.set_ylabel('Stellar Temperature (K)')
    ax5.set_title('Habitable Zone Position')
    ax5.legend()
    
    # 6. Stellar Type Distribution (Middle Center)
    ax6 = plt.subplot(3, 4, 6)
    # Classify stars by temperature
    def classify_star_type(temp):
        if temp < 3700:
            return 'M'
        elif temp < 5200:
            return 'K'
        elif temp < 6000:
            return 'G'
        elif temp < 7500:
            return 'F'
        else:
            return 'A'
    
    df['star_type'] = df['koi_steff'].apply(classify_star_type)
    habitable['star_type'] = habitable['koi_steff'].apply(classify_star_type)
    
    # Count by star type
    all_counts = df['star_type'].value_counts().sort_index()
    hab_counts = habitable['star_type'].value_counts()
    
    x_pos = np.arange(len(all_counts))
    bars = ax6.bar(x_pos, all_counts.values, alpha=0.7, color='lightblue', label='All Planets')
    hab_bars = ax6.bar(x_pos, [hab_counts.get(star, 0) for star in all_counts.index], 
                       alpha=0.8, color='red', label='Habitable')
    
    ax6.set_xlabel('Stellar Type')
    ax6.set_ylabel('Number of Planets')
    ax6.set_title('Planet Distribution by Stellar Type')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(all_counts.index)
    ax6.legend()
    
    # 7. Orbital Period vs Stellar Mass (Middle Right)
    ax7 = plt.subplot(3, 4, 7)
    scatter7 = ax7.scatter(df['koi_smass'], df['period_days'], 
                          c=df['habitability_score'], cmap='viridis', 
                          alpha=0.6, s=20)
    ax7.scatter(habitable['koi_smass'], habitable['period_days'], 
                c='red', s=100, alpha=0.8, edgecolors='black', linewidth=2, label='Habitable')
    ax7.set_xlabel('Stellar Mass (M☉)')
    ax7.set_ylabel('Orbital Period (days)')
    ax7.set_title('Orbital Period vs Stellar Mass')
    ax7.legend()
    plt.colorbar(scatter7, ax=ax7, label='Habitability Score')
    
    # 8. ESI vs Stellar Temperature (Middle Far Right)
    ax8 = plt.subplot(3, 4, 8)
    scatter8 = ax8.scatter(df['koi_steff'], df['esi'], 
                          c=df['habitability_score'], cmap='viridis', 
                          alpha=0.6, s=20)
    ax8.scatter(habitable['koi_steff'], habitable['esi'], 
                c='red', s=100, alpha=0.8, edgecolors='black', linewidth=2, label='Habitable')
    ax8.axhline(y=0.8, color='orange', linestyle='--', alpha=0.7, label='ESI > 0.8')
    ax8.set_xlabel('Stellar Temperature (K)')
    ax8.set_ylabel('Earth Similarity Index')
    ax8.set_title('ESI vs Stellar Temperature')
    ax8.legend()
    plt.colorbar(scatter8, ax=ax8, label='Habitability Score')
    
    # 9. Top Habitable Planets Table (Bottom Left)
    ax9 = plt.subplot(3, 4, 9)
    ax9.axis('off')
    
    # Get top 10 habitable planets
    top_10 = habitable.nlargest(10, 'habitability_score')
    
    # Create table
    table_data = []
    for i, (idx, row) in enumerate(top_10.iterrows(), 1):
        name = row['planet_name'] if pd.notna(row['planet_name']) else f"Planet {i}"
        table_data.append([
            f"{i}",
            name[:15] + "..." if len(str(name)) > 15 else name,
            f"{row['habitability_score']:.3f}",
            f"{row['koi_steff']:.0f}",
            f"{row['koi_srad']:.2f}"
        ])
    
    table = ax9.table(cellText=table_data,
                     colLabels=['Rank', 'Planet', 'Score', 'T* (K)', 'R* (R☉)'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    ax9.set_title('Top 10 Habitable Planets', fontsize=12, fontweight='bold')
    
    # 10. Correlation Heatmap (Bottom Center)
    ax10 = plt.subplot(3, 4, 10)
    
    # Select key parameters for correlation
    corr_data = habitable[['habitability_score', 'esi', 'temperature_k', 'radius_earth', 
                          'koi_steff', 'koi_srad', 'koi_smass', 'insolation_flux', 
                          'semi_major_axis_au', 'period_days']].corr()
    
    im = ax10.imshow(corr_data, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax10.set_xticks(range(len(corr_data.columns)))
    ax10.set_yticks(range(len(corr_data.columns)))
    ax10.set_xticklabels([col.replace('_', ' ').title() for col in corr_data.columns], 
                        rotation=45, ha='right', fontsize=8)
    ax10.set_yticklabels([col.replace('_', ' ').title() for col in corr_data.columns], 
                        fontsize=8)
    ax10.set_title('Parameter Correlations', fontsize=10)
    
    # Add correlation values
    for i in range(len(corr_data.columns)):
        for j in range(len(corr_data.columns)):
            text = ax10.text(j, i, f'{corr_data.iloc[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=6)
    
    plt.colorbar(im, ax=ax10, label='Correlation')
    
    # 11. Stellar Mass vs Radius (Bottom Right)
    ax11 = plt.subplot(3, 4, 11)
    scatter11 = ax11.scatter(df['koi_smass'], df['koi_srad'], 
                            c=df['habitability_score'], cmap='viridis', 
                            alpha=0.6, s=20)
    ax11.scatter(habitable['koi_smass'], habitable['koi_srad'], 
                c='red', s=100, alpha=0.8, edgecolors='black', linewidth=2, label='Habitable')
    ax11.set_xlabel('Stellar Mass (M☉)')
    ax11.set_ylabel('Stellar Radius (R☉)')
    ax11.set_title('Stellar Mass-Radius Relation')
    ax11.legend()
    plt.colorbar(scatter11, ax=ax11, label='Habitability Score')
    
    # 12. Summary Statistics (Bottom Far Right)
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')
    
    # Calculate summary statistics
    total_planets = len(df)
    habitable_count = len(habitable)
    hz_conservative = len(df[df['in_hz_conservative'] == True])
    hz_optimistic = len(df[df['in_hz_optimistic'] == True])
    
    # Stellar type distribution for habitable planets
    hab_star_types = habitable['star_type'].value_counts()
    
    stats_text = f"""
    HABITABLE PLANETS SUMMARY
    
    Total Planets: {total_planets:,}
    Habitable (Score ≥ 0.4): {habitable_count}
    Conservative HZ: {hz_conservative}
    Optimistic HZ: {hz_optimistic}
    
    HABITABLE BY STAR TYPE:
    """
    
    for star_type in ['M', 'K', 'G', 'F', 'A']:
        count = hab_star_types.get(star_type, 0)
        stats_text += f"{star_type}: {count}\n"
    
    stats_text += f"""
    TOP STELLAR TYPE: {hab_star_types.index[0] if len(hab_star_types) > 0 else 'N/A'}
    
    AVERAGE PROPERTIES:
    Stellar Temp: {habitable['koi_steff'].mean():.0f} K
    Stellar Mass: {habitable['koi_smass'].mean():.2f} M☉
    Stellar Radius: {habitable['koi_srad'].mean():.2f} R☉
    """
    
    ax12.text(0.05, 0.95, stats_text, transform=ax12.transAxes, 
              fontsize=9, verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    return fig

def main():
    """Main execution function."""
    print("Creating habitable planets and stellar correlations figure...")
    
    # Load data
    df = load_and_prepare_data()
    
    # Create figure
    fig = create_habitable_stars_figure(df)
    
    # Save figure
    output_path = '/Users/armin/Documents/xplorer/srcs/figures/fig_habitable_planets_and_stars.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to: {output_path}")
    
    # Also create a simplified version
    create_simplified_figure(df, output_path.replace('.png', '_simplified.png'))
    
    plt.show()

def create_simplified_figure(df, output_path):
    """Create a simplified version focusing on key relationships."""
    
    habitable = df[df['habitability_score'] >= 0.4].copy()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Stellar Temperature vs Planet Temperature
    ax1 = axes[0, 0]
    ax1.scatter(df['koi_steff'], df['temperature_k'], 
                c=df['habitability_score'], cmap='viridis', alpha=0.6, s=20)
    ax1.scatter(habitable['koi_steff'], habitable['temperature_k'], 
                c='red', s=100, alpha=0.8, edgecolors='black', linewidth=2, label='Habitable')
    ax1.set_xlabel('Stellar Temperature (K)')
    ax1.set_ylabel('Planet Temperature (K)')
    ax1.set_title('Stellar vs Planet Temperature')
    ax1.legend()
    
    # 2. Habitable Zone Position
    ax2 = axes[0, 1]
    ax2.scatter(df['semi_major_axis_au'], df['koi_steff'], 
                c=df['habitability_score'], cmap='viridis', alpha=0.6, s=20)
    ax2.scatter(habitable['semi_major_axis_au'], habitable['koi_steff'], 
                c='red', s=100, alpha=0.8, edgecolors='black', linewidth=2, label='Habitable')
    
    # Add HZ boundaries
    temp_range = np.linspace(3000, 7000, 100)
    hz_inner = 0.95 * (temp_range / 5778) ** 2
    hz_outer = 1.37 * (temp_range / 5778) ** 2
    ax2.plot(hz_inner, temp_range, 'g--', alpha=0.7, label='Conservative HZ')
    ax2.plot(hz_outer, temp_range, 'g--', alpha=0.7)
    ax2.fill_betweenx(temp_range, hz_inner, hz_outer, alpha=0.2, color='green')
    
    ax2.set_xlabel('Semi-major Axis (AU)')
    ax2.set_ylabel('Stellar Temperature (K)')
    ax2.set_title('Habitable Zone Position')
    ax2.legend()
    
    # 3. Stellar Type Distribution
    ax3 = axes[0, 2]
    df['star_type'] = df['koi_steff'].apply(lambda t: 'M' if t < 3700 else 'K' if t < 5200 else 'G' if t < 6000 else 'F' if t < 7500 else 'A')
    habitable['star_type'] = habitable['koi_steff'].apply(lambda t: 'M' if t < 3700 else 'K' if t < 5200 else 'G' if t < 6000 else 'F' if t < 7500 else 'A')
    
    all_counts = df['star_type'].value_counts().sort_index()
    hab_counts = habitable['star_type'].value_counts()
    
    x_pos = np.arange(len(all_counts))
    ax3.bar(x_pos, all_counts.values, alpha=0.7, color='lightblue', label='All Planets')
    ax3.bar(x_pos, [hab_counts.get(star, 0) for star in all_counts.index], 
            alpha=0.8, color='red', label='Habitable')
    
    ax3.set_xlabel('Stellar Type')
    ax3.set_ylabel('Number of Planets')
    ax3.set_title('Planet Distribution by Stellar Type')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(all_counts.index)
    ax3.legend()
    
    # 4. ESI vs Stellar Temperature
    ax4 = axes[1, 0]
    ax4.scatter(df['koi_steff'], df['esi'], 
                c=df['habitability_score'], cmap='viridis', alpha=0.6, s=20)
    ax4.scatter(habitable['koi_steff'], habitable['esi'], 
                c='red', s=100, alpha=0.8, edgecolors='black', linewidth=2, label='Habitable')
    ax4.axhline(y=0.8, color='orange', linestyle='--', alpha=0.7, label='ESI > 0.8')
    ax4.set_xlabel('Stellar Temperature (K)')
    ax4.set_ylabel('Earth Similarity Index')
    ax4.set_title('ESI vs Stellar Temperature')
    ax4.legend()
    
    # 5. Stellar Mass vs Radius
    ax5 = axes[1, 1]
    ax5.scatter(df['koi_smass'], df['koi_srad'], 
                c=df['habitability_score'], cmap='viridis', alpha=0.6, s=20)
    ax5.scatter(habitable['koi_smass'], habitable['koi_srad'], 
                c='red', s=100, alpha=0.8, edgecolors='black', linewidth=2, label='Habitable')
    ax5.set_xlabel('Stellar Mass (M☉)')
    ax5.set_ylabel('Stellar Radius (R☉)')
    ax5.set_title('Stellar Mass-Radius Relation')
    ax5.legend()
    
    # 6. Summary Statistics
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    total_planets = len(df)
    habitable_count = len(habitable)
    hz_conservative = len(df[df['in_hz_conservative'] == True])
    
    stats_text = f"""
    HABITABLE PLANETS SUMMARY
    
    Total Planets: {total_planets:,}
    Habitable (Score ≥ 0.4): {habitable_count}
    Conservative HZ: {hz_conservative}
    
    HABITABLE BY STAR TYPE:
    """
    
    hab_star_types = habitable['star_type'].value_counts()
    for star_type in ['M', 'K', 'G', 'F', 'A']:
        count = hab_star_types.get(star_type, 0)
        stats_text += f"{star_type}: {count}\n"
    
    stats_text += f"""
    AVERAGE STELLAR PROPERTIES:
    Temperature: {habitable['koi_steff'].mean():.0f} K
    Mass: {habitable['koi_smass'].mean():.2f} M☉
    Radius: {habitable['koi_srad'].mean():.2f} R☉
    """
    
    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, 
              fontsize=10, verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Simplified figure saved to: {output_path}")

if __name__ == '__main__':
    main()
