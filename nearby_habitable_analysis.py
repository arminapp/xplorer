#!/usr/bin/env python3
"""
Identify closest potentially habitable exoplanets.
Combines our habitability scores with distance estimates.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load our habitability results
df = pd.read_csv('cumulative_2025.10.04_15.04.09_habitability_scores.csv')

# Known distances for notable Kepler planets (from literature and Gaia)
# Format: planet_name -> distance in light years
known_distances = {
    # Top habitable candidates with known distances
    'Kepler-186 f': 580,      # Famous habitable zone planet
    'Kepler-186 e': 580,      # Same system
    'Kepler-442 b': 1206,     # One of our top candidates
    'Kepler-438 b': 640,      # High ESI planet
    'Kepler-296 e': 1650,     # Our #1 ranked planet
    'Kepler-62 e': 1200,      # Historic habitable zone discovery
    'Kepler-62 f': 1200,      # Same system
    'Kepler-452 b': 1800,     # "Earth's cousin"
    'Kepler-1229 b': 770,     # Habitable zone super-Earth
    'Kepler-1410 b': 1600,    # Our #3 candidate
    'Kepler-1544 b': 1300,
    'Kepler-1638 b': 2700,
    'Kepler-1649 c': 300,     # VERY close! Recently discovered
    'Kepler-1652 b': 822,
    'Kepler-283 c': 1740,
    'Kepler-296 f': 1650,     # Same system as our #1
    'Kepler-395 c': 1100,     # Our top 10
    'Kepler-440 b': 851,
    'Kepler-443 b': 2540,
    'Kepler-705 b': 1780,
    'Kepler-1090 b': 2043,
    'Kepler-1455 b': 2150,
    'Kepler-1512 b': 3000,
    'Kepler-1544 b': 1300,
    'Kepler-1606 b': 2900,
    'Kepler-174 d': 1174,

    # Other notable planets
    'Kepler-22 b': 620,       # First HZ planet discovery
    'Kepler-69 c': 2700,
}

# Estimate distances for other Kepler planets using apparent magnitude
# Distance modulus: m - M = 5*log10(d) - 5
# For Kepler stars, we can estimate rough distances from magnitude
def estimate_distance_from_magnitude(kepmag):
    """
    Rough distance estimate from Kepler magnitude.
    Assumes Sun-like stars (absolute magnitude ~4.5)
    Returns distance in light years.
    """
    if pd.isna(kepmag) or kepmag <= 0:
        return np.nan

    # Typical Kepler field is 400-3000 light years
    # Using distance modulus with assumed absolute magnitude
    M_abs = 4.5  # Assume Sun-like star
    distance_parsecs = 10 ** ((kepmag - M_abs + 5) / 5)
    distance_ly = distance_parsecs * 3.262  # parsecs to light years

    # Kepler observed stars in a specific field, typical range
    distance_ly = np.clip(distance_ly, 100, 5000)

    return distance_ly

# Add distance column
df['distance_ly'] = df['planet_name'].map(known_distances)

# For planets without known distance, use typical Kepler field distance
print("Using known distances for notable Kepler planets...")
print(f"Known distances available for {df['distance_ly'].notna().sum()} planets")

print("="*80)
print("CLOSEST POTENTIALLY HABITABLE EXOPLANETS")
print("="*80)

# Filter for decent habitability and known/estimated distance
habitable = df[
    (df['habitability_score'] >= 0.3) &
    (df['distance_ly'] > 0) &
    (df['radius_earth'] > 0) &
    (df['temperature_k'] > 0)
].copy()

# Sort by distance
habitable_sorted = habitable.sort_values('distance_ly')

print(f"\nFound {len(habitable_sorted)} potentially habitable planets with distance estimates")
print("\n" + "="*80)
print("TOP 20 CLOSEST HABITABLE CANDIDATES")
print("="*80)

# Display top 20 closest
top_20_closest = habitable_sorted.head(20)

for idx, (i, row) in enumerate(top_20_closest.iterrows(), 1):
    known_dist = row['planet_name'] in known_distances
    dist_marker = "✓" if known_dist else "~"

    print(f"\n{idx}. {row['planet_name']}")
    print(f"   Distance: {dist_marker}{row['distance_ly']:.0f} light years" +
          (" (known)" if known_dist else " (estimated)"))
    print(f"   Habitability Score: {row['habitability_score']:.3f} ({row['habitability_category']})")
    print(f"   ESI: {row['esi']:.3f}")
    print(f"   Radius: {row['radius_earth']:.2f} R⊕, Temp: {row['temperature_k']:.0f} K ({row['temperature_k']-273:.0f}°C)")
    print(f"   Status: {row['disposition']}, In HZ: {'Yes' if row['in_hz_conservative'] else ('Optimistic' if row['in_hz_optimistic'] else 'No')}")

# Save results
top_20_closest.to_csv('top_20_closest_habitable.csv', index=False)
print("\n" + "="*80)
print(f"Saved to: top_20_closest_habitable.csv")

# Best combination: High habitability AND close
print("\n" + "="*80)
print("BEST TARGETS: HIGH HABITABILITY + CLOSE DISTANCE")
print("="*80)

best_targets = habitable[
    (habitable['habitability_score'] >= 0.5) &
    (habitable['distance_ly'] < 1000)
].sort_values('habitability_score', ascending=False)

print(f"\nFound {len(best_targets)} planets with habitability ≥ 0.5 AND distance < 1000 ly")

if len(best_targets) > 0:
    print("\nTHESE ARE THE PRIME TARGETS FOR OBSERVATION:")
    for idx, (i, row) in enumerate(best_targets.iterrows(), 1):
        known_dist = row['planet_name'] in known_distances
        dist_marker = "✓" if known_dist else "~"
        print(f"\n{idx}. {row['planet_name']}")
        print(f"   Distance: {dist_marker}{row['distance_ly']:.0f} ly | Habitability: {row['habitability_score']:.3f} | ESI: {row['esi']:.3f}")
        print(f"   {row['radius_earth']:.2f} R⊕, {row['temperature_k']:.0f} K, {row['period_days']:.1f} day orbit")
else:
    print("\nNo planets found with both high habitability and close distance.")
    print("Best compromise:")
    compromise = habitable[habitable['distance_ly'] < 1500].nlargest(5, 'habitability_score')
    for idx, (i, row) in enumerate(compromise.iterrows(), 1):
        known_dist = row['planet_name'] in known_distances
        dist_marker = "✓" if known_dist else "~"
        print(f"{idx}. {row['planet_name']}: {dist_marker}{row['distance_ly']:.0f} ly, score {row['habitability_score']:.3f}")

# Visualizations
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS...")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Distance vs Habitability scatter
ax1 = axes[0, 0]
scatter_data = habitable[habitable['distance_ly'] < 3000]
scatter = ax1.scatter(scatter_data['distance_ly'],
                     scatter_data['habitability_score'],
                     c=scatter_data['esi'],
                     s=100*scatter_data['radius_earth'],
                     alpha=0.6, cmap='RdYlGn',
                     edgecolors='black', linewidth=1)

# Annotate closest high-scoring planets
annotate_data = habitable[(habitable['habitability_score'] > 0.5) |
                          (habitable['distance_ly'] < 400)]
for _, row in annotate_data.head(10).iterrows():
    if pd.notna(row['planet_name']) and isinstance(row['planet_name'], str):
        ax1.annotate(row['planet_name'],
                    (row['distance_ly'], row['habitability_score']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, fontweight='bold')

ax1.set_xlabel('Distance (Light Years)', fontsize=12)
ax1.set_ylabel('Habitability Score', fontsize=12)
ax1.set_title('Distance vs Habitability Score', fontsize=13, fontweight='bold')
ax1.axhline(0.5, color='green', linestyle='--', alpha=0.5, label='High habitability threshold')
ax1.axvline(1000, color='red', linestyle='--', alpha=0.5, label='1000 ly (observation limit)')
ax1.legend()
ax1.grid(alpha=0.3)
cbar1 = plt.colorbar(scatter, ax=ax1)
cbar1.set_label('ESI', fontsize=10)

# 2. Distribution of distances
ax2 = axes[0, 1]
ax2.hist(habitable['distance_ly'], bins=30, color='#3498db', alpha=0.7, edgecolor='black')
ax2.axvline(habitable['distance_ly'].median(), color='red', linestyle='--',
            linewidth=2, label=f'Median: {habitable["distance_ly"].median():.0f} ly')
ax2.set_xlabel('Distance (Light Years)', fontsize=12)
ax2.set_ylabel('Number of Habitable Planets', fontsize=12)
ax2.set_title('Distance Distribution of Habitable Candidates', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

# 3. Top 20 closest planets
ax3 = axes[1, 0]
top_10_plot = top_20_closest.head(10)
y_pos = np.arange(len(top_10_plot))
colors = ['#27ae60' if score >= 0.5 else '#f39c12' if score >= 0.3 else '#e67e22'
          for score in top_10_plot['habitability_score']]
ax3.barh(y_pos, top_10_plot['distance_ly'], color=colors, edgecolor='black', linewidth=1)
ax3.set_yticks(y_pos)
ax3.set_yticklabels(top_10_plot['planet_name'], fontsize=9)
ax3.invert_yaxis()
ax3.set_xlabel('Distance (Light Years)', fontsize=12)
ax3.set_title('10 Closest Potentially Habitable Planets', fontsize=13, fontweight='bold')
ax3.grid(axis='x', alpha=0.3)

# Add habitability scores as text
for i, (idx, row) in enumerate(top_10_plot.iterrows()):
    ax3.text(row['distance_ly'] + 20, i, f" {row['habitability_score']:.2f}",
            va='center', fontsize=9, fontweight='bold')

# 4. Distance vs ESI with temperature coloring
ax4 = axes[1, 1]
scatter2 = ax4.scatter(scatter_data['distance_ly'],
                      scatter_data['esi'],
                      c=scatter_data['temperature_k'],
                      s=80,
                      alpha=0.6, cmap='coolwarm',
                      edgecolors='black', linewidth=0.5)
ax4.axhspan(0.8, 1.0, alpha=0.2, color='green', label='High ESI (>0.8)')
ax4.axvline(1000, color='red', linestyle='--', alpha=0.5, label='1000 ly limit')
ax4.set_xlabel('Distance (Light Years)', fontsize=12)
ax4.set_ylabel('Earth Similarity Index', fontsize=12)
ax4.set_title('Distance vs ESI (colored by temperature)', fontsize=13, fontweight='bold')
ax4.legend()
ax4.grid(alpha=0.3)
cbar2 = plt.colorbar(scatter2, ax=ax4)
cbar2.set_label('Temperature (K)', fontsize=10)

plt.tight_layout()
plt.savefig('fig_closest_habitable_planets.png', dpi=300, bbox_inches='tight')
print("Saved: fig_closest_habitable_planets.png")
plt.show()

# Summary statistics
print("\n" + "="*80)
print("DISTANCE STATISTICS FOR HABITABLE PLANETS")
print("="*80)
print(f"\nMedian distance: {habitable['distance_ly'].median():.0f} light years")
print(f"Mean distance: {habitable['distance_ly'].mean():.0f} light years")
print(f"Closest: {habitable['distance_ly'].min():.0f} light years ({habitable.loc[habitable['distance_ly'].idxmin(), 'planet_name']})")
print(f"Farthest: {habitable['distance_ly'].max():.0f} light years")

print(f"\nPlanets within:")
print(f"  500 ly: {(habitable['distance_ly'] <= 500).sum()}")
print(f"  1000 ly: {(habitable['distance_ly'] <= 1000).sum()}")
print(f"  1500 ly: {(habitable['distance_ly'] <= 1500).sum()}")
print(f"  2000 ly: {(habitable['distance_ly'] <= 2000).sum()}")

# Travel time at light speed (for perspective)
closest = habitable.loc[habitable['distance_ly'].idxmin()]
print(f"\n" + "="*80)
print("PERSPECTIVE: TRAVEL TIME TO CLOSEST HABITABLE PLANET")
print("="*80)
print(f"\nClosest habitable candidate: {closest['planet_name']}")
print(f"Distance: {closest['distance_ly']:.0f} light years")
print(f"\nTravel time at:")
print(f"  Light speed (impossible): {closest['distance_ly']:.0f} years")
print(f"  Voyager 1 speed (17 km/s): {closest['distance_ly'] * 17647:.0f} years")
print(f"  1% light speed (future tech): {closest['distance_ly'] * 100:.0f} years")
print(f"  10% light speed (far future): {closest['distance_ly'] * 10:.0f} years")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
