#!/usr/bin/env python3
"""
Exoplanet Habitability Analysis Tool
Analyzes exoplanet data to estimate habitability likelihood using scientific metrics.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class HabitabilityAnalyzer:
    """
    Analyzes exoplanet habitability using multiple scientific metrics:
    - Habitable Zone criterion (conservative and optimistic)
    - Earth Similarity Index (ESI)
    - Planetary Habitability Index (PHI)
    - Surface temperature suitability
    - Mass/Radius constraints for rocky planets
    """

    # Constants
    EARTH_RADIUS = 1.0  # Earth radii
    EARTH_MASS = 1.0    # Earth masses
    EARTH_TEMP = 288    # Kelvin
    EARTH_PERIOD = 365.25  # Days

    # Habitability thresholds
    TEMP_MIN = 273      # 0°C - water freezes
    TEMP_MAX = 373      # 100°C - water boils
    TEMP_OPTIMUM = 288  # 15°C - Earth average

    RADIUS_MIN = 0.5    # Minimum radius for rocky planet (Earth radii)
    RADIUS_MAX = 1.6    # Maximum radius for likely rocky planet
    MASS_MIN = 0.1      # Minimum mass (Earth masses)
    MASS_MAX = 5.0      # Maximum mass for super-Earth

    # Habitable zone bounds (AU) - depends on stellar luminosity
    # For Sun-like stars (conservative and optimistic estimates)
    HZ_INNER_CONSERVATIVE = 0.95
    HZ_OUTER_CONSERVATIVE = 1.37
    HZ_INNER_OPTIMISTIC = 0.75
    HZ_OUTER_OPTIMISTIC = 1.77

    def __init__(self, data_file: str):
        """Load and prepare exoplanet data."""
        print(f"Loading data from: {data_file}")

        # Read CSV, skipping comment lines
        with open(data_file, 'r') as f:
            for i, line in enumerate(f):
                if not line.startswith('#'):
                    skip_rows = i
                    break

        self.df = pd.read_csv(data_file, skiprows=skip_rows)
        print(f"Loaded {len(self.df)} exoplanet candidates/confirmations")

        # Map column names (Kepler dataset specific)
        self.col_map = {
            'radius': 'koi_prad',           # Planet radius (Earth radii)
            'period': 'koi_period',          # Orbital period (days)
            'temp': 'koi_teq',              # Equilibrium temperature (K)
            'sma': 'koi_sma',               # Semi-major axis (AU)
            'insol': 'koi_insol',           # Insolation flux (Earth flux)
            'disposition': 'koi_disposition', # CANDIDATE/CONFIRMED/FALSE POSITIVE
            'star_teff': 'koi_steff',       # Stellar effective temperature
            'star_rad': 'koi_srad',         # Stellar radius (solar radii)
            'star_mass': 'koi_smass',       # Stellar mass (solar masses)
        }

        # Calculate derived mass from radius (using mass-radius relationship)
        self._estimate_mass()

    def _estimate_mass(self):
        """
        Estimate planetary mass from radius using empirical mass-radius relationships.
        Uses Weiss & Marcy (2014) and Chen & Kipping (2017) relations.
        """
        R = self.df[self.col_map['radius']].values

        # Mass-radius relationship for different regimes
        M = np.zeros_like(R)

        # Rocky planets (R < 1.5 R_Earth): M ∝ R^3.7
        rocky_mask = (R < 1.5) & (R > 0)
        M[rocky_mask] = R[rocky_mask] ** 3.7

        # Transition planets (1.5 < R < 4): M ∝ R^2.04
        transition_mask = (R >= 1.5) & (R < 4.0)
        M[transition_mask] = 2.7 * (R[transition_mask] ** 2.04)

        # Gas giants (R >= 4): M ∝ R^1.3
        giant_mask = R >= 4.0
        M[giant_mask] = 60 * (R[giant_mask] ** 1.3)

        self.df['estimated_mass'] = M

    def calculate_esi(self) -> np.ndarray:
        """
        Calculate Earth Similarity Index (ESI).
        ESI ranges from 0 (completely dissimilar) to 1 (identical to Earth).

        ESI = [Product of (1 - |x_i - x_Earth|/|x_i + x_Earth|)]^(1/n)

        We use: radius, density (mass/radius proxy), escape velocity, and temperature
        """
        R = self.df[self.col_map['radius']].fillna(0).values
        M = self.df['estimated_mass'].fillna(0).values
        T = self.df[self.col_map['temp']].fillna(0).values

        # Avoid division by zero
        epsilon = 1e-10

        # Component 1: Radius similarity
        esi_r = 1 - np.abs((R - self.EARTH_RADIUS) / (R + self.EARTH_RADIUS + epsilon))

        # Component 2: Density similarity (using mass as proxy)
        esi_m = 1 - np.abs((M - self.EARTH_MASS) / (M + self.EARTH_MASS + epsilon))

        # Component 3: Escape velocity similarity (proportional to sqrt(M/R))
        V_esc = np.sqrt(M / (R + epsilon))
        V_esc_earth = np.sqrt(self.EARTH_MASS / self.EARTH_RADIUS)
        esi_v = 1 - np.abs((V_esc - V_esc_earth) / (V_esc + V_esc_earth + epsilon))

        # Component 4: Temperature similarity
        esi_t = 1 - np.abs((T - self.EARTH_TEMP) / (T + self.EARTH_TEMP + epsilon))

        # Combined ESI (geometric mean)
        esi = (esi_r * esi_m * esi_v * esi_t) ** 0.25

        # Set invalid values to 0
        esi = np.nan_to_num(esi, 0)
        esi = np.clip(esi, 0, 1)

        return esi

    def check_habitable_zone(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Check if planet is in the habitable zone.
        Uses stellar luminosity to scale HZ boundaries.

        Returns:
            conservative_hz: Boolean array for conservative HZ
            optimistic_hz: Boolean array for optimistic HZ
        """
        sma = self.df[self.col_map['sma']].fillna(0).values
        insol = self.df[self.col_map['insol']].fillna(0).values

        # For conservative HZ: 0.32 < S/S_Earth < 1.1 (runaway greenhouse to maximum greenhouse)
        # For optimistic HZ: 0.25 < S/S_Earth < 1.67

        conservative_hz = (insol >= (1.0/self.HZ_OUTER_CONSERVATIVE)) & \
                         (insol <= (1.0/self.HZ_INNER_CONSERVATIVE))

        optimistic_hz = (insol >= (1.0/self.HZ_OUTER_OPTIMISTIC)) & \
                       (insol <= (1.0/self.HZ_INNER_OPTIMISTIC))

        return conservative_hz, optimistic_hz

    def check_temperature(self) -> np.ndarray:
        """
        Score planets based on temperature suitability for liquid water.
        Returns score 0-1.
        """
        T = self.df[self.col_map['temp']].fillna(0).values

        # Gaussian-like scoring centered on optimal temperature
        temp_score = np.exp(-((T - self.TEMP_OPTIMUM) ** 2) / (2 * 50 ** 2))

        # Hard cutoff for temperatures outside water range
        temp_score[(T < self.TEMP_MIN) | (T > self.TEMP_MAX)] = 0

        return temp_score

    def check_size_mass(self) -> np.ndarray:
        """
        Score planets based on size/mass suitability (rocky planet likelihood).
        Returns score 0-1.
        """
        R = self.df[self.col_map['radius']].fillna(0).values
        M = self.df['estimated_mass'].fillna(0).values

        # Optimal range: Earth-like to Super-Earth
        radius_score = np.zeros_like(R)

        # Optimal radius: 0.8 - 1.4 Earth radii (score = 1)
        optimal_mask = (R >= 0.8) & (R <= 1.4)
        radius_score[optimal_mask] = 1.0

        # Acceptable radius: 0.5 - 1.6 Earth radii (linear decay)
        small_mask = (R >= self.RADIUS_MIN) & (R < 0.8)
        radius_score[small_mask] = (R[small_mask] - self.RADIUS_MIN) / (0.8 - self.RADIUS_MIN)

        large_mask = (R > 1.4) & (R <= self.RADIUS_MAX)
        radius_score[large_mask] = (self.RADIUS_MAX - R[large_mask]) / (self.RADIUS_MAX - 1.4)

        # Mass constraint (similar approach)
        mass_score = np.zeros_like(M)

        optimal_mass_mask = (M >= 0.5) & (M <= 2.0)
        mass_score[optimal_mass_mask] = 1.0

        small_mass_mask = (M >= self.MASS_MIN) & (M < 0.5)
        mass_score[small_mass_mask] = (M[small_mass_mask] - self.MASS_MIN) / (0.5 - self.MASS_MIN)

        large_mass_mask = (M > 2.0) & (M <= self.MASS_MAX)
        mass_score[large_mass_mask] = (self.MASS_MAX - M[large_mass_mask]) / (self.MASS_MAX - 2.0)

        # Combined size/mass score
        return (radius_score + mass_score) / 2

    def calculate_habitability_score(self) -> pd.DataFrame:
        """
        Calculate comprehensive habitability score combining all metrics.

        Returns:
            DataFrame with habitability metrics and overall score
        """
        print("\nCalculating habitability metrics...")

        # Calculate individual metrics
        esi = self.calculate_esi()
        conservative_hz, optimistic_hz = self.check_habitable_zone()
        temp_score = self.check_temperature()
        size_mass_score = self.check_size_mass()

        # Overall habitability score (weighted combination)
        # Weights: HZ is most important, then temperature, then ESI, then size
        habitability_score = (
            0.35 * conservative_hz.astype(float) +
            0.25 * temp_score +
            0.25 * esi +
            0.15 * size_mass_score
        )

        # Bonus for confirmed planets
        confirmed_mask = self.df[self.col_map['disposition']] == 'CONFIRMED'
        confidence_multiplier = np.where(confirmed_mask, 1.0, 0.8)

        habitability_score *= confidence_multiplier

        # Create results dataframe
        results = pd.DataFrame({
            'planet_name': self.df.get('kepler_name', self.df.get('kepoi_name', 'Unknown')),
            'disposition': self.df[self.col_map['disposition']],
            'radius_earth': self.df[self.col_map['radius']],
            'estimated_mass_earth': self.df['estimated_mass'],
            'period_days': self.df[self.col_map['period']],
            'temperature_k': self.df[self.col_map['temp']],
            'semi_major_axis_au': self.df[self.col_map['sma']],
            'insolation_flux': self.df[self.col_map['insol']],
            'in_hz_conservative': conservative_hz,
            'in_hz_optimistic': optimistic_hz,
            'esi': esi,
            'temp_score': temp_score,
            'size_mass_score': size_mass_score,
            'habitability_score': habitability_score,
            'habitability_category': self._categorize_habitability(habitability_score)
        })

        return results

    def _categorize_habitability(self, scores: np.ndarray) -> np.ndarray:
        """Categorize planets by habitability score."""
        categories = np.empty(len(scores), dtype=object)
        categories[scores >= 0.7] = 'High'
        categories[(scores >= 0.4) & (scores < 0.7)] = 'Moderate'
        categories[(scores >= 0.2) & (scores < 0.4)] = 'Low'
        categories[scores < 0.2] = 'Very Low'
        return categories

    def get_top_candidates(self, results: pd.DataFrame, n: int = 50) -> pd.DataFrame:
        """Get top N potentially habitable planets."""
        # Filter for reasonable candidates
        filtered = results[
            (results['radius_earth'] > 0) &
            (results['temperature_k'] > 0) &
            (results['habitability_score'] > 0.1)
        ].copy()

        # Sort by habitability score
        top = filtered.nlargest(n, 'habitability_score')

        return top

    def print_summary(self, results: pd.DataFrame):
        """Print summary statistics."""
        print("\n" + "="*80)
        print("HABITABILITY ANALYSIS SUMMARY")
        print("="*80)

        print(f"\nTotal planets analyzed: {len(results)}")
        print(f"Confirmed planets: {(results['disposition'] == 'CONFIRMED').sum()}")
        print(f"Candidate planets: {(results['disposition'] == 'CANDIDATE').sum()}")
        print(f"False positives: {(results['disposition'] == 'FALSE POSITIVE').sum()}")

        print("\n" + "-"*80)
        print("HABITABILITY DISTRIBUTION")
        print("-"*80)
        for category in ['High', 'Moderate', 'Low', 'Very Low']:
            count = (results['habitability_category'] == category).sum()
            pct = 100 * count / len(results)
            print(f"{category:15s}: {count:5d} planets ({pct:5.2f}%)")

        print("\n" + "-"*80)
        print("HABITABLE ZONE STATISTICS")
        print("-"*80)
        hz_conservative = results['in_hz_conservative'].sum()
        hz_optimistic = results['in_hz_optimistic'].sum()
        print(f"In conservative HZ: {hz_conservative:5d} planets ({100*hz_conservative/len(results):5.2f}%)")
        print(f"In optimistic HZ:   {hz_optimistic:5d} planets ({100*hz_optimistic/len(results):5.2f}%)")

        print("\n" + "-"*80)
        print("TOP 10 MOST POTENTIALLY HABITABLE PLANETS")
        print("-"*80)

        top_10 = self.get_top_candidates(results, n=10)

        for idx, (i, row) in enumerate(top_10.iterrows(), 1):
            print(f"\n{idx}. {row['planet_name']}")
            print(f"   Habitability Score: {row['habitability_score']:.3f} ({row['habitability_category']})")
            print(f"   Status: {row['disposition']}")
            print(f"   Radius: {row['radius_earth']:.2f} R_Earth")
            print(f"   Mass (est): {row['estimated_mass_earth']:.2f} M_Earth")
            print(f"   Temperature: {row['temperature_k']:.0f} K ({row['temperature_k']-273:.0f}°C)")
            print(f"   Period: {row['period_days']:.1f} days")
            print(f"   In HZ: {'Yes' if row['in_hz_conservative'] else ('Optimistic' if row['in_hz_optimistic'] else 'No')}")
            print(f"   ESI: {row['esi']:.3f}")


def generate_ml_recommendations():
    """
    Print recommendations for machine learning approaches to habitability prediction.
    """
    print("\n" + "="*80)
    print("MACHINE LEARNING RECOMMENDATIONS FOR HABITABILITY PREDICTION")
    print("="*80)

    recommendations = """
1. SUPERVISED LEARNING APPROACHES

   a) Random Forest Classifier
      - Train on confirmed Earth-like planets vs non-habitable planets
      - Features: radius, mass, temperature, orbital period, stellar properties
      - Handles non-linear relationships and feature interactions well
      - Provides feature importance rankings
      - Can handle missing data reasonably well

   b) Gradient Boosting (XGBoost/LightGBM)
      - Often outperforms Random Forest for structured data
      - Better handling of imbalanced datasets (few habitable planets)
      - Use SMOTE or class weighting to handle class imbalance

   c) Neural Networks (Deep Learning)
      - Multi-layer perceptron for feature learning
      - Can discover complex non-linear patterns
      - Requires more data; consider transfer learning from known systems
      - Use dropout and regularization to prevent overfitting

2. SEMI-SUPERVISED LEARNING

   - Most planets are "candidates" without confirmed habitability
   - Use label propagation or co-training approaches
   - Train on confirmed planets, apply to candidates
   - Active learning: identify most informative candidates for follow-up

3. ENSEMBLE METHODS

   - Combine multiple models (stacking/voting)
   - Example: Random Forest + XGBoost + Neural Network
   - Physics-informed ML: combine data-driven model with physics-based constraints
   - Use this analysis's rule-based scores as one ensemble component

4. FEATURE ENGINEERING

   Key features to include/derive:
   - Planet: radius, mass, density, escape velocity, temperature
   - Orbit: period, semi-major axis, eccentricity, insolation
   - Star: temperature, radius, mass, age, metallicity
   - Derived: ESI, HZ position, tidal locking likelihood
   - Atmospheric proxies: H/He retention likelihood based on mass/radius

5. VALIDATION STRATEGY

   - K-fold cross-validation (k=5 or k=10)
   - Stratified sampling to maintain class balance
   - Leave-one-group-out: test on different stellar systems
   - Compare against known habitable zone metrics as baseline

6. HANDLING CHALLENGES

   a) Class Imbalance
      - Very few truly "habitable" planets in dataset
      - Use SMOTE, ADASYN for synthetic oversampling
      - Cost-sensitive learning with class weights
      - Anomaly detection approach (habitable = rare/anomalous)

   b) Missing Data
      - Multiple imputation methods
      - Model-based imputation (MICE, missForest)
      - Use algorithms robust to missing data (XGBoost, LightGBM)

   c) Uncertainty Quantification
      - Bayesian neural networks for prediction intervals
      - Conformal prediction for confidence regions
      - Monte Carlo dropout for uncertainty estimation

7. RECOMMENDED PYTHON LIBRARIES

   - scikit-learn: Random Forest, SVM, preprocessing
   - xgboost/lightgbm: Gradient boosting
   - tensorflow/pytorch: Deep learning
   - imbalanced-learn: Handling class imbalance
   - gpytorch: Gaussian process models
   - astropy: Astronomical calculations

8. PHYSICS-INFORMED MACHINE LEARNING

   - Constrain ML predictions with known physics
   - Penalty terms for violating habitability constraints
   - Hybrid models: neural network + differential equations
   - Incorporate stellar evolution models

9. INTERPRETABILITY

   - SHAP (SHapley Additive exPlanations) values
   - LIME (Local Interpretable Model-agnostic Explanations)
   - Partial dependence plots
   - Feature importance analysis
   - Critical for scientific validation and discovery

10. RECOMMENDED WORKFLOW

    Step 1: Use this script's physics-based scores as baseline
    Step 2: Split data: 70% train, 15% validation, 15% test
    Step 3: Train multiple models with cross-validation
    Step 4: Hyperparameter tuning (Grid/Random/Bayesian search)
    Step 5: Ensemble best models
    Step 6: Evaluate on test set with multiple metrics
    Step 7: Interpret results with SHAP/LIME
    Step 8: Generate predictions for unlabeled candidates
"""

    print(recommendations)

    print("\n" + "="*80)
    print("EVALUATION METRICS TO USE")
    print("="*80)

    metrics = """
- Precision: Of planets predicted habitable, how many truly are?
- Recall: Of truly habitable planets, how many did we identify?
- F1-Score: Harmonic mean of precision and recall
- ROC-AUC: Trade-off between true positive and false positive rates
- Precision-Recall curve: Better for imbalanced datasets
- Top-K accuracy: Are the true habitable planets in top K predictions?
- Calibration: Are predicted probabilities well-calibrated?
"""
    print(metrics)


def main():
    """Main analysis workflow."""
    import sys

    # Default to cumulative Kepler dataset
    data_file = '/Users/armin/Documents/xplorer/cumulative_2025.10.04_15.04.09.csv'

    if len(sys.argv) > 1:
        data_file = sys.argv[1]

    print("="*80)
    print("EXOPLANET HABITABILITY ANALYSIS")
    print("="*80)
    print(f"\nData source: {data_file}")

    # Initialize analyzer
    analyzer = HabitabilityAnalyzer(data_file)

    # Calculate habitability scores
    results = analyzer.calculate_habitability_score()

    # Print summary
    analyzer.print_summary(results)

    # Save results
    output_file = data_file.replace('.csv', '_habitability_scores.csv')
    results.to_csv(output_file, index=False)
    print(f"\n{'='*80}")
    print(f"Full results saved to: {output_file}")

    # Save top candidates
    top_candidates = analyzer.get_top_candidates(results, n=100)
    top_file = data_file.replace('.csv', '_top_habitable.csv')
    top_candidates.to_csv(top_file, index=False)
    print(f"Top 100 candidates saved to: {top_file}")

    # Print ML recommendations
    generate_ml_recommendations()

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
