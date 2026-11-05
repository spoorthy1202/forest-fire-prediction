import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('outputs/figures', exist_ok=True)
df = pd.read_csv('dataset/Algerian_forest_fires_dataset_CLEANED.csv')
df.columns = df.columns.str.strip()

# Plot 1: Target Distribution
plt.figure()
sns.countplot(data=df, x='Classes')
plt.title('Target Distribution')
plt.savefig('outputs/figures/01_target_distribution.png')
plt.close()

# Plot 2: Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.select_dtypes(include=['float64', 'int64']).corr(), annot=False, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('outputs/figures/02_correlation_heatmap.png')
plt.close()

# Plot 3: Meteorological Distributions (Boxplots)
fig, axes = plt.subplots(1, 4, figsize=(15, 4))
sns.boxplot(y=df['Temperature'], ax=axes[0]).set_title('Temperature')
sns.boxplot(y=df['RH'], ax=axes[1]).set_title('Relative Humidity')
sns.boxplot(y=df['Ws'], ax=axes[2]).set_title('Wind Speed')
sns.boxplot(y=df['Rain'], ax=axes[3]).set_title('Rain')
plt.tight_layout()
plt.savefig('outputs/figures/03_meteorological_distributions.png')
plt.close()

# Plot 4: FWI Indices Density (KDE)
plt.figure(figsize=(10,6))
for col in ['FFMC', 'DMC', 'DC', 'ISI', 'BUI']:
    sns.kdeplot(df[col], label=col)
plt.title('Density Distribution of FWI Indices')
plt.legend()
plt.savefig('outputs/figures/04_fwi_density_distributions.png')
plt.close()

# Plot 5: Monthly Fire Trends
plt.figure()
if 'month' in df.columns:
    sns.countplot(data=df, x='month', hue='Classes')
    plt.title('Monthly Fire Incidence')
    plt.savefig('outputs/figures/05_monthly_fire_trends.png')
plt.close()

# Plot 6: Temp vs RH Interaction
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='Temperature', y='RH', hue='Classes')
plt.title('Temperature vs RH by Fire Class')
plt.savefig('outputs/figures/06_temp_rh_interaction.png')
plt.close()

print("6 EDA plots saved to outputs/figures/")