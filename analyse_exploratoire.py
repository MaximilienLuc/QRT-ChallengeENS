"""
Analyse Exploratoire des Données
Challenge: Prédiction des Performances d'Allocations d'Actifs
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuration des graphiques
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 10

print("="*80)
print("ANALYSE EXPLORATOIRE - CHALLENGE ALLOCATION D'ACTIFS")
print("="*80)

# ============================================================================
# 1. CHARGEMENT DES DONNÉES
# ============================================================================
print("\n" + "="*80)
print("1. CHARGEMENT DES DONNÉES")
print("="*80)

X_train = pd.read_csv('Data/X_train.csv')
y_train = pd.read_csv('Data/y_train.csv')

print(f"\n✓ X_train chargé: {X_train.shape[0]:,} lignes × {X_train.shape[1]} colonnes")
print(f"✓ y_train chargé: {y_train.shape[0]:,} lignes × {y_train.shape[1]} colonnes")

# ============================================================================
# 2. APERÇU GÉNÉRAL DES DONNÉES
# ============================================================================
print("\n" + "="*80)
print("2. APERÇU GÉNÉRAL")
print("="*80)

print("\n--- Premières lignes de X_train ---")
print(X_train.head())

print("\n--- Premières lignes de y_train ---")
print(y_train.head())

print("\n--- Types de données X_train ---")
print(X_train.dtypes.value_counts())

print("\n--- Informations sur X_train ---")
print(X_train.info())

# ============================================================================
# 3. STRUCTURE DES DONNÉES
# ============================================================================
print("\n" + "="*80)
print("3. STRUCTURE DES DONNÉES")
print("="*80)

# Identifier les colonnes par type
ret_cols = [col for col in X_train.columns if col.startswith('RET_')]
volume_cols = [col for col in X_train.columns if col.startswith('SIGNED_VOLUME_')]

print(f"\n✓ Colonnes de rendements (RET_*): {len(ret_cols)}")
print(f"✓ Colonnes de volumes signés (SIGNED_VOLUME_*): {len(volume_cols)}")
print(f"✓ Autres colonnes: {[col for col in X_train.columns if col not in ret_cols + volume_cols]}")

# Vérifier la cohérence des indices
print("\n--- Cohérence des indices ---")
print(f"X_train ROW_ID min: {X_train['ROW_ID'].min()}, max: {X_train['ROW_ID'].max()}")
print(f"y_train ROW_ID min: {y_train['ROW_ID'].min()}, max: {y_train['ROW_ID'].max()}")
print(f"ROW_ID identiques: {X_train['ROW_ID'].equals(y_train['ROW_ID'])}")

# Merger les datasets
df = X_train.merge(y_train, on='ROW_ID', how='inner')
print(f"\nDataset fusionné: {df.shape[0]:,} lignes × {df.shape[1]} colonnes")

# ============================================================================
# 4. ANALYSE DES VALEURS MANQUANTES
# ============================================================================
print("\n" + "="*80)
print("4. ANALYSE DES VALEURS MANQUANTES")
print("="*80)

missing = df.isnull().sum()
missing_pct = 100 * missing / len(df)
missing_df = pd.DataFrame({
    'Colonnes': missing.index,
    'Valeurs Manquantes': missing.values,
    'Pourcentage': missing_pct.values
}).sort_values('Valeurs Manquantes', ascending=False)

print("\n--- Top 10 colonnes avec valeurs manquantes ---")
print(missing_df[missing_df['Valeurs Manquantes'] > 0].head(10))

total_missing = missing.sum()
print(f"\n✓ Total de valeurs manquantes: {total_missing:,} ({100*total_missing/(df.shape[0]*df.shape[1]):.2f}%)")

# ============================================================================
# 5. STATISTIQUES DESCRIPTIVES
# ============================================================================
print("\n" + "="*80)
print("5. STATISTIQUES DESCRIPTIVES")
print("="*80)

print("\n--- Statistiques des rendements (RET_*) ---")
print(df[ret_cols].describe())

print("\n--- Statistiques des volumes signés (SIGNED_VOLUME_*) ---")
print(df[volume_cols].describe())

print("\n--- Statistiques MEDIAN_DAILY_TURNOVER ---")
if 'MEDIAN_DAILY_TURNOVER' in df.columns:
    print(df['MEDIAN_DAILY_TURNOVER'].describe())

# ============================================================================
# 6. ANALYSE DE LA VARIABLE CIBLE (TARGET)
# ============================================================================
print("\n" + "="*80)
print("6. ANALYSE DE LA VARIABLE CIBLE (TARGET)")
print("="*80)

print("\n--- Statistiques de target ---")
print(df['target'].describe())

print("\n--- Distribution du signe de target ---")
target_positive = (df['target'] > 0).sum()
target_negative = (df['target'] <= 0).sum()
print(f"target > 0:  {target_positive:,} ({100*target_positive/len(df):.2f}%)")
print(f"target <= 0: {target_negative:,} ({100*target_negative/len(df):.2f}%)")

# Test de normalité
_, p_value = stats.normaltest(df['target'].dropna())
print(f"\nTest de normalité (p-value): {p_value:.6f}")
print(f"Distribution {'normale' if p_value > 0.05 else 'non-normale'} (seuil α=0.05)")

# ============================================================================
# 7. ANALYSE TEMPORELLE
# ============================================================================
print("\n" + "="*80)
print("7. ANALYSE TEMPORELLE")
print("="*80)

if 'TS' in df.columns:
    print(f"\n✓ Nombre de timestamps uniques: {df['TS'].nunique()}")
    print(f"✓ Timestamps: de {df['TS'].min()} à {df['TS'].max()}")
    
    print("\n--- Distribution des observations par timestamp ---")
    ts_counts = df['TS'].value_counts().sort_index()
    print(ts_counts.describe())

# ============================================================================
# 8. ANALYSE PAR ALLOCATION
# ============================================================================
print("\n" + "="*80)
print("8. ANALYSE PAR ALLOCATION")
print("="*80)

if 'ALLOCATION' in df.columns:
    print(f"\n✓ Nombre d'allocations uniques: {df['ALLOCATION'].nunique()}")
    
    print("\n--- Distribution des observations par allocation ---")
    alloc_counts = df['ALLOCATION'].value_counts()
    print(alloc_counts.describe())
    
    print("\n--- Top 10 allocations les plus fréquentes ---")
    print(alloc_counts.head(10))

# ============================================================================
# 9. ANALYSE DES GROUPES
# ============================================================================
print("\n" + "="*80)
print("9. ANALYSE DES GROUPES")
print("="*80)

if 'GROUP' in df.columns:
    print(f"\n✓ Nombre de groupes uniques: {df['GROUP'].nunique()}")
    
    print("\n--- Distribution par groupe ---")
    group_counts = df['GROUP'].value_counts()
    print(group_counts)
    
    print("\n--- Statistiques TARGET par groupe ---")
    print(df.groupby('GROUP')['target'].agg(['count', 'mean', 'std', 'min', 'max']))
    
    print("\n--- Proportion de TARGET positifs par groupe ---")
    print(df.groupby('GROUP').apply(lambda x: (x['target'] > 0).sum() / len(x) * 100))

# ============================================================================
# 10. ANALYSE DES CORRÉLATIONS
# ============================================================================
print("\n" + "="*80)
print("10. ANALYSE DES CORRÉLATIONS")
print("="*80)

# Corrélation entre rendements historiques et TARGET
print("\n--- Corrélation RET_* avec TARGET ---")
ret_corr = df[ret_cols + ['target']].corr()['target'].drop('target').sort_values(ascending=False)
print(ret_corr)

# Corrélation entre volumes signés et TARGET
print("\n--- Corrélation SIGNED_VOLUME_* avec TARGET ---")
vol_corr = df[volume_cols + ['target']].corr()['target'].drop('target').sort_values(ascending=False)
print(vol_corr)

# Corrélation MEDIAN_DAILY_TURNOVER avec TARGET
if 'MEDIAN_DAILY_TURNOVER' in df.columns:
    turnover_corr = df[['MEDIAN_DAILY_TURNOVER', 'target']].corr().iloc[0, 1]
    print(f"\n--- Corrélation MEDIAN_DAILY_TURNOVER avec TARGET: {turnover_corr:.4f} ---")

# ============================================================================
# 11. PATTERNS ET INSIGHTS
# ============================================================================
print("\n" + "="*80)
print("11. PATTERNS ET INSIGHTS CLÉS")
print("="*80)

# Analyse de momentum
if len(ret_cols) >= 3:
    df['momentum_recent'] = df[[ret_cols[-1], ret_cols[-2], ret_cols[-3]]].mean(axis=1)
    print("\n--- Momentum récent (moyenne RET des 3 derniers jours) vs TARGET ---")
    print(f"Corrélation: {df[['momentum_recent', 'target']].corr().iloc[0, 1]:.4f}")
    
    momentum_positive = df[df['momentum_recent'] > 0]['target'].mean()
    momentum_negative = df[df['momentum_recent'] <= 0]['target'].mean()
    print(f"TARGET moyen si momentum > 0: {momentum_positive:.6f}")
    print(f"TARGET moyen si momentum <= 0: {momentum_negative:.6f}")

# Volatilité des rendements
df['volatility'] = df[ret_cols].std(axis=1)
print("\n--- Volatilité des rendements historiques ---")
print(f"Corrélation volatilité vs TARGET: {df[['volatility', 'target']].corr().iloc[0, 1]:.4f}")

# Persistence du signe
if len(ret_cols) >= 1:
    df['last_ret_positive'] = (df[ret_cols[-1]] > 0).astype(int)
    print("\n--- Persistance du signe ---")
    print("TARGET moyen selon le signe du dernier rendement:")
    print(df.groupby('last_ret_positive')['target'].agg(['count', 'mean']))

# ============================================================================
# 12. VISUALISATIONS
# ============================================================================
print("\n" + "="*80)
print("12. GÉNÉRATION DES VISUALISATIONS")
print("="*80)

# Figure 1: Distribution de TARGET
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].hist(df['target'], bins=100, edgecolor='black', alpha=0.7)
axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zéro')
axes[0].set_xlabel('target')
axes[0].set_ylabel('Fréquence')
axes[0].set_title('Distribution de TARGET')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].hist(df['target'], bins=100, edgecolor='black', alpha=0.7, log=True)
axes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='Zéro')
axes[1].set_xlabel('target')
axes[1].set_ylabel('Fréquence (échelle log)')
axes[1].set_title('Distribution de TARGET (échelle log)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

boxplot_data = [df[df['target'] > 0]['target'], df[df['target'] <= 0]['target']]
axes[2].boxplot(boxplot_data, labels=['TARGET > 0', 'TARGET ≤ 0'])
axes[2].set_ylabel('target')
axes[2].set_title('Boxplot de TARGET par signe')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/fig1_distribution_target.png', dpi=150, bbox_inches='tight')
print("✓ Figure 1 sauvegardée: distribution de TARGET")
plt.close()

# Figure 2: Évolution temporelle
if 'TS' in df.columns:
    fig, axes = plt.subplots(2, 1, figsize=(16, 8))
    
    ts_stats = df.groupby('TS')['target'].agg(['mean', 'std', 'count'])
    
    axes[0].plot(ts_stats.index, ts_stats['mean'], marker='o', markersize=3, linewidth=1)
    axes[0].axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.7)
    axes[0].set_xlabel('Timestamp')
    axes[0].set_ylabel('TARGET moyen')
    axes[0].set_title('Évolution temporelle du TARGET moyen')
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='x', rotation=45)
    
    axes[1].bar(ts_stats.index, ts_stats['count'], alpha=0.7)
    axes[1].set_xlabel('Timestamp')
    axes[1].set_ylabel('Nombre d\'observations')
    axes[1].set_title('Nombre d\'observations par timestamp')
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('/home/claude/fig2_evolution_temporelle.png', dpi=150, bbox_inches='tight')
    print("✓ Figure 2 sauvegardée: évolution temporelle")
    plt.close()

# Figure 3: Heatmap des corrélations des rendements
fig, ax = plt.subplots(figsize=(14, 10))
corr_matrix = df[ret_cols + ['target']].corr()
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Matrice de corrélation des rendements historiques et TARGET')
plt.tight_layout()
plt.savefig('/home/claude/fig3_correlation_rendements.png', dpi=150, bbox_inches='tight')
print("✓ Figure 3 sauvegardée: corrélations des rendements")
plt.close()

# Figure 4: Corrélations avec TARGET
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].bar(range(len(ret_corr)), ret_corr.values, alpha=0.7)
axes[0].axhline(0, color='red', linestyle='--', linewidth=1)
axes[0].set_xlabel('Rendement historique (jour)')
axes[0].set_ylabel('Corrélation avec TARGET')
axes[0].set_title('Corrélation RET_* avec TARGET')
axes[0].set_xticks(range(len(ret_corr)))
axes[0].set_xticklabels([col.replace('RET_', '') for col in ret_corr.index], rotation=45)
axes[0].grid(True, alpha=0.3)

axes[1].bar(range(len(vol_corr)), vol_corr.values, alpha=0.7, color='orange')
axes[1].axhline(0, color='red', linestyle='--', linewidth=1)
axes[1].set_xlabel('Volume signé (jour)')
axes[1].set_ylabel('Corrélation avec TARGET')
axes[1].set_title('Corrélation SIGNED_VOLUME_* avec TARGET')
axes[1].set_xticks(range(len(vol_corr)))
axes[1].set_xticklabels([col.replace('SIGNED_VOLUME_', '') for col in vol_corr.index], rotation=45)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/fig4_correlations_target.png', dpi=150, bbox_inches='tight')
print("✓ Figure 4 sauvegardée: corrélations avec TARGET")
plt.close()

# Figure 5: Analyse par groupe
if 'GROUP' in df.columns:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    group_stats = df.groupby('GROUP')['target'].agg(['mean', 'std', 'count'])
    
    axes[0, 0].bar(group_stats.index, group_stats['mean'], alpha=0.7)
    axes[0, 0].axhline(0, color='red', linestyle='--', linewidth=1)
    axes[0, 0].set_xlabel('Groupe')
    axes[0, 0].set_ylabel('TARGET moyen')
    axes[0, 0].set_title('TARGET moyen par groupe')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].bar(group_stats.index, group_stats['count'], alpha=0.7, color='green')
    axes[0, 1].set_xlabel('Groupe')
    axes[0, 1].set_ylabel('Nombre d\'observations')
    axes[0, 1].set_title('Nombre d\'observations par groupe')
    axes[0, 1].grid(True, alpha=0.3)
    
    for i, group in enumerate(df['GROUP'].unique()):
        group_data = df[df['GROUP'] == group]['target']
        axes[1, 0].hist(group_data, bins=50, alpha=0.5, label=f'Groupe {group}')
    axes[1, 0].set_xlabel('target')
    axes[1, 0].set_ylabel('Fréquence')
    axes[1, 0].set_title('Distribution de TARGET par groupe')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    group_positive_pct = df.groupby('GROUP').apply(lambda x: (x['target'] > 0).sum() / len(x) * 100)
    axes[1, 1].bar(group_positive_pct.index, group_positive_pct.values, alpha=0.7, color='purple')
    axes[1, 1].axhline(50, color='red', linestyle='--', linewidth=1, label='50%')
    axes[1, 1].set_xlabel('Groupe')
    axes[1, 1].set_ylabel('% TARGET > 0')
    axes[1, 1].set_title('Pourcentage de TARGET positifs par groupe')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/claude/fig5_analyse_groupes.png', dpi=150, bbox_inches='tight')
    print("✓ Figure 5 sauvegardée: analyse par groupe")
    plt.close()

# Figure 6: Volatilité et momentum
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Scatter plot volatilité vs TARGET
sample_size = min(5000, len(df))
sample_idx = np.random.choice(len(df), sample_size, replace=False)
axes[0].scatter(df.iloc[sample_idx]['volatility'], df.iloc[sample_idx]['target'], 
                alpha=0.3, s=10)
axes[0].set_xlabel('Volatilité des rendements historiques')
axes[0].set_ylabel('target')
axes[0].set_title(f'Volatilité vs TARGET (échantillon de {sample_size:,} points)')
axes[0].grid(True, alpha=0.3)

# Boxplot momentum vs TARGET sign
if 'momentum_recent' in df.columns:
    df['target_sign'] = (df['target'] > 0).map({True: 'Positif', False: 'Négatif ou nul'})
    df.boxplot(column='momentum_recent', by='target_sign', ax=axes[1])
    axes[1].set_xlabel('Signe de TARGET')
    axes[1].set_ylabel('Momentum récent')
    axes[1].set_title('Momentum récent selon le signe de TARGET')
    axes[1].get_figure().suptitle('')
    axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/fig6_volatilite_momentum.png', dpi=150, bbox_inches='tight')
print("✓ Figure 6 sauvegardée: volatilité et momentum")
plt.close()

# ============================================================================
# 13. RÉSUMÉ ET RECOMMANDATIONS
# ============================================================================
print("\n" + "="*80)
print("13. RÉSUMÉ ET RECOMMANDATIONS")
print("="*80)

print("\n📊 RÉSUMÉ DES INSIGHTS CLÉS:")
print("-" * 80)

print(f"\n1. VOLUME DES DONNÉES")
print(f"   • {df.shape[0]:,} observations au total")
print(f"   • {df['TS'].nunique() if 'TS' in df.columns else 'N/A'} timestamps uniques")
print(f"   • {df['ALLOCATION'].nunique() if 'ALLOCATION' in df.columns else 'N/A'} allocations uniques")

print(f"\n2. VARIABLE CIBLE (TARGET)")
print(f"   • Moyenne: {df['target'].mean():.6f}")
print(f"   • Écart-type: {df['target'].std():.6f}")
print(f"   • Médiane: {df['target'].median():.6f}")
print(f"   • Balance positive/négative: {100*target_positive/len(df):.2f}% / {100*target_negative/len(df):.2f}%")

print(f"\n3. FEATURES IMPORTANTES")
print(f"   • {len(ret_cols)} rendements historiques (RET_*)")
print(f"   • {len(volume_cols)} volumes signés (SIGNED_VOLUME_*)")
print(f"   • Turnover médian disponible: {'Oui' if 'MEDIAN_DAILY_TURNOVER' in df.columns else 'Non'}")

print(f"\n4. CORRÉLATIONS AVEC TARGET")
print(f"   • Plus forte corrélation RET: {ret_corr.abs().max():.4f} (RET_{ret_corr.abs().idxmax().replace('RET_', '')})")
print(f"   • Plus forte corrélation VOLUME: {vol_corr.abs().max():.4f} (SIGNED_VOLUME_{vol_corr.abs().idxmax().replace('SIGNED_VOLUME_', '')})")

print("\n💡 RECOMMANDATIONS POUR LA MODÉLISATION:")
print("-" * 80)
print("""
1. FEATURES ENGINEERING
   ✓ Créer des features de momentum (moyennes mobiles des RET)
   ✓ Calculer la volatilité sur différentes fenêtres
   ✓ Encoder les interactions entre rendements et volumes
   ✓ Features temporelles (jour de la semaine, tendance temporelle)
   ✓ Features d'allocation (statistiques par ALLOCATION)

2. GESTION DU DÉSÉQUILIBRE
   ✓ La distribution TARGET semble relativement équilibrée
   ✓ Vérifier si le déséquilibre varie par groupe ou timestamp
   ✓ Considérer des techniques de pondération si nécessaire

3. VALIDATION CROISÉE
   ✓ Utiliser une validation temporelle (TimeSeriesSplit) 
   ✓ Respecter l'ordre chronologique des données
   ✓ Éviter le data leakage entre train et validation

4. MODÈLES À TESTER
   ✓ Baseline: Logistic Regression, Random Forest
   ✓ Gradient Boosting: XGBoost, LightGBM, CatBoost
   ✓ Deep Learning: LSTM, Transformer si pattern temporel fort
   ✓ Ensemble de modèles pour robustesse

5. MÉTRIQUES D'ÉVALUATION
   ✓ Objectif: Accuracy (prédiction du signe)
   ✓ Suivre aussi: Precision, Recall, F1-score par classe
   ✓ ROC-AUC pour évaluer la qualité des probabilités

6. ANALYSE APPROFONDIE À FAIRE
   ✓ Analyser les patterns temporels plus en détail
   ✓ Étudier les différences entre groupes
   ✓ Identifier les allocations difficiles à prédire
   ✓ Détecter les outliers et leur impact
""")

print("\n" + "="*80)
print("FIN DE L'ANALYSE EXPLORATOIRE")
print("="*80)
print("\n✓ Toutes les visualisations ont été sauvegardées dans le répertoire courant")
print("✓ Fichiers générés:")
print("   - fig1_distribution_target.png")
print("   - fig2_evolution_temporelle.png")
print("   - fig3_correlation_rendements.png")
print("   - fig4_correlations_target.png")
print("   - fig5_analyse_groupes.png")
print("   - fig6_volatilite_momentum.png")
