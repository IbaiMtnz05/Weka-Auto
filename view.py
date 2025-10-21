#!/usr/bin/env python3
"""
Script to visualize and analyze ARFF file classification results
Generates statistics, heatmaps, graphs and detailed reports
Supports multiple database analysis (DB1, DB2, DB3, etc.)
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
import warnings
import argparse
warnings.filterwarnings('ignore')

# Configurar matplotlib para mejor visualización
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
sns.set_palette("husl")


class ResultsAnalyzer:
    """Clase para analizar y visualizar resultados de clasificación"""
    
    def __init__(self, csv_file: str = "classification_results.csv"):
        """
        Inicializar el analizador
        
        Args:
            csv_file: Ruta al archivo CSV con resultados
        """
        self.csv_file = csv_file
        self.df = None
        
        # Extraer número de DB del nombre del archivo para el directorio de salida
        if "classification_results_DB" in csv_file:
            db_part = csv_file.replace("classification_results_", "").replace(".csv", "")
            self.output_dir = f"analysis_output_{db_part}"
        else:
            self.output_dir = "analysis_output"
        
        # Crear directorio de salida
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Cargar datos
        self.load_data()
    
    def load_data(self):
        """Cargar datos del archivo CSV"""
        try:
            if not os.path.exists(self.csv_file):
                print(f"Error: El archivo {self.csv_file} no existe.")
                print("Ejecuta primero auto.py para generar los resultados.")
                sys.exit(1)
            
            self.df = pd.read_csv(self.csv_file)
            print(f"Datos cargados: {len(self.df)} resultados de {self.csv_file}")
            
            # Verificar columnas requeridas
            required_cols = ['Dataset', 'Classifier', 'Accuracy']
            missing_cols = [col for col in required_cols if col not in self.df.columns]
            if missing_cols:
                print(f"Error: Faltan columnas requeridas: {missing_cols}")
                sys.exit(1)
                
        except Exception as e:
            print(f"Error cargando {self.csv_file}: {e}")
            sys.exit(1)
    
    def print_basic_statistics(self):
        """Imprimir estadísticas básicas"""
        print("\n" + "="*80)
        print("BASIC STATISTICS")
        print("="*80)
        
        print(f"Total experiments: {len(self.df)}")
        print(f"Datasets evaluated: {self.df['Dataset'].nunique()}")
        print(f"Classifiers tested: {self.df['Classifier'].nunique()}")
        
        if 'Accuracy' in self.df.columns:
            print(f"\nAccuracy:")
            print(f"  Average: {self.df['Accuracy'].mean():.2f}%")
            print(f"  Median: {self.df['Accuracy'].median():.2f}%")
            print(f"  Standard deviation: {self.df['Accuracy'].std():.2f}%")
            print(f"  Range: {self.df['Accuracy'].min():.2f}% - {self.df['Accuracy'].max():.2f}%")
        
        print(f"\nDatasets:")
        for dataset in sorted(self.df['Dataset'].unique()):
            count = len(self.df[self.df['Dataset'] == dataset])
            print(f"  {dataset}: {count} experiments")
        
        print(f"\nClassifiers:")
        for classifier in sorted(self.df['Classifier'].unique()):
            count = len(self.df[self.df['Classifier'] == classifier])
            avg_acc = self.df[self.df['Classifier'] == classifier]['Accuracy'].mean()
            print(f"  {classifier}: {count} experiments (Avg accuracy: {avg_acc:.2f}%)")
    
    def create_accuracy_heatmap(self):
        """Crear heatmap de accuracy por dataset y clasificador"""
        print("\nGenerando heatmap de accuracy...")
        
        # Crear pivot table
        pivot_data = self.df.pivot(index='Dataset', columns='Classifier', values='Accuracy')
        
        # Crear figura
        plt.figure(figsize=(16, 10))
        
        # Crear heatmap
        sns.heatmap(pivot_data, 
                   annot=True, 
                   fmt='.1f', 
                   cmap='coolwarm',
                   center=pivot_data.mean().mean(),
                   cbar_kws={'label': 'Accuracy (%)'},
                   linewidths=0.5)
        
        plt.title('Heatmap de Accuracy por Dataset y Clasificador', fontsize=16, fontweight='bold')
        plt.xlabel('Clasificador', fontsize=12, fontweight='bold')
        plt.ylabel('Dataset', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Guardar
        output_path = os.path.join(self.output_dir, 'accuracy_heatmap.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Heatmap guardado en: {output_path}")
    
    def create_performance_comparison(self):
        """Crear gráfico de comparación de rendimiento por clasificador"""
        print("\nGenerando gráfico de comparación de rendimiento...")
        
        # Preparar datos
        classifier_stats = self.df.groupby('Classifier')['Accuracy'].agg(['mean', 'std', 'count']).reset_index()
        classifier_stats = classifier_stats.sort_values('mean', ascending=False)
        
        # Crear figura con un solo gráfico
        plt.figure(figsize=(16, 10))
        
        # Gráfico: Accuracy promedio con barras de error
        bars = plt.bar(range(len(classifier_stats)), classifier_stats['mean'], 
                      yerr=classifier_stats['std'], capsize=5, 
                      color=sns.color_palette("husl", len(classifier_stats)))
        
        plt.xlabel('Clasificador', fontsize=12, fontweight='bold')
        plt.ylabel('Accuracy Promedio (%)', fontsize=12, fontweight='bold')
        plt.title('Accuracy Promedio por Clasificador', fontsize=16, fontweight='bold')
        plt.xticks(range(len(classifier_stats)), classifier_stats['Classifier'], rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Añadir valores en las barras
        for i, (bar, val) in enumerate(zip(bars, classifier_stats['mean'])):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Guardar
        output_path = os.path.join(self.output_dir, 'performance_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Gráfico de comparación guardado en: {output_path}")
    
    def create_dataset_analysis(self):
        """Análisis de rendimiento por dataset"""
        print("\nGenerando análisis por dataset...")
        
        # Estadísticas por dataset
        dataset_stats = self.df.groupby('Dataset')['Accuracy'].agg(['mean', 'std', 'min', 'max']).reset_index()
        dataset_stats = dataset_stats.sort_values('mean', ascending=False)
        
        # Crear figura
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        
        # Gráfico 1: Accuracy promedio por dataset
        bars = ax1.bar(range(len(dataset_stats)), dataset_stats['mean'], 
                      color=sns.color_palette("viridis", len(dataset_stats)))
        
        ax1.set_xlabel('Dataset', fontweight='bold')
        ax1.set_ylabel('Accuracy Promedio (%)', fontweight='bold')
        ax1.set_title('Accuracy Promedio por Dataset', fontweight='bold')
        ax1.set_xticks(range(len(dataset_stats)))
        ax1.set_xticklabels(dataset_stats['Dataset'], rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)
        
        # Añadir valores
        for i, (bar, val) in enumerate(zip(bars, dataset_stats['mean'])):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Gráfico 2: Rango de accuracy por dataset (min-max)
        datasets = dataset_stats['Dataset']
        mins = dataset_stats['min']
        maxs = dataset_stats['max']
        ranges = maxs - mins
        
        ax2.bar(range(len(dataset_stats)), ranges, bottom=mins,
               color=sns.color_palette("plasma", len(dataset_stats)), alpha=0.7)
        
        ax2.set_xlabel('Dataset', fontweight='bold')
        ax2.set_ylabel('Accuracy (%)', fontweight='bold')
        ax2.set_title('Rango de Accuracy por Dataset (Min-Max)', fontweight='bold')
        ax2.set_xticks(range(len(dataset_stats)))
        ax2.set_xticklabels(datasets, rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Guardar
        output_path = os.path.join(self.output_dir, 'dataset_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Análisis por dataset guardado en: {output_path}")
    
    def create_knn_comparison(self):
        """Análisis específico de clasificadores K-NN"""
        print("\nGenerando análisis específico de K-NN...")
        
        # Filtrar solo K-NN
        knn_df = self.df[self.df['Classifier'].str.contains('NN', case=False)]
        
        if len(knn_df) == 0:
            print("No se encontraron clasificadores K-NN")
            return
        
        # Separar por tipo (Simple vs Weighted)
        knn_df['K_Value'] = knn_df['Classifier'].str.extract(r'(\d+)-NN')[0].astype(int)
        knn_df['Vote_Type'] = knn_df['Classifier'].str.extract(r'NN (Simple|Weighted)')[0]
        
        # Crear pivot para comparación
        knn_pivot = knn_df.pivot_table(index='Dataset', columns=['K_Value', 'Vote_Type'], 
                                      values='Accuracy', aggfunc='first')
        
        # Crear figura
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Gráfico 1: Comparación Simple vs Weighted por K
        k_values = sorted(knn_df['K_Value'].unique())
        simple_means = [knn_df[(knn_df['K_Value'] == k) & (knn_df['Vote_Type'] == 'Simple')]['Accuracy'].mean() 
                       for k in k_values]
        weighted_means = [knn_df[(knn_df['K_Value'] == k) & (knn_df['Vote_Type'] == 'Weighted')]['Accuracy'].mean() 
                         for k in k_values]
        
        x = np.arange(len(k_values))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, simple_means, width, label='Simple Vote', alpha=0.8)
        bars2 = ax1.bar(x + width/2, weighted_means, width, label='Weighted Vote', alpha=0.8)
        
        ax1.set_xlabel('Valor de K', fontweight='bold')
        ax1.set_ylabel('Accuracy Promedio (%)', fontweight='bold')
        ax1.set_title('K-NN: Simple vs Weighted Vote', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'{k}-NN' for k in k_values])
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Añadir valores
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.1f}%', ha='center', va='bottom')
        
        # Gráfico 2: Heatmap específico de K-NN
        if not knn_pivot.empty:
            sns.heatmap(knn_pivot, annot=True, fmt='.1f', cmap='coolwarm', ax=ax2,
                       cbar_kws={'label': 'Accuracy (%)'})
            ax2.set_title('Heatmap K-NN por Dataset', fontweight='bold')
            ax2.set_xlabel('(K, Vote Type)', fontweight='bold')
            ax2.set_ylabel('Dataset', fontweight='bold')
        
        plt.tight_layout()
        
        # Guardar
        output_path = os.path.join(self.output_dir, 'knn_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Análisis K-NN guardado en: {output_path}")
    
    def create_time_analysis(self):
        """Análisis de tiempos de ejecución"""
        if 'Time(s)' not in self.df.columns:
            print("No hay datos de tiempo disponibles")
            return
        
        print("\nGenerando análisis de tiempos de ejecución...")
        
        # Estadísticas de tiempo por clasificador
        time_stats = self.df.groupby('Classifier')['Time(s)'].agg(['mean', 'std']).reset_index()
        time_stats = time_stats.sort_values('mean', ascending=True)
        
        # Crear figura
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        
        # Gráfico 1: Tiempo promedio por clasificador
        bars = ax1.barh(range(len(time_stats)), time_stats['mean'], 
                       xerr=time_stats['std'], capsize=3,
                       color=sns.color_palette("rocket", len(time_stats)))
        
        ax1.set_ylabel('Clasificador', fontweight='bold')
        ax1.set_xlabel('Tiempo Promedio (segundos)', fontweight='bold')
        ax1.set_title('Tiempo de Ejecución Promedio por Clasificador', fontweight='bold')
        ax1.set_yticks(range(len(time_stats)))
        ax1.set_yticklabels(time_stats['Classifier'])
        ax1.grid(axis='x', alpha=0.3)
        
        # Añadir valores
        for i, (bar, val) in enumerate(zip(bars, time_stats['mean'])):
            ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                    f'{val:.2f}s', ha='left', va='center', fontweight='bold')
        
        # Gráfico 2: Scatter plot Accuracy vs Time
        ax2.scatter(self.df['Time(s)'], self.df['Accuracy'], 
                   c=pd.Categorical(self.df['Classifier']).codes, 
                   cmap='tab20', alpha=0.7, s=50)
        
        ax2.set_xlabel('Tiempo de Ejecución (segundos)', fontweight='bold')
        ax2.set_ylabel('Accuracy (%)', fontweight='bold')
        ax2.set_title('Accuracy vs Tiempo de Ejecución', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Añadir línea de tendencia
        z = np.polyfit(self.df['Time(s)'], self.df['Accuracy'], 1)
        p = np.poly1d(z)
        ax2.plot(self.df['Time(s)'], p(self.df['Time(s)']), "r--", alpha=0.8)
        
        plt.tight_layout()
        
        # Guardar
        output_path = os.path.join(self.output_dir, 'time_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Análisis de tiempos guardado en: {output_path}")
    
    def generate_ranking_tables(self):
        """Generar tablas de ranking"""
        print("\nGenerando tablas de ranking...")
        
        # Ranking por accuracy promedio
        classifier_ranking = self.df.groupby('Classifier').agg({
            'Accuracy': ['mean', 'std', 'count'],
            'Time(s)': 'mean' if 'Time(s)' in self.df.columns else lambda x: 0
        }).round(2)
        
        classifier_ranking.columns = ['Accuracy_Mean', 'Accuracy_Std', 'Count', 'Time_Mean']
        classifier_ranking = classifier_ranking.sort_values('Accuracy_Mean', ascending=False).reset_index()
        classifier_ranking['Rank'] = range(1, len(classifier_ranking) + 1)
        
        print("\n" + "="*80)
        print("RANKING DE CLASIFICADORES POR ACCURACY PROMEDIO")
        print("="*80)
        print(classifier_ranking[['Rank', 'Classifier', 'Accuracy_Mean', 'Accuracy_Std', 'Time_Mean']].to_string(index=False))
        
        # Mejor clasificador por dataset
        best_by_dataset = self.df.loc[self.df.groupby('Dataset')['Accuracy'].idxmax()]
        best_summary = best_by_dataset[['Dataset', 'Classifier', 'Accuracy', 'F1-Score']].sort_values('Dataset')
        
        print(f"\n{'='*80}")
        print("MEJOR CLASIFICADOR POR DATASET")
        print("="*80)
        print(best_summary.to_string(index=False))
        
        # Guardar rankings en CSV
        classifier_ranking.to_csv(os.path.join(self.output_dir, 'classifier_ranking.csv'), index=False)
        best_summary.to_csv(os.path.join(self.output_dir, 'best_by_dataset.csv'), index=False)
        
        # Generar CSV ordenado con mejores accuracies por dataset
        self.generate_best_accuracy_csv()
        
        print(f"\nRankings guardados en:")
        print(f"  - {self.output_dir}/classifier_ranking.csv")
        print(f"  - {self.output_dir}/best_by_dataset.csv")
        print(f"  - {self.output_dir}/best_accuracy_ordered.csv")
    
    def create_correlation_analysis(self):
        """Análisis de correlaciones entre métricas"""
        print("\nGenerando análisis de correlaciones...")
        
        # Seleccionar columnas numéricas
        numeric_cols = ['Accuracy', 'Error_Rate', 'Kappa', 'MAE', 'RMSE', 
                       'Precision', 'Recall', 'F1-Score']
        if 'Time(s)' in self.df.columns:
            numeric_cols.append('Time(s)')
        
        # Filtrar solo columnas que existen
        available_cols = [col for col in numeric_cols if col in self.df.columns]
        
        if len(available_cols) < 2:
            print("No hay suficientes métricas numéricas para análisis de correlación")
            return
        
        # Calcular correlaciones
        corr_matrix = self.df[available_cols].corr()
        
        # Crear heatmap de correlaciones
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.3f', cbar_kws={'label': 'Correlación'})
        
        plt.title('Matriz de Correlación entre Métricas', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Guardar
        output_path = os.path.join(self.output_dir, 'correlation_matrix.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Matriz de correlación guardada en: {output_path}")
    
    def create_roc_auc_analysis(self):
        """Análisis de curvas ROC y valores AUC"""
        print("\nGenerando análisis de ROC/AUC...")
        
        # Verificar si existe la columna AUC
        if 'AUC' not in self.df.columns:
            print("No hay datos de AUC disponibles para análisis ROC")
            return
        
        # Filtrar datos válidos de AUC (no 'N/A')
        valid_auc_df = self.df[self.df['AUC'] != 'N/A'].copy()
        if len(valid_auc_df) == 0:
            print("No hay datos válidos de AUC para análisis")
            return
        
        # Convertir AUC a numérico
        valid_auc_df['AUC'] = pd.to_numeric(valid_auc_df['AUC'])
        
        # GRÁFICO 1: Heatmap de AUC por dataset y clasificador
        plt.figure(figsize=(14, 10))
        pivot_auc = valid_auc_df.pivot(index='Dataset', columns='Classifier', values='AUC')
        sns.heatmap(pivot_auc, annot=True, fmt='.2f', cmap='coolwarm',
                   cbar_kws={'label': 'AUC'}, linewidths=0.5)
        plt.title('AUC por Dataset y Clasificador', fontsize=16, fontweight='bold')
        plt.xlabel('Clasificador', fontsize=12, fontweight='bold')
        plt.ylabel('Dataset', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Guardar primer gráfico
        output_path1 = os.path.join(self.output_dir, 'auc_heatmap.png')
        plt.savefig(output_path1, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Heatmap AUC guardado en: {output_path1}")
        
        # GRÁFICO 2: AUC promedio por clasificador con boxplot
        classifier_auc = valid_auc_df.groupby('Classifier')['AUC'].mean().sort_values(ascending=False)
        
        # Crear datos para boxplot
        classifiers_ordered = classifier_auc.index.tolist()
        auc_data_ordered = [valid_auc_df[valid_auc_df['Classifier'] == clf]['AUC'].values 
                           for clf in classifiers_ordered]
        
        plt.figure(figsize=(16, 10))
        
        # Boxplot con colores personalizados
        bp = plt.boxplot(auc_data_ordered, positions=range(len(classifiers_ordered)), 
                        patch_artist=True, widths=0.6)
        
        # Colorear boxplots según calidad
        def get_color_by_auc(auc_val):
            if auc_val >= 0.9:
                return '#2E8B57'  # Verde oscuro - Excelente
            elif auc_val >= 0.8:
                return '#32CD32'  # Verde - Bueno
            elif auc_val >= 0.7:
                return '#FFD700'  # Amarillo - Aceptable
            elif auc_val >= 0.6:
                return '#FF8C00'  # Naranja - Pobre
            else:
                return '#DC143C'  # Rojo - Fallo
        
        for patch, clf in zip(bp['boxes'], classifiers_ordered):
            avg_auc = classifier_auc[clf]
            patch.set_facecolor(get_color_by_auc(avg_auc))
            patch.set_alpha(0.8)
        
        # Líneas de referencia
        plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Random (0.5)')
        plt.axhline(y=0.8, color='green', linestyle=':', alpha=0.7, label='Bueno (≥0.8)')
        plt.axhline(y=0.9, color='darkgreen', linestyle=':', alpha=0.7, label='Excelente (≥0.9)')
        
        # Añadir valores promedio en cada boxplot
        for i, (clf, avg_auc) in enumerate(classifier_auc.items()):
            plt.text(i, avg_auc + 0.02, f'{avg_auc:.3f}', 
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.xlabel('Clasificador', fontsize=12, fontweight='bold')
        plt.ylabel('AUC', fontsize=12, fontweight='bold')
        plt.title('Distribución y Promedio AUC por Clasificador', fontsize=16, fontweight='bold')
        plt.xticks(range(len(classifiers_ordered)), classifiers_ordered, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.legend(loc='lower right')
        plt.ylim(0.4, 1.0)
        plt.tight_layout()
        
        # Guardar segundo gráfico
        output_path2 = os.path.join(self.output_dir, 'auc_boxplot.png')
        plt.savefig(output_path2, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Boxplot AUC guardado en: {output_path2}")
        
        # Imprimir estadísticas de AUC
        print(f"\n{'='*60}")
        print("ESTADÍSTICAS AUC")
        print(f"{'='*60}")
        print(f"AUC Promedio: {valid_auc_df['AUC'].mean():.4f}")
        print(f"AUC Mediana: {valid_auc_df['AUC'].median():.4f}")
        print(f"AUC Desviación Estándar: {valid_auc_df['AUC'].std():.4f}")
        print(f"AUC Mínimo: {valid_auc_df['AUC'].min():.4f}")
        print(f"AUC Máximo: {valid_auc_df['AUC'].max():.4f}")
        
        # Mejores clasificadores por AUC
        best_auc_classifiers = classifier_auc.head()
        print(f"\nMejores clasificadores por AUC:")
        for i, (clf, auc_val) in enumerate(best_auc_classifiers.items(), 1):
            print(f"{i}. {clf}: {auc_val:.4f}")
        
        # Clasificadores con AUC > 0.8 (considerados buenos)
        good_classifiers = classifier_auc[classifier_auc >= 0.8]
        print(f"\nClasificadores con AUC ≥ 0.8 (Bueno/Excelente): {len(good_classifiers)}")
        for clf, auc_val in good_classifiers.items():
            print(f"  - {clf}: {auc_val:.4f}")

    def generate_summary_report(self):
        """Generar reporte resumen completo"""
        print("\nGenerando reporte resumen...")
        
        report_path = os.path.join(self.output_dir, 'analysis_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("REPORTE DE ANÁLISIS DE RESULTADOS DE CLASIFICACIÓN\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Archivo analizado: {self.csv_file}\n")
            f.write(f"Total de experimentos: {len(self.df)}\n")
            f.write(f"Datasets: {self.df['Dataset'].nunique()}\n")
            f.write(f"Clasificadores: {self.df['Classifier'].nunique()}\n\n")
            
            # Estadísticas generales
            f.write("ESTADÍSTICAS GENERALES DE ACCURACY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Media: {self.df['Accuracy'].mean():.2f}%\n")
            f.write(f"Mediana: {self.df['Accuracy'].median():.2f}%\n")
            f.write(f"Desviación estándar: {self.df['Accuracy'].std():.2f}%\n")
            f.write(f"Mínimo: {self.df['Accuracy'].min():.2f}%\n")
            f.write(f"Máximo: {self.df['Accuracy'].max():.2f}%\n\n")
            
            # Mejores resultados
            best_overall = self.df.loc[self.df['Accuracy'].idxmax()]
            f.write("MEJOR RESULTADO GENERAL\n")
            f.write("-" * 25 + "\n")
            f.write(f"Dataset: {best_overall['Dataset']}\n")
            f.write(f"Clasificador: {best_overall['Classifier']}\n")
            f.write(f"Accuracy: {best_overall['Accuracy']:.2f}%\n")
            if 'F1-Score' in best_overall:
                f.write(f"F1-Score: {best_overall['F1-Score']:.4f}\n")
            if 'Time(s)' in best_overall:
                f.write(f"Tiempo: {best_overall['Time(s)']:.2f}s\n")
            f.write("\n")
            
            # Clasificadores más consistentes (menor std)
            classifier_std = self.df.groupby('Classifier')['Accuracy'].std().sort_values()
            f.write("CLASIFICADORES MÁS CONSISTENTES (menor variabilidad)\n")
            f.write("-" * 55 + "\n")
            for i, (clf, std_val) in enumerate(classifier_std.head().items(), 1):
                mean_acc = self.df[self.df['Classifier'] == clf]['Accuracy'].mean()
                f.write(f"{i}. {clf}: σ={std_val:.2f}%, μ={mean_acc:.2f}%\n")
            f.write("\n")
            
            f.write("Archivos generados en la carpeta 'analysis_output/':\n")
            f.write("- accuracy_heatmap.png\n")
            f.write("- performance_comparison.png\n")
            f.write("- dataset_analysis.png\n")
            f.write("- knn_analysis.png\n")
            f.write("- time_analysis.png (si hay datos de tiempo)\n")
            f.write("- correlation_matrix.png\n")
            f.write("- auc_heatmap.png (si hay datos de AUC)\n")
            f.write("- auc_boxplot.png (si hay datos de AUC)\n")
            f.write("- classifier_ranking.csv\n")
            f.write("- best_by_dataset.csv\n")
        
        print(f"Reporte completo guardado en: {report_path}")
    
    def generate_best_accuracy_csv(self):
        """Generar CSV ordenado con la mejor accuracy por dataset/filter"""
        print("\nGenerando CSV ordenado por mejor accuracy...")
        
        # Encontrar la mejor accuracy por dataset
        best_results = self.df.loc[self.df.groupby('Dataset')['Accuracy'].idxmax()]
        
        # Seleccionar columnas relevantes y ordenar por accuracy descendente
        ordered_results = best_results[['Dataset', 'Classifier', 'Accuracy', 'F1-Score', 
                                       'Precision', 'Recall', 'Kappa']].copy()
        
        # Agregar columnas adicionales si están disponibles
        if 'AUC' in self.df.columns:
            ordered_results['AUC'] = best_results['AUC']
        if 'Time(s)' in self.df.columns:
            ordered_results['Time(s)'] = best_results['Time(s)']
        
        # Ordenar por accuracy descendente
        ordered_results = ordered_results.sort_values('Accuracy', ascending=False).reset_index(drop=True)
        
        # Agregar ranking
        ordered_results.insert(0, 'Rank', range(1, len(ordered_results) + 1))
        
        # Extraer información del filtro del nombre del dataset
        def extract_filter_info(dataset_name):
            """Extraer información del filtro del nombre del dataset"""
            parts = dataset_name.split('_')
            if len(parts) >= 2:
                # Intentar identificar el tipo de filtro
                for part in parts:
                    if any(filter_type in part.lower() for filter_type in 
                          ['edge', 'emboss', 'gaussian', 'magnify', 'equalize']):
                        return part.capitalize()
                # Si no se encuentra un filtro específico, usar la primera parte
                return parts[0].capitalize()
            return dataset_name
        
        ordered_results['Filter_Type'] = ordered_results['Dataset'].apply(extract_filter_info)
        
        # Reordenar columnas para mejor legibilidad
        column_order = ['Rank', 'Dataset', 'Filter_Type', 'Classifier', 'Accuracy', 
                       'F1-Score', 'Precision', 'Recall', 'Kappa']
        
        if 'AUC' in ordered_results.columns:
            column_order.append('AUC')
        if 'Time(s)' in ordered_results.columns:
            column_order.append('Time(s)')
        
        ordered_results = ordered_results[column_order]
        
        # Redondear valores numéricos para mejor presentación
        numeric_columns = ['Accuracy', 'F1-Score', 'Precision', 'Recall', 'Kappa']
        if 'AUC' in ordered_results.columns:
            numeric_columns.append('AUC')
        if 'Time(s)' in ordered_results.columns:
            numeric_columns.append('Time(s)')
        
        for col in numeric_columns:
            if col in ordered_results.columns:
                ordered_results[col] = ordered_results[col].round(4)
        
        # Guardar CSV
        output_path = os.path.join(self.output_dir, 'best_accuracy_ordered.csv')
        ordered_results.to_csv(output_path, index=False)
        
        # Mostrar resumen en consola
        print(f"\n{'='*80}")
        print("RANKING DE DATASETS POR MEJOR ACCURACY")
        print(f"{'='*80}")
        
        # Mostrar top 10 o todos si son menos
        display_count = min(10, len(ordered_results))
        display_results = ordered_results.head(display_count)[['Rank', 'Dataset', 'Filter_Type', 
                                                              'Classifier', 'Accuracy']]
        print(display_results.to_string(index=False))
        
        if len(ordered_results) > 10:
            print(f"\n... y {len(ordered_results) - 10} datasets más")
        
        print(f"\nCSV completo guardado en: {output_path}")
        
        # Estadísticas adicionales
        print(f"\nEstadísticas del ranking:")
        print(f"  - Total datasets: {len(ordered_results)}")
        print(f"  - Mejor accuracy: {ordered_results['Accuracy'].max():.2f}% ({ordered_results.iloc[0]['Dataset']})")
        print(f"  - Accuracy promedio: {ordered_results['Accuracy'].mean():.2f}%")
        print(f"  - Clasificador más frecuente: {ordered_results['Classifier'].mode().iloc[0]}")
        
        # Análisis por tipo de filtro si es posible
        if 'Filter_Type' in ordered_results.columns:
            filter_stats = ordered_results.groupby('Filter_Type')['Accuracy'].agg(['mean', 'count']).round(2)
            filter_stats = filter_stats.sort_values('mean', ascending=False)
            
            print(f"\nRendimiento promedio por tipo de filtro:")
            for filter_type, stats in filter_stats.iterrows():
                print(f"  - {filter_type}: {stats['mean']:.2f}% (n={int(stats['count'])})")
    
    def create_confusion_matrix_interactive(self, db_folder: str):
        """
        Generar matriz de confusión de forma interactiva
        Permite elegir filtro y clasificador mediante menús
        """
        print("\n" + "="*80)
        print("GENERADOR INTERACTIVO DE MATRIZ DE CONFUSIÓN")
        print("="*80)
        
        # Verificar que el folder de base de datos existe
        if not os.path.exists(db_folder):
            print(f"Error: El directorio '{db_folder}' no existe.")
            return
        
        try:
            # Paso 1: Elegir tipo de filtro (ImageMagick)
            print("\n1. Seleccione el tipo de filtro de ImageMagick:")
            magick_filters = ['converted_edge', 'converted_emboss', 'converted_equalize', 
                            'converted_gaussian', 'converted_magnify', 'original']
            
            available_magick = []
            for i, filter_name in enumerate(magick_filters, 1):
                # Verificar si existen datasets con este filtro
                datasets_with_filter = self.df[self.df['Dataset'].str.startswith(filter_name)]
                if len(datasets_with_filter) > 0:
                    available_magick.append(filter_name)
                    print(f"  {len(available_magick)}. {filter_name} ({len(datasets_with_filter)} datasets)")
            
            if not available_magick:
                print("No se encontraron filtros disponibles en los datos.")
                return
            
            while True:
                try:
                    choice = int(input(f"\nElegir filtro (1-{len(available_magick)}): "))
                    if 1 <= choice <= len(available_magick):
                        selected_magick = available_magick[choice - 1]
                        break
                    print(f"Por favor, ingrese un número entre 1 y {len(available_magick)}")
                except ValueError:
                    print("Por favor, ingrese un número válido")
            
            # Paso 2: Elegir filtro de ImageFilters (sufijo ACCF, FCTHF, FOHF)
            print(f"\n2. Seleccione el filtro de ImageFilters:")
            
            # Extraer sufijos únicos para el filtro seleccionado
            # Los datasets tienen formato: converted_edge_ACCF, converted_magnify_FOHF, etc.
            # Queremos extraer solo el último componente (ACCF, FCTHF, FOHF)
            filtered_datasets = self.df[self.df['Dataset'].str.startswith(selected_magick)]
            suffixes = set()
            for dataset in filtered_datasets['Dataset'].unique():
                # Extraer el último componente después del último _
                parts = dataset.split('_')
                if len(parts) >= 2:
                    suffix = parts[-1]  # Solo el último componente (ACCF, FCTHF, FOHF)
                    suffixes.add(suffix)
            
            suffixes = sorted(list(suffixes))
            
            if not suffixes:
                print("No se encontraron sufijos disponibles.")
                return
            
            for i, suffix in enumerate(suffixes, 1):
                print(f"  {i}. {suffix}")
            
            while True:
                try:
                    choice = int(input(f"\nElegir sufijo (1-{len(suffixes)}): "))
                    if 1 <= choice <= len(suffixes):
                        selected_suffix = suffixes[choice - 1]
                        break
                    print(f"Por favor, ingrese un número entre 1 y {len(suffixes)}")
                except ValueError:
                    print("Por favor, ingrese un número válido")
            
            # Construir el nombre completo del dataset
            dataset_name = f"{selected_magick}_{selected_suffix}"
            
            # Verificar que el dataset existe
            dataset_data = self.df[self.df['Dataset'] == dataset_name]
            if len(dataset_data) == 0:
                print(f"\nError: No se encontraron datos para el dataset '{dataset_name}'")
                return
            
            # Paso 3: Elegir clasificador
            print(f"\n3. Seleccione el clasificador para '{dataset_name}':")
            classifiers = sorted(dataset_data['Classifier'].unique())
            
            for i, clf in enumerate(classifiers, 1):
                accuracy = dataset_data[dataset_data['Classifier'] == clf]['Accuracy'].values[0]
                print(f"  {i}. {clf} (Accuracy: {accuracy:.2f}%)")
            
            while True:
                try:
                    choice = int(input(f"\nElegir clasificador (1-{len(classifiers)}): "))
                    if 1 <= choice <= len(classifiers):
                        selected_classifier = classifiers[choice - 1]
                        break
                    print(f"Por favor, ingrese un número entre 1 y {len(classifiers)}")
                except ValueError:
                    print("Por favor, ingrese un número válido")
            
            # Ahora generar la matriz de confusión desde el archivo ARFF
            print(f"\n{'='*80}")
            print(f"Generando matriz de confusión...")
            print(f"  Dataset: {dataset_name}")
            print(f"  Clasificador: {selected_classifier}")
            print(f"{'='*80}")
            
            # Buscar el archivo ARFF correspondiente
            arff_file = self._find_arff_file(db_folder, dataset_name)
            if not arff_file:
                print(f"\nError: No se encontró el archivo ARFF para '{dataset_name}'")
                print(f"Buscando en: {db_folder}")
                return
            
            print(f"\nArchivo ARFF encontrado: {arff_file}")
            
            # Generar matriz de confusión usando Weka
            self._generate_confusion_matrix_weka(arff_file, selected_classifier, dataset_name)
            
        except KeyboardInterrupt:
            print("\n\nProceso interrumpido por el usuario.")
            return
        except Exception as e:
            print(f"\nError al generar matriz de confusión: {e}")
            import traceback
            traceback.print_exc()
    
    def _find_arff_file(self, db_folder: str, dataset_name: str) -> Optional[str]:
        """Buscar el archivo ARFF correspondiente al dataset"""
        # Ejemplo: dataset_name = "converted_magnify_ACCF"
        # Estructura: DB2/converted/magnify/ACCF.arff
        
        # Parsear el nombre del dataset
        parts = dataset_name.split('_')
        
        if len(parts) < 2:
            return None
        
        # Determinar el tipo de filtro y el sufijo
        if parts[0] == 'converted':
            # converted_edge_ACCF -> edge/ACCF.arff
            # converted_magnify_FOHF -> magnify/FOHF.arff
            if len(parts) >= 3:
                filter_type = parts[1]  # edge, magnify, gaussian, etc.
                suffix = parts[2]        # ACCF, FCTHF, FOHF
                
                # Construir rutas posibles
                possible_paths = [
                    os.path.join(db_folder, 'converted', filter_type, f"{suffix}.arff"),
                    os.path.join(db_folder, 'arff', 'converted', filter_type, f"{suffix}.arff"),
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        return path
        
        elif parts[0] == 'original':
            # original_ACCF -> original/ACCF.arff
            if len(parts) >= 2:
                suffix = parts[1]
                
                possible_paths = [
                    os.path.join(db_folder, 'original', f"{suffix}.arff"),
                    os.path.join(db_folder, 'arff', 'original', f"{suffix}.arff"),
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        return path
        
        # Si no se encontró con la lógica específica, buscar recursivamente
        for root, dirs, files in os.walk(db_folder):
            for file in files:
                if file.endswith('.arff'):
                    file_base = os.path.splitext(file)[0]
                    
                    # Intentar coincidencia con el sufijo (ACCF, FCTHF, FOHF)
                    if len(parts) >= 2:
                        suffix = parts[-1]  # Último componente
                        if file_base == suffix:
                            return os.path.join(root, file)
        
        return None
    
    def _generate_confusion_matrix_weka(self, arff_file: str, classifier_name: str, dataset_name: str):
        """Generar matriz de confusión"""
        try:
            import weka.core.jvm as jvm
            from weka.core.converters import Loader
            from weka.classifiers import Classifier, Evaluation
            from weka.core.classes import Random
            
            # Iniciar JVM si no está iniciada
            if not jvm.started:
                jvm.start(packages=True, max_heap_size="2g")
            
            print("\nCargando datos desde ARFF...")
            loader = Loader(classname="weka.core.converters.ArffLoader")
            data = loader.load_file(arff_file)
            data.class_is_last()
            
            # Remover atributo 'filename' si existe
            if data.attribute_by_name('filename'):
                from weka.filters import Filter
                remove = Filter(classname="weka.filters.unsupervised.attribute.Remove",
                              options=["-R", str(data.attribute_by_name('filename').index + 1)])
                remove.inputformat(data)
                data = remove.filter(data)
            
            print(f"Datos cargados: {data.num_instances} instancias, {data.num_attributes} atributos")
            print(f"Clases: {data.class_attribute.num_values}")
            
            # Crear el clasificador según el nombre
            classifier = self._get_classifier_by_name(classifier_name)
            
            if not classifier:
                print(f"Error: No se pudo crear el clasificador '{classifier_name}'")
                return
            
            print(f"\nEntrenando clasificador: {classifier_name}...")
            
            # Evaluar con cross-validation
            evaluation = Evaluation(data)
            evaluation.crossvalidate_model(classifier, data, 10, Random(1))
            
            print("\n" + "="*80)
            print("MATRIZ DE CONFUSIÓN")
            print("="*80)
            print(evaluation.matrix())
            
            # Obtener matriz como numpy array
            conf_matrix = []
            matrix_str = str(evaluation.matrix())
            lines = matrix_str.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if line and not line.startswith('='):
                    # Extraer números de la línea
                    numbers = []
                    parts = line.split()
                    for part in parts:
                        try:
                            numbers.append(float(part))
                        except ValueError:
                            continue
                    if numbers:
                        conf_matrix.append(numbers)
            
            if conf_matrix:
                conf_matrix = np.array(conf_matrix)
                
                # Obtener nombres de las clases
                class_names = [data.class_attribute.value(i) for i in range(data.class_attribute.num_values)]
                
                # Crear visualización de la matriz de confusión
                plt.figure(figsize=(10, 8))
                
                # Normalizar para mostrar porcentajes
                conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
                
                # Crear anotaciones con valores absolutos y porcentajes
                annot = np.empty_like(conf_matrix_percent, dtype=object)
                for i in range(conf_matrix.shape[0]):
                    for j in range(conf_matrix.shape[1]):
                        annot[i, j] = f'{int(conf_matrix[i, j])}\n({conf_matrix_percent[i, j]:.1f}%)'
                
                sns.heatmap(conf_matrix, annot=annot, fmt='', cmap='coolwarm', 
                           xticklabels=class_names, yticklabels=class_names,
                           cbar_kws={'label': 'Número de instancias'})
                
                plt.title(f'Matriz de Confusión\nDataset: {dataset_name}\nClasificador: {classifier_name}',
                         fontsize=14, fontweight='bold')
                plt.xlabel('Clase Predicha', fontsize=12, fontweight='bold')
                plt.ylabel('Clase Real', fontsize=12, fontweight='bold')
                plt.tight_layout()
                
                # Guardar
                output_filename = f"confusion_matrix_{dataset_name}_{classifier_name.replace(' ', '_')}.png"
                output_path = os.path.join(self.output_dir, output_filename)
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.show()
                
                print(f"\nMatriz de confusión guardada en: {output_path}")
                
                # Imprimir estadísticas adicionales
                print("\n" + "="*80)
                print("ESTADÍSTICAS DETALLADAS")
                print("="*80)
                print(evaluation.summary())
                print("\nESTADÍSTICAS POR CLASE:")
                print(evaluation.class_details())
            
        except ImportError:
            print("\nError: python-weka-wrapper3 no está instalado.")
            print("Instalar con: pip install python-weka-wrapper3")
        except Exception as e:
            print(f"\nError al generar matriz de confusión: {e}")
            import traceback
            traceback.print_exc()
    
    def _get_classifier_by_name(self, classifier_name: str):
        """Obtener instancia del clasificador por nombre"""
        from weka.classifiers import Classifier
        
        # Mapeo de nombres a clasificadores Weka
        classifier_map = {
            'NaiveBayesMultinomial': ('weka.classifiers.bayes.NaiveBayesMultinomial', []),
            '1-NN Simple': ('weka.classifiers.lazy.IBk', ['-K', '1', '-I']),
            '3-NN Simple': ('weka.classifiers.lazy.IBk', ['-K', '3', '-I']),
            '5-NN Simple': ('weka.classifiers.lazy.IBk', ['-K', '5', '-I']),
            '11-NN Simple': ('weka.classifiers.lazy.IBk', ['-K', '11', '-I']),
            '1-NN Weighted': ('weka.classifiers.lazy.IBk', ['-K', '1', '-F']),
            '3-NN Weighted': ('weka.classifiers.lazy.IBk', ['-K', '3', '-F']),
            '5-NN Weighted': ('weka.classifiers.lazy.IBk', ['-K', '5', '-F']),
            '11-NN Weighted': ('weka.classifiers.lazy.IBk', ['-K', '11', '-F']),
            'J48': ('weka.classifiers.trees.J48', []),
            'RandomForest': ('weka.classifiers.trees.RandomForest', []),
            'SMO': ('weka.classifiers.functions.SMO', []),
            'PART': ('weka.classifiers.rules.PART', []),
            'SimpleLogistic': ('weka.classifiers.functions.SimpleLogistic', []),
            'BayesNet(3padres)': ('weka.classifiers.bayes.BayesNet', ['-D', '-Q', 'weka.classifiers.bayes.net.search.local.K2', '--', '-P', '3', '-S', 'BAYES'])
        }
        
        if classifier_name in classifier_map:
            classname, options = classifier_map[classifier_name]
            return Classifier(classname=classname, options=options)
        
        return None
    
    def run_complete_analysis(self):
        """Ejecutar análisis completo"""
        print("CLASSIFICATION RESULTS ANALYZER")
        print("=" * 50)
        print(f"Analyzing: {self.csv_file}")
        print(f"Output directory: {self.output_dir}")
        print("=" * 50)
        
        # Estadísticas básicas
        self.print_basic_statistics()
        
        # Generar todos los gráficos y análisis
        self.create_accuracy_heatmap()
        self.create_performance_comparison()
        self.create_dataset_analysis()
        self.create_knn_comparison()
        self.create_time_analysis()
        self.create_correlation_analysis()
        self.create_roc_auc_analysis()
        
        # Generar tablas y reportes
        self.generate_ranking_tables()
        self.generate_summary_report()
        
        print(f"\n{'='*80}")
        print("ANÁLISIS COMPLETO FINALIZADO")
        print(f"{'='*80}")
        print(f"Todos los archivos se guardaron en: {self.output_dir}/")


def main():
    """Función principal"""
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(
        description='Analyze and visualize ARFF classification results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Análisis completo de DB2
  python view.py classification_results_DB2.csv
  
  # Análisis completo interactivo
  python view.py
  
  # Generar matriz de confusión interactiva para DB2
  python view.py -m DB2
  
  # Generar matriz de confusión para DB1
  python view.py -m DB1
        """)
    
    parser.add_argument('csv_file', nargs='?', help='CSV file with classification results')
    parser.add_argument('-m', '--confusion-matrix', metavar='DB_FOLDER', 
                       help='Generate confusion matrix interactively for specified database folder (e.g., DB1, DB2, DB3)')
    
    args = parser.parse_args()
    
    # Modo matriz de confusión
    if args.confusion_matrix:
        # Determinar el CSV y la carpeta de la base de datos
        db_folder = args.confusion_matrix
        db_number = db_folder.replace('DB', '')
        
        csv_file = f"classification_results_DB{db_number}.csv"
        
        if not os.path.exists(csv_file):
            print(f"Error: El archivo '{csv_file}' no existe.")
            print(f"Primero ejecute auto.py para generar los resultados.")
            return
        
        if not os.path.exists(db_folder):
            print(f"Error: El directorio '{db_folder}' no existe.")
            return
        
        # Crear analizador y generar matriz de confusión
        analyzer = ResultsAnalyzer(csv_file)
        analyzer.create_confusion_matrix_interactive(db_folder)
        return
    
    # Modo análisis completo
    csv_file = args.csv_file
    
    if not csv_file:
        # Solicitar al usuario qué base de datos analizar
        while True:
            try:
                db_input = input("\nEnter database number to analyze (e.g., 1 for DB1, 2 for DB2, 3 for DB3): ").strip()
                
                if not db_input:
                    print("Please enter a valid number.")
                    continue
                
                db_number = int(db_input)
                csv_file = f"classification_results_DB{db_number}.csv"
                
                # Verificar que el archivo existe
                if not os.path.exists(csv_file):
                    print(f"Warning: File '{csv_file}' does not exist.")
                    create_anyway = input("Do you want to continue anyway? (y/n): ").strip().lower()
                    if create_anyway not in ['y', 'yes']:
                        continue
                
                print(f"Using results file: {csv_file}")
                break
                
            except ValueError:
                print("Please enter a valid integer number.")
            except KeyboardInterrupt:
                print("\nProcess interrupted by user.")
                return
    else:
        if not os.path.exists(csv_file):
            print(f"Error: El archivo '{csv_file}' no existe.")
            return
    
    # Crear analizador y ejecutar análisis completo
    analyzer = ResultsAnalyzer(csv_file)
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()