#!/usr/bin/env python3
"""
Script para visualizar y analizar resultados de clasificación de archivos ARFF
Genera estadísticas, heatmaps, gráficos y reportes detallados
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
import warnings
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
        print("ESTADÍSTICAS BÁSICAS")
        print("="*80)
        
        print(f"Número total de experimentos: {len(self.df)}")
        print(f"Datasets evaluados: {self.df['Dataset'].nunique()}")
        print(f"Clasificadores probados: {self.df['Classifier'].nunique()}")
        
        if 'Accuracy' in self.df.columns:
            print(f"\nAccuracy:")
            print(f"  Promedio: {self.df['Accuracy'].mean():.2f}%")
            print(f"  Mediana: {self.df['Accuracy'].median():.2f}%")
            print(f"  Desviación estándar: {self.df['Accuracy'].std():.2f}%")
            print(f"  Rango: {self.df['Accuracy'].min():.2f}% - {self.df['Accuracy'].max():.2f}%")
        
        print(f"\nDatasets:")
        for dataset in sorted(self.df['Dataset'].unique()):
            count = len(self.df[self.df['Dataset'] == dataset])
            print(f"  {dataset}: {count} experimentos")
        
        print(f"\nClasificadores:")
        for classifier in sorted(self.df['Classifier'].unique()):
            count = len(self.df[self.df['Classifier'] == classifier])
            avg_acc = self.df[self.df['Classifier'] == classifier]['Accuracy'].mean()
            print(f"  {classifier}: {count} experimentos (Acc promedio: {avg_acc:.2f}%)")
    
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
                   cmap='RdYlBu_r',
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
        
        # Crear figura con subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Gráfico 1: Accuracy promedio con barras de error
        bars = ax1.bar(range(len(classifier_stats)), classifier_stats['mean'], 
                      yerr=classifier_stats['std'], capsize=5, 
                      color=sns.color_palette("husl", len(classifier_stats)))
        
        ax1.set_xlabel('Clasificador', fontweight='bold')
        ax1.set_ylabel('Accuracy Promedio (%)', fontweight='bold')
        ax1.set_title('Accuracy Promedio por Clasificador', fontweight='bold')
        ax1.set_xticks(range(len(classifier_stats)))
        ax1.set_xticklabels(classifier_stats['Classifier'], rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)
        
        # Añadir valores en las barras
        for i, (bar, val) in enumerate(zip(bars, classifier_stats['mean'])):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Gráfico 2: Boxplot de distribución de accuracy
        classifiers_for_box = self.df['Classifier'].unique()
        data_for_box = [self.df[self.df['Classifier'] == clf]['Accuracy'].values 
                       for clf in classifiers_for_box]
        
        bp = ax2.boxplot(data_for_box, labels=classifiers_for_box, patch_artist=True)
        colors = sns.color_palette("husl", len(classifiers_for_box))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_xlabel('Clasificador', fontweight='bold')
        ax2.set_ylabel('Accuracy (%)', fontweight='bold')
        ax2.set_title('Distribución de Accuracy por Clasificador', fontweight='bold')
        ax2.set_xticklabels(classifiers_for_box, rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)
        
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
            sns.heatmap(knn_pivot, annot=True, fmt='.1f', cmap='RdYlBu_r', ax=ax2,
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
        
        print(f"\nRankings guardados en:")
        print(f"  - {self.output_dir}/classifier_ranking.csv")
        print(f"  - {self.output_dir}/best_by_dataset.csv")
    
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
            f.write("- classifier_ranking.csv\n")
            f.write("- best_by_dataset.csv\n")
        
        print(f"Reporte completo guardado en: {report_path}")
    
    def run_complete_analysis(self):
        """Ejecutar análisis completo"""
        print("ANALIZADOR DE RESULTADOS DE CLASIFICACIÓN")
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
        
        # Generar tablas y reportes
        self.generate_ranking_tables()
        self.generate_summary_report()
        
        print(f"\n{'='*80}")
        print("ANÁLISIS COMPLETO FINALIZADO")
        print(f"{'='*80}")
        print(f"Todos los archivos se guardaron en: {self.output_dir}/")


def main():
    """Función principal"""
    # Verificar si hay argumentos de línea de comandos
    csv_file = "classification_results.csv"
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    
    # Crear analizador y ejecutar
    analyzer = ResultsAnalyzer(csv_file)
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()