#!/usr/bin/env python3
"""
Script para clasificar archivos ARFF usando diferentes algoritmos de machine learning
Utiliza python-weka-wrapper3 para acceder a los clasificadores de Weka
"""

import os
import sys
import glob
import pandas as pd
from typing import List, Dict, Tuple
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing
from functools import partial

# Importar weka
try:
    import weka.core.jvm as jvm
    from weka.core.converters import Loader
    from weka.classifiers import Classifier, Evaluation
    from weka.core.classes import Random
    from weka.filters import Filter
    print("Weka importado correctamente")
except ImportError:
    print("Error: python-weka-wrapper3 no está instalado.")
    print("Instale con: pip install python-weka-wrapper3")
    sys.exit(1)


class ARFFClassifier:
    """Clase para clasificar archivos ARFF con múltiples algoritmos"""
    
    def __init__(self, db_path: str = "DB"):
        """
        Inicializar el clasificador
        
        Args:
            db_path: Ruta a la carpeta DB con archivos ARFF
        """
        self.db_path = db_path
        self.results = []
        
        # Inicializar JVM de Weka
        if not jvm.started:
            jvm.start()
    
    def find_arff_files(self) -> List[str]:
        """
        Encontrar todos los archivos ARFF recursivamente en las carpetas original y converted
        
        Returns:
            Lista de rutas a archivos ARFF
        """
        arff_files = []
        
        # Carpetas donde buscar archivos ARFF
        search_folders = [
            os.path.join(self.db_path, "original"),
            os.path.join(self.db_path, "converted")
        ]
        
        for folder in search_folders:
            if os.path.exists(folder):
                # Buscar recursivamente archivos .arff
                pattern = os.path.join(folder, "**", "*.arff")
                folder_files = glob.glob(pattern, recursive=True)
                arff_files.extend(folder_files)
                print(f"En {folder}: encontrados {len(folder_files)} archivos ARFF")
            else:
                print(f"Carpeta no encontrada: {folder}")
        
        if not arff_files:
            print(f"No se encontraron archivos ARFF en las carpetas original y converted de {self.db_path}")
            return []
        
        print(f"\nTotal encontrados: {len(arff_files)} archivos ARFF:")
        for file in sorted(arff_files):
            # Mostrar ruta relativa desde DB para mejor legibilidad
            rel_path = os.path.relpath(file, self.db_path)
            print(f"  - {rel_path}")
        
        return sorted(arff_files)
    
    def load_dataset(self, arff_path: str):
        """
        Cargar dataset ARFF y eliminar atributo 'filename' si existe
        
        Args:
            arff_path: Ruta al archivo ARFF
            
        Returns:
            Dataset de Weka sin el atributo filename
        """
        try:
            loader = Loader(classname="weka.core.converters.ArffLoader")
            dataset = loader.load_file(arff_path)
            dataset.class_is_last()  # Asumir que la clase está en la última columna
            
            # Buscar y eliminar atributo 'filename' si existe
            filename_index = -1
            for i in range(dataset.num_attributes):
                attr_name = dataset.attribute(i).name.lower()
                if attr_name == 'filename':
                    filename_index = i
                    break
            
            if filename_index != -1:
                print(f"    Eliminando atributo 'filename' (índice {filename_index})")
                # Usar filtro Remove para eliminar el atributo filename
                from weka.filters import Filter
                remove_filter = Filter(classname="weka.filters.unsupervised.attribute.Remove", 
                                     options=["-R", str(filename_index + 1)])  # Weka usa índices base-1
                remove_filter.inputformat(dataset)
                dataset = remove_filter.filter(dataset)
                dataset.class_is_last()  # Reconfigurar la clase después del filtrado
            
            return dataset
        except Exception as e:
            print(f"Error cargando {arff_path}: {e}")
            return None
    
    def get_classifiers(self) -> Dict[str, Dict]:
        """
        Configurar todos los clasificadores solicitados
        
        Returns:
            Diccionario con nombre -> configuración del clasificador
        """
        classifiers = {}
        
        # K-NN Classifiers (IBk en Weka)
        classifiers["1-NN Simple"] = {
            'classname': "weka.classifiers.lazy.IBk",
            'options': ["-K", "1"]
        }
        
        classifiers["3-NN Simple"] = {
            'classname': "weka.classifiers.lazy.IBk",
            'options': ["-K", "3"]
        }
        
        classifiers["3-NN Weighted"] = {
            'classname': "weka.classifiers.lazy.IBk",
            'options': ["-K", "3", "-I"]
        }
        
        classifiers["5-NN Simple"] = {
            'classname': "weka.classifiers.lazy.IBk",
            'options': ["-K", "5"]
        }
        
        classifiers["5-NN Weighted"] = {
            'classname': "weka.classifiers.lazy.IBk",
            'options': ["-K", "5", "-I"]
        }
        
        classifiers["11-NN Simple"] = {
            'classname': "weka.classifiers.lazy.IBk",
            'options': ["-K", "11"]
        }
        
        classifiers["11-NN Weighted"] = {
            'classname': "weka.classifiers.lazy.IBk",
            'options': ["-K", "11", "-I"]
        }
        
        # J48 (C4.5 Decision Tree)
        classifiers["J48"] = {
            'classname': "weka.classifiers.trees.J48",
            'options': ["-C", "0.25", "-M", "2"]
        }
        
        # Random Forest
        classifiers["RandomForest"] = {
            'classname': "weka.classifiers.trees.RandomForest",
            'options': ["-I", "100", "-K", "0", "-S", "1"]
        }
        
        # Naive Bayes Multinomial
        classifiers["NaiveBayesMultinomial"] = {
            'classname': "weka.classifiers.bayes.NaiveBayesMultinomial",
            'options': []
        }
        
        # BayesNet (con 3 padres máximo)
        classifiers["BayesNet(3padres)"] = {
            'classname': "weka.classifiers.bayes.BayesNet",
            'options': ["-D", "-Q", "weka.classifiers.bayes.net.search.local.K2", 
                       "--", "-P", "3", "-S", "BAYES"]
        }
        
        # PART
        classifiers["PART"] = {
            'classname': "weka.classifiers.rules.PART",
            'options': ["-C", "0.25", "-M", "2"]
        }
        
        # SimpleLogistic
        classifiers["SimpleLogistic"] = {
            'classname': "weka.classifiers.functions.SimpleLogistic",
            'options': ["-I", "0", "-M", "500", "-H", "50", "-W", "0.0"]
        }
        
        # SMO (Support Vector Machine)
        classifiers["SMO"] = {
            'classname': "weka.classifiers.functions.SMO",
            'options': ["-C", "1.0", "-L", "0.001", "-P", "1.0E-12", "-N", "0", "-V", "-1", "-W", "1", "-K", "weka.classifiers.functions.supportVector.PolyKernel -E 1.0 -C 250007"]
        }
        
        return classifiers
    
    def evaluate_classifier(self, classifier, dataset, classifier_name: str, 
                          dataset_name: str, folds: int = 10) -> Dict:
        """
        Evaluar un clasificador usando validación cruzada
        
        Args:
            classifier: Clasificador de Weka (puede ser objeto o configuración)
            dataset: Dataset de Weka
            classifier_name: Nombre del clasificador
            dataset_name: Nombre del dataset
            folds: Número de folds para validación cruzada
            
        Returns:
            Diccionario con métricas de evaluación
        """
        try:
            # Si classifier es un diccionario de configuración, crear el objeto
            if isinstance(classifier, dict):
                classifier = Classifier(
                    classname=classifier['classname'],
                    options=classifier.get('options', [])
                )
            
            # Crear evaluación con validación cruzada
            evaluation = Evaluation(dataset)
            evaluation.crossvalidate_model(classifier, dataset, folds, Random(1))
            
            # Extraer métricas
            results = {
                'Dataset': dataset_name,
                'Classifier': classifier_name,
                'Accuracy': round(evaluation.percent_correct, 2),
                'Error_Rate': round(evaluation.percent_incorrect, 2),
                'Kappa': round(evaluation.kappa, 4),
                'MAE': round(evaluation.mean_absolute_error, 4),
                'RMSE': round(evaluation.root_mean_squared_error, 4),
                'Precision': round(evaluation.weighted_precision, 4),
                'Recall': round(evaluation.weighted_recall, 4),
                'F1-Score': round(evaluation.weighted_f_measure, 4),
                'AUC': round(evaluation.weighted_area_under_roc, 4) if evaluation.weighted_area_under_roc != -1 else 'N/A'
            }
            
            return results
            
        except Exception as e:
            print(f"Error evaluando {classifier_name} en {dataset_name}: {e}")
            return {
                'Dataset': dataset_name,
                'Classifier': classifier_name,
                'Error': str(e)
            }
    
    def evaluate_single_classifier(self, classifier_config: Tuple, dataset, dataset_name: str) -> Dict:
        """
        Evaluar un solo clasificador (para paralelización)
        
        Args:
            classifier_config: Tupla (nombre_clasificador, configuración_clasificador)
            dataset: Dataset de Weka
            dataset_name: Nombre del dataset
            
        Returns:
            Diccionario con métricas de evaluación
        """
        clf_name, clf_config = classifier_config
        
        try:
            # Recrear el clasificador para evitar problemas de concurrencia
            classifier = Classifier(classname=clf_config['classname'], 
                                  options=clf_config.get('options', []))
            
            start_time = time.time()
            result = self.evaluate_classifier(classifier, dataset, clf_name, dataset_name)
            elapsed_time = time.time() - start_time
            
            result['Time(s)'] = round(elapsed_time, 2)
            return result
            
        except Exception as e:
            return {
                'Dataset': dataset_name,
                'Classifier': clf_name,
                'Error': str(e),
                'Time(s)': 0
            }
    
    def classify_all_datasets(self, max_workers: int = None):
        """
        Clasificar todos los datasets ARFF con todos los clasificadores (paralelo)
        
        Args:
            max_workers: Número máximo de workers para paralelización (None = auto)
        """
        # Encontrar archivos ARFF
        arff_files = self.find_arff_files()
        if not arff_files:
            return
        
        # Obtener configuraciones de clasificadores
        classifiers = self.get_classifiers()
        
        # Configurar número de workers
        if max_workers is None:
            max_workers = min(len(classifiers), multiprocessing.cpu_count())
        
        print(f"\n{'='*80}")
        print("INICIANDO CLASIFICACIÓN DE DATASETS (PARALELO)")
        print(f"{'='*80}")
        print(f"Datasets a procesar: {len(arff_files)}")
        print(f"Clasificadores a usar: {len(classifiers)}")
        print(f"Total de evaluaciones: {len(arff_files) * len(classifiers)}")
        print(f"Workers paralelos: {max_workers}")
        print(f"{'='*80}\n")
        
        # Procesar cada archivo ARFF
        for arff_file in arff_files:
            # Crear nombre descriptivo del dataset con la ruta relativa
            rel_path = os.path.relpath(arff_file, self.db_path)
            dataset_name = rel_path.replace('.arff', '').replace(os.sep, '_')
            print(f"Procesando: {rel_path}")
            
            # Cargar dataset
            dataset = self.load_dataset(arff_file)
            if dataset is None:
                continue
            
            print(f"  - Instancias: {dataset.num_instances}")
            print(f"  - Atributos: {dataset.num_attributes} (después del preprocesamiento)")
            print(f"  - Clases: {dataset.class_attribute.num_values}")
            print(f"  - Ejecutando {len(classifiers)} clasificadores en paralelo...")
            
            # Preparar tareas para paralelización
            classifier_items = list(classifiers.items())
            
            # Ejecutar clasificadores en paralelo usando ThreadPoolExecutor
            # (ThreadPool es mejor que ProcessPool para Weka debido a la JVM compartida)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Crear función parcial para el dataset actual
                evaluate_func = partial(self.evaluate_single_classifier, 
                                      dataset=dataset, 
                                      dataset_name=dataset_name)
                
                # Enviar todas las tareas
                future_to_classifier = {
                    executor.submit(evaluate_func, clf_config): clf_config[0]
                    for clf_config in classifier_items
                }
                
                # Recoger resultados conforme se completen
                completed = 0
                start_time = time.time()
                
                for future in as_completed(future_to_classifier):
                    clf_name = future_to_classifier[future]
                    
                    try:
                        result = future.result()
                        self.results.append(result)
                        completed += 1
                        
                        # Mostrar progreso
                        progress = (completed / len(classifiers)) * 100
                        elapsed = time.time() - start_time
                        print(f"    [{progress:5.1f}%] {clf_name}: {result.get('Accuracy', 'Error')}% "
                              f"({result.get('Time(s)', 0):.2f}s)")
                        
                    except Exception as e:
                        print(f"    [ERROR] {clf_name}: {e}")
                        self.results.append({
                            'Dataset': dataset_name,
                            'Classifier': clf_name,
                            'Error': str(e),
                            'Time(s)': 0
                        })
                        completed += 1
                
                total_time = time.time() - start_time
                print(f"    Dataset completado en {total_time:.2f}s\n")
    
    def save_results(self, output_file: str = "classification_results.csv"):
        """
        Guardar resultados en archivo CSV
        
        Args:
            output_file: Nombre del archivo de salida
        """
        if not self.results:
            print("No hay resultados para guardar")
            return
        
        df = pd.DataFrame(self.results)
        df.to_csv(output_file, index=False)
        print(f"\nResultados guardados en: {output_file}")
        
        # Mostrar resumen
        print(f"\n{'='*80}")
        print("RESUMEN DE RESULTADOS")
        print(f"{'='*80}")
        
        # Mostrar mejores resultados por dataset
        for dataset in df['Dataset'].unique():
            dataset_results = df[df['Dataset'] == dataset]
            if 'Accuracy' in dataset_results.columns:
                best_result = dataset_results.loc[dataset_results['Accuracy'].idxmax()]
                print(f"\n{dataset}:")
                print(f"  Mejor clasificador: {best_result['Classifier']}")
                print(f"  Accuracy: {best_result['Accuracy']}%")
                print(f"  F1-Score: {best_result['F1-Score']}")
                print(f"  Tiempo: {best_result['Time(s)']}s")
    
    def print_summary_table(self):
        """
        Mostrar tabla resumen de resultados
        """
        if not self.results:
            print("No hay resultados para mostrar")
            return
        
        df = pd.DataFrame(self.results)
        
        print(f"\n{'='*120}")
        print("TABLA COMPLETA DE RESULTADOS")
        print(f"{'='*120}")
        
        # Configurar pandas para mostrar todas las columnas
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        
        print(df.to_string(index=False))
    
    def cleanup(self):
        """Limpiar recursos"""
        if jvm.started:
            jvm.stop()


def main():
    """Función principal"""
    print("AUTOMATIC ARFF FILE CLASSIFIER (PARALLEL)")
    print("="*55)
    
    # Solicitar al usuario qué base de datos usar
    while True:
        try:
            db_input = input("\nEnter database folder number (e.g., 1 for DB1, 2 for DB2, 3 for DB3): ").strip()
            
            if not db_input:
                print("Please enter a valid number.")
                continue
            
            db_number = int(db_input)
            db_folder = f"DB{db_number}"
            
            # Verificar que la carpeta existe
            if not os.path.exists(db_folder):
                print(f"Warning: Folder '{db_folder}' does not exist in current directory.")
                create_anyway = input("Do you want to continue anyway? (y/n): ").strip().lower()
                if create_anyway not in ['y', 'yes']:
                    continue
            
            print(f"Using database folder: {db_folder}")
            break
            
        except ValueError:
            print("Please enter a valid integer number.")
        except KeyboardInterrupt:
            print("\nProcess interrupted by user.")
            return
    
    # Crear instancia del clasificador con la carpeta especificada
    classifier = ARFFClassifier(db_folder)
    
    try:
        # Ejecutar clasificación paralela
        # Usar None para auto-detectar número óptimo de workers
        classifier.classify_all_datasets(max_workers=None)
        
        # Mostrar y guardar resultados
        classifier.print_summary_table()
        output_file = f"classification_results_{db_folder}.csv"
        classifier.save_results(output_file)
        
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user")
    except Exception as e:
        print(f"\nError during execution: {e}")
    finally:
        # Limpiar recursos
        classifier.cleanup()
        print("\nProcess finished")


if __name__ == "__main__":
    main()
