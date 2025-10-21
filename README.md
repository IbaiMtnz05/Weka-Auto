# Automatic ARFF File Classifier

This script automatically classifies all ARFF files in any database folder (DB1, DB2, DB3, etc.) using multiple machine learning algorithms with parallel execution.

> [!NOTE]
> The script supports any database folder structure (DB1, DB2, DB3, etc.) and will prompt you to select which one to use during execution.

## Implemented Algorithms

### K-NN (K-Nearest Neighbors)
- **1-NN Simple**: Nearest neighbor without weighting
- **1-NN Weighted**: Nearest neighbor with distance-weighted voting
- **3-NN Simple**: 3 nearest neighbors, simple voting
- **3-NN Weighted**: 3 nearest neighbors, weighted voting
- **5-NN Simple**: 5 nearest neighbors, simple voting
- **5-NN Weighted**: 5 nearest neighbors, weighted voting
- **11-NN Simple**: 11 nearest neighbors, simple voting
- **11-NN Weighted**: 11 nearest neighbors, weighted voting

### Decision Trees
- **J48**: C4.5 implementation in Weka
- **RandomForest**: Random forest with 100 trees

### Bayesian Classifiers
- **Naive Bayes Multinomial**: Multinomial naive Bayes classifier
- **BayesNet**: Bayesian network with maximum 3 parents per node

### Rule-Based Classifiers
- **DecisionTable**: Decision table with best-first search

### Logistic Regression
- **SimpleLogistic**: Simple logistic regression classifier

> [!TIP]
> The script uses 14 different classifiers and runs them in parallel for optimal performance. On a multi-core system, you can expect 5-8x speedup compared to sequential execution.

## System Requirements

### Python Dependencies
```bash
pip install -r requirements.txt
```

### Java Runtime Environment
- Java 8 or higher (required by Weka)
- Verify with: `java -version`

> [!CAUTION]
> Make sure Java is properly installed and accessible from command line. Without Java, the Weka library cannot function.

## Installation

1. **Install dependencies:**
   ```bash
   pip install python-weka-wrapper3 pandas numpy
   ```

2. **Verify Java:**
   ```bash
   java -version
   ```

> [!TIP]
> If you encounter Java-related issues, try setting the JAVA_HOME environment variable to point to your Java installation directory.

## Usage

### Interactive Execution
```bash
python auto.py
```

The script will prompt you to select which database folder to use:

```
Enter database folder number (e.g., 1 for DB1, 2 for DB2, 3 for DB3): 2
Using database folder: DB2
```

> [!NOTE]
> The script automatically validates that the specified database folder exists before proceeding. If the folder doesn't exist, you'll be asked whether to continue anyway.

### Expected Folder Structure

Your database folders should follow this structure:
```
DB1/
├── original/
│   ├── image_filter1.arff
│   ├── image_filter2.arff
│   └── ...
└── converted/
    ├── edge/
    │   └── image_filter1.arff
    ├── gaussian/
    │   └── image_filter2.arff
    └── ...

DB2/
├── original/
└── converted/

DB3/
├── original/
└── converted/
```

> [!CAUTION]
> The script searches recursively in `original/` and `converted/` subfolders within your selected database folder. Make sure your ARFF files are placed in these locations.

### What the Script Does

The script:
1. **Prompts for database selection** (DB1, DB2, DB3, etc.)
2. **Automatically finds all `.arff` files** in `DBi/original/` and `DBi/converted/` folders recursively
3. **Loads each dataset** and configures the target class (last column)
4. **Automatically removes 'filename' attributes** if present
5. **Runs all 14 classifiers** on each dataset in parallel
6. **Uses 10-fold cross-validation** for each evaluation
7. **Calculates comprehensive metrics**: Accuracy, Error Rate, Kappa, MAE, RMSE, Precision, Recall, F1-Score, AUC
8. **Shows real-time progress** with parallel execution monitoring
9. **Saves results** to `classification_results_DBi.csv`
10. **Displays summary table** and best results per dataset

### Program Output

The script generates:

**Console:**
- Database folder selection prompt
- Real-time classification progress with parallel execution
- Complete results table
- Summary with best classifier per dataset

**CSV File:**
- `classification_results_DB1.csv` (or DB2, DB3, etc.): All results with detailed metrics

> [!TIP]
> Results are saved with the database folder name included in the filename, so you can run multiple database analyses without overwriting previous results.

## Calculated Metrics

- **Accuracy**: Percentage of correct classifications
- **Error Rate**: Percentage of errors
- **Kappa**: Agreement measure (chance-adjusted)
- **MAE**: Mean absolute error
- **RMSE**: Root mean squared error
- **Precision**: Weighted average precision
- **Recall**: Weighted average recall
- **F1-Score**: Weighted average F1-score
- **AUC**: Weighted average area under ROC curve
- **Time(s)**: Execution time per classifier

> [!NOTE]
> All metrics are calculated using 10-fold cross-validation to ensure robust performance estimates.

## Code Structure

### Class `ARFFClassifier`
- `find_arff_files()`: Finds ARFF files recursively in DBi/original and DBi/converted
- `load_dataset()`: Loads datasets with Weka and removes 'filename' attribute
- `get_classifiers()`: Configures all classifier configurations
- `evaluate_classifier()`: Cross-validation evaluation
- `evaluate_single_classifier()`: Single classifier evaluation for parallelization
- `classify_all_datasets()`: Main parallel classification process
- `save_results()`: Saves and displays results
- `cleanup()`: Cleans JVM resources

> [!TIP]
> The classifier automatically handles preprocessing by removing 'filename' attributes that are not useful for classification but often present in ARFF files.

## Classifier Configuration

### K-NN (IBk in Weka)
- `-K n`: Number of neighbors
- `-I`: Enable inverse distance weighting

### J48
- `-C 0.25`: Confidence factor for pruning
- `-M 2`: Minimum number of instances per leaf

### RandomForest
- `-I 100`: Number of trees
- `-K 0`: Number of random features (0 = √total_features)
- `-S 1`: Random seed

### BayesNet
- `-P 3`: Maximum 3 parents per node
- `-S BAYES`: Use Bayesian scoring

### DecisionTable
- `-X 1`: Use cross-validation to evaluate table
- `-S "BestFirst -D 1"`: Use BestFirst search forward direction
- `-N 5`: Number of non-improving nodes to end search

### SimpleLogistic
- `-I 0`: Maximum boosting iterations
- `-M 500`: Maximum iterations for LogitBoost
- `-H 50`: Heuristic stop criterion
- `-W 0.0`: Weight trimming parameter

> [!NOTE]
> All classifier parameters have been optimized for general performance across different types of datasets.

## Troubleshooting

### Error: "python-weka-wrapper3 is not installed"
```bash
pip install python-weka-wrapper3
```

### Error: "Could not find or load main class"
- Verify Java is installed
- Check JAVA_HOME variable

> [!CAUTION]
> Java-related errors are the most common issue. Ensure Java 8+ is installed and accessible from your command line.

### Error loading ARFF files
- Verify files exist in DBi/original/ and DBi/converted/
- Verify valid ARFF format
- Verify last column is the target class

> [!TIP]
> If you're unsure about ARFF file format, you can open them in a text editor. They should have a header section with @attribute declarations and a @data section with the actual data.

### Insufficient memory
- Increase JVM memory in code:
  ```python
  jvm.start(max_heap_size="4g")
  ```

> [!CAUTION]
> For large datasets or many parallel workers, you may need to increase JVM memory allocation to prevent out-of-memory errors.

## Project Files

- `auto.py`: Main classification script with parallelization and database selection
- `requirements.txt`: Python dependencies
- `README.md`: This documentation
- `DB1/, DB2/, DB3/, ...`: Database folders with ARFF files to classify
  - `original/`: Original image ARFF files
  - `converted/`: Processed image ARFF files (subfolders like edge/, emboss/, etc.)
- `classification_results_DB1.csv`: Generated results for DB1
- `classification_results_DB2.csv`: Generated results for DB2
- `classification_results_DB3.csv`: Generated results for DB3

> [!NOTE]
> Results files are automatically named based on the database folder selected, preventing accidental overwrites when processing multiple databases.

## Example Output

```
AUTOMATIC ARFF FILE CLASSIFIER (PARALLEL)
=======================================================

Enter database folder number (e.g., 1 for DB1, 2 for DB2, 3 for DB3): 2
Using database folder: DB2

Found 0 ARFF files in DB2/original
Found 15 ARFF files in DB2/converted

Total found: 15 ARFF files:
  - converted/edge/image_filter1.arff
  - converted/emboss/image_filter2.arff
  - converted/gaussian/image_filter3.arff
  - ...

================================================================================
STARTING DATASET CLASSIFICATION (PARALLEL)
================================================================================
Datasets to process: 15
Classifiers to use: 14
Total evaluations: 210
Parallel workers: 8
================================================================================

Processing: converted/edge/image_filter1.arff
    Removing 'filename' attribute (index 0)
  - Instances: 40
  - Attributes: 19 (after preprocessing)
  - Classes: 4
  - Running 14 classifiers in parallel...
    [ 14.3%] RandomForest: 90.0% (2.45s)
    [ 28.6%] J48: 87.5% (1.87s)
    [ 42.9%] DecisionTable: 92.5% (3.21s)
    [ 57.1%] 1-NN Simple: 85.0% (1.23s)
    [ 71.4%] SimpleLogistic: 88.0% (4.12s)
    [ 85.7%] BayesNet(3padres): 86.5% (2.78s)
    [100.0%] NaiveBayesMultinomial: 83.5% (1.45s)
    Dataset completed in 5.34s

================================================================================
RESULTS SUMMARY
================================================================================

converted_edge_image_filter1:
  Best classifier: DecisionTable
  Accuracy: 92.5%
  F1-Score: 0.9234
  Time: 3.21s

Results saved to: classification_results_DB2.csv
```

> [!TIP]
> Pay attention to the "Best classifier" information in the summary - this can help you understand which algorithms work best for your specific type of data.

## Performance Features

### Parallelization
- **ThreadPoolExecutor**: Uses threads instead of processes to maintain shared Weka JVM
- **Auto Worker Detection**: Automatically determines optimal number of threads
- **Parallel Evaluation**: All classifiers for a dataset run simultaneously
- **Real-time Progress**: Shows progress of each classifier as it completes

### Speed Improvements
- **5-8x faster** depending on CPU cores
- **Better resource utilization**
- **Same accuracy** as sequential version
- **Visible progress** for monitoring

### Database Flexibility
- **Interactive selection**: Choose any database folder (DB1, DB2, DB3, etc.)
- **Automatic validation**: Checks if folder exists before processing
- **Separate result files**: Each database gets its own results CSV
- **No configuration needed**: Just run and select your database

> [!NOTE]
> The parallel execution is most beneficial when you have multiple CPU cores and are processing datasets with many instances or using complex classifiers like RandomForest.

## Results Visualization and Analysis with view.py

The `view.py` script is a comprehensive analysis and visualization tool that processes the classification results generated by `auto.py`. It creates detailed statistical reports, interactive visualizations, and confusion matrices.

### Features Overview

- **📊 Comprehensive Visualizations**: Generate 8+ types of charts and heatmaps
- **📈 Statistical Analysis**: Detailed metrics, rankings, and correlations
- **🔍 Interactive Confusion Matrices**: Generate confusion matrices with menu-based selection
- **📁 Multi-Database Support**: Analyze results from DB1, DB2, DB3, etc.
- **💾 Export Reports**: CSV files, PNG images, and text reports

### Basic Usage

```bash
# Complete analysis of all results
python view.py classification_results_DB2.csv

# Interactive mode (prompts for database selection)
python view.py

# Show help and examples
python view.py -h
```

> [!TIP]
> The script automatically creates an output directory (`analysis_output_DB2/`) based on your database name, keeping results organized.

### Generated Visualizations

The script creates the following visualizations automatically:

#### 1. **Accuracy Heatmap** (`accuracy_heatmap.png`)
- **Description**: Matrix showing accuracy for each dataset-classifier combination
- **Features**: Color-coded values, centered colormap, annotations with accuracy percentages
- **Use Case**: Quick visual comparison of all classifiers across all datasets

#### 2. **Performance Comparison** (`performance_comparison.png`)
- **Description**: Bar chart with accuracy averages and standard deviation error bars
- **Features**: Sorted by best performance, color-coded bars, value labels
- **Use Case**: Identify the best-performing classifiers overall

#### 3. **Dataset Analysis** (`dataset_analysis.png`)
- **Description**: Two-panel visualization showing:
  - Average accuracy per dataset
  - Min-Max accuracy range per dataset
- **Features**: Identifies which datasets are easier/harder to classify
- **Use Case**: Understand dataset difficulty and classifier variability

#### 4. **K-NN Comparison** (`knn_analysis.png`)
- **Description**: Specialized analysis for K-NN classifiers comparing:
  - Simple vs Weighted voting by K value
  - Heatmap of K-NN performance across datasets
- **Features**: Side-by-side comparison, grouped by K values
- **Use Case**: Optimize K-NN parameter selection

#### 5. **Time Analysis** (`time_analysis.png`)
- **Description**: Two-panel visualization showing:
  - Average execution time per classifier (horizontal bar chart)
  - Scatter plot of accuracy vs execution time
- **Features**: Includes trend line, identifies fast and accurate classifiers
- **Use Case**: Balance accuracy and computational cost
- **Note**: Only generated if timing data is available

#### 6. **Correlation Matrix** (`correlation_matrix.png`)
- **Description**: Heatmap showing correlations between all metrics
- **Features**: Symmetric matrix with correlation coefficients
- **Metrics Analyzed**: Accuracy, Kappa, MAE, RMSE, Precision, Recall, F1-Score, AUC
- **Use Case**: Understand relationships between different performance metrics

#### 7. **AUC Heatmap** (`auc_heatmap.png`)
- **Description**: Matrix of AUC (Area Under ROC Curve) values by dataset and classifier
- **Features**: Color-coded values, detailed annotations
- **Use Case**: Evaluate classifier discrimination ability for multi-class problems

#### 8. **AUC Boxplot** (`auc_boxplot.png`)
- **Description**: Distribution of AUC values per classifier with quality indicators
- **Features**: 
  - Color-coded boxes (green=excellent, yellow=good, orange=fair, red=poor)
  - Reference lines at 0.5 (random), 0.8 (good), 0.9 (excellent)
  - Average values displayed above each boxplot
- **Use Case**: Compare classifier reliability and consistency

All visualizations are saved in `analysis_output_DBi/` directory with high resolution (300 DPI) for publication quality.

### Generated Reports and CSV Files

In addition to visualizations, `view.py` generates several analytical reports:

#### 1. **Classifier Ranking** (`classifier_ranking.csv`)
- Classifiers ranked by average accuracy
- Includes: mean accuracy, standard deviation, number of experiments, average time
- **Use Case**: Quick reference for best overall classifiers

#### 2. **Best Classifier per Dataset** (`best_by_dataset.csv`)
- Shows the top-performing classifier for each dataset
- Includes: dataset name, best classifier, accuracy, F1-score
- **Use Case**: Identify optimal classifier for specific data types

#### 3. **Best Accuracy Ordered** (`best_accuracy_ordered.csv`)
- Comprehensive ranking of datasets by their best achieved accuracy
- Includes: filter type extraction, all performance metrics
- **Use Case**: Understand which datasets are most accurately classifiable

#### 4. **Analysis Report** (`analysis_report.txt`)
- Text summary with overall statistics
- Best results and most consistent classifiers
- Complete file inventory
- **Use Case**: Quick overview without opening visualization files

### Interactive Confusion Matrix Generation

> [!NOTE]
> Advanced feature! Generate detailed confusion matrices interactively with menu-based selection system.

#### Command Usage

```bash
# Generate confusion matrix for DB2
python view.py -m DB2

# Generate confusion matrix for DB1
python view.py -m DB1

# General format
python view.py -m <DATABASE_FOLDER>
```

#### Interactive 3-Step Process

The system guides you through three selection menus:

**Step 1: Select ImageMagick Filter Type**

Choose the image preprocessing filter:
```
1. Seleccione el tipo de filtro de ImageMagick:
  1. converted_edge (42 datasets)      - Edge detection
  2. converted_emboss (42 datasets)    - Emboss effect
  3. converted_equalize (42 datasets)  - Histogram equalization
  4. converted_gaussian (42 datasets)  - Gaussian blur
  5. converted_magnify (42 datasets)   - Image magnification
  6. original (42 datasets)            - No preprocessing

Elegir filtro (1-6): 2
```

**Step 2: Select ImageFilters Feature Extractor**

Choose the feature extraction method:
```
2. Seleccione el filtro de ImageFilters:
  1. ACCF   - Average Color Class Filter
  2. FCTHF  - First Color Texture Histogram Filter
  3. FOHF   - First Order Histogram Filter

Elegir sufijo (1-3): 1
```

> [!NOTE]
> These are the actual feature extraction methods applied to the images, not to be confused with the ImageMagick preprocessing filters. ACCF extracts color averages, FCTHF extracts texture histograms, and FOHF extracts first-order statistics.

**Step 3: Select Classifier**

Choose from all available classifiers, sorted alphabetically with accuracy displayed:
```
3. Seleccione el clasificador para 'converted_emboss_ACCF':
  1. 1-NN Simple (Accuracy: 80.72%)
  2. 3-NN Simple (Accuracy: 73.78%)
  3. 3-NN Weighted (Accuracy: 76.09%)
  4. 5-NN Simple (Accuracy: 72.49%)
  5. 5-NN Weighted (Accuracy: 74.04%)
  6. 11-NN Simple (Accuracy: 63.50%)
  7. 11-NN Weighted (Accuracy: 65.55%)
  8. BayesNet(3padres) (Accuracy: 86.38%)
  9. J48 (Accuracy: 65.04%)
  10. NaiveBayesMultinomial (Accuracy: 74.55%)
  11. PART (Accuracy: 68.89%)
  12. RandomForest (Accuracy: 91.26%)
  13. SMO (Accuracy: 88.17%)
  14. SimpleLogistic (Accuracy: 86.12%)

Elegir clasificador (1-14): 12
```

> [!TIP]
> The accuracy values shown help you choose classifiers that performed well on the dataset, making it easier to analyze successful classifications.

#### Confusion Matrix Output

The confusion matrix generation provides comprehensive analysis:

**1. Console Output**
- Text-based confusion matrix with absolute counts
- Cross-validation summary statistics
- Detailed per-class metrics (precision, recall, F1-score)

**2. Visual Heatmap**
- Color-coded confusion matrix (blue intensity scale)
- Dual annotations: absolute counts and row-wise percentages
- Clear axis labels with predicted vs actual classes
- Dataset and classifier information in title

**3. Saved Image**
- Filename format: `confusion_matrix_{dataset}_{classifier}.png`
- Example: `confusion_matrix_converted_emboss_ACCF_RandomForest.png`
- High-resolution (300 DPI) for publication quality
- Saved in `analysis_output_DBi/` directory

**4. Statistical Summary**
```
================================================================================
ESTADÍSTICAS DETALLADAS
================================================================================
Correctly Classified Instances: 348 (91.26%)
Incorrectly Classified Instances: 33 (8.74%)
Kappa statistic: 0.8834
Mean absolute error: 0.1177
Root mean squared error: 0.1988

ESTADÍSTICAS POR CLASE:
                 TP Rate  FP Rate  Precision  Recall   F-Score  ROC Area  Class
                 0.912    0.025    0.918      0.912    0.911    0.991     class1
                 0.890    0.018    0.902      0.890    0.898    0.988     class2
                 0.935    0.031    0.921      0.935    0.924    0.993     class3
                 0.915    0.022    0.919      0.915    0.914    0.989     class4
Weighted Avg.    0.913    0.024    0.915      0.913    0.912    0.990
```

> [!TIP]
> The confusion matrix uses 10-fold cross-validation to ensure reliable and robust results. Each cell shows both the absolute count and the percentage of instances, making it easy to identify classification patterns and errors.

#### Requirements

> [!CAUTION]
> Confusion matrix generation requires:
> - ✅ Original ARFF files in the database folder structure (`DB2/converted/filter_type/suffix.arff`)
> - ✅ `python-weka-wrapper3` installed (`pip install python-weka-wrapper3`)
> - ✅ Java Runtime Environment 8+ available (`java -version`)
> - ✅ Results CSV file from auto.py execution (`classification_results_DB2.csv`)

#### File Structure Expected

```
DB2/
├── converted/
│   ├── edge/
│   │   ├── ACCF.arff
│   │   ├── FCTHF.arff
│   │   └── FOHF.arff
│   ├── emboss/
│   │   ├── ACCF.arff
│   │   ├── FCTHF.arff
│   │   └── FOHF.arff
│   └── ...
└── original/
    ├── ACCF.arff
    ├── FCTHF.arff
    └── FOHF.arff
```

> [!NOTE]
> The script intelligently maps dataset names from the CSV (e.g., `converted_emboss_ACCF`) to the correct ARFF file location (`DB2/converted/emboss/ACCF.arff`).

For detailed confusion matrix usage examples and troubleshooting, see [CONFUSION_MATRIX_USAGE.md](CONFUSION_MATRIX_USAGE.md)

### Complete Analysis Workflow Example

Here's a typical workflow for analyzing your classification results:

```bash
# Step 1: Run classification on your database
python auto.py
# Select database: 2 (for DB2)

# Step 2: Generate complete visual analysis
python view.py classification_results_DB2.csv

# Step 3: Generate confusion matrix for specific case
python view.py -m DB2
# Select: edge filter -> ACCF -> RandomForest

# Step 4: Review generated files
cd analysis_output_DB2/
ls -la
# accuracy_heatmap.png
# performance_comparison.png
# dataset_analysis.png
# knn_analysis.png
# auc_heatmap.png
# auc_boxplot.png
# correlation_matrix.png
# confusion_matrix_converted_edge_ACCF_RandomForest.png
# classifier_ranking.csv
# best_by_dataset.csv
# best_accuracy_ordered.csv
# analysis_report.txt
```

### Advanced Tips

> [!TIP]
> **For Publication-Ready Figures**: All images are generated at 300 DPI with tight bounding boxes. You can directly use them in papers or presentations.

> [!TIP]
> **For Quick Insights**: Check `analysis_report.txt` first for a text summary before opening all visualizations.

> [!TIP]
> **For Comparative Analysis**: Run `view.py` on multiple databases (DB1, DB2, DB3) to compare performance across different image sets.

> [!TIP]
> **For Parameter Tuning**: Use the K-NN analysis chart to understand how K value and voting strategy affect performance on your specific data.
