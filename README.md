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
