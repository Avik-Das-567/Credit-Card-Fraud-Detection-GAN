# Credit Card Fraud Detection using GAN-Based Data Balancing

## Overview

This project applies a Generative Adversarial Network (GAN) to the problem of class imbalance in credit card fraud detection. The dataset contains a very small number of fraudulent transactions compared with genuine transactions, which creates a difficult learning environment for fraud detection systems because the minority class is underrepresented.

The main objective is to learn the feature distribution of fraudulent transactions and generate synthetic fraud-like records that can be used for data balancing. Instead of relying only on random oversampling, the project trains a neural generator to produce new minority-class samples in the same numerical feature space as the original data. A discriminator is trained at the same time to distinguish real fraud records from generated records, creating the adversarial feedback loop that drives the generator to produce more realistic synthetic samples.

The completed notebook covers the full pipeline: dataset loading, class imbalance analysis, preprocessing, dimensionality reduction for visualization, generator construction, discriminator construction, GAN assembly, adversarial training, synthetic fraud generation, and visual comparison between real and generated fraud samples.

## Problem Statement

A financial institution needs to improve credit card fraud detection on a binary classification problem where the dataset is highly imbalanced. Genuine transactions dominate the dataset, while fraudulent transactions represent less than 1% of all records.

This imbalance can cause a fraud detection model to learn majority-class behavior much more strongly than minority-class behavior. In practice, that means a classifier can appear accurate overall while still failing to capture the transaction patterns that matter most: fraudulent activity.

The project addresses this imbalance by using generative AI to synthesize additional fraudulent transaction records. The GAN is trained only around the minority-class transaction feature space so that generated samples approximate the structure of real fraud records rather than the broader majority-class population.

## Dataset Summary

The dataset is loaded from `Creditcard_dataset.csv` into a pandas DataFrame.

Initial dataset characteristics:

| Attribute | Value |
| --- | ---: |
| Initial rows | 50,492 |
| Initial columns | 31 |
| Genuine transactions (`Class = 0`) | 50,000 |
| Fraudulent transactions (`Class = 1`) | 492 |
| Fraudulent transaction share | Approximately 0.97% |
| Genuine-to-fraud ratio | Approximately 101.6:1 |

The original columns include:

| Column group | Description |
| --- | --- |
| `Time` | Transaction time field removed during preprocessing |
| `V1` to `V28` | Numerical transaction features |
| `Amount` | Transaction amount, standardized before modeling |
| `Class` | Binary label where `0` is genuine and `1` is fraud |

After preprocessing, the model operates on 29 input features:

- `V1` through `V28`
- Standardized `Amount`

The target column `Class` is removed from the model input feature matrix and retained separately as the binary label.

## Technical Stack

The notebook uses the following Python libraries and machine learning tools:

| Area | Libraries and components |
| --- | --- |
| Numerical computing | `numpy` |
| Data manipulation | `pandas` |
| Deep learning | `tensorflow`, `keras`, `Sequential`, `Model`, `Input`, `Dense`, `BatchNormalization` |
| Optimizer and loss | `Adam`, binary cross-entropy |
| Preprocessing | `StandardScaler` from scikit-learn |
| Dimensionality reduction | `PCA` from scikit-learn |
| Visualization | `seaborn`, `matplotlib`, `plotly.express` |

TensorFlow/Keras is used to define the generator, discriminator, and combined GAN model. Scikit-learn is used for scaling and PCA-based visualization. Plotly and Seaborn are used to inspect class separation and compare generated fraud distributions with real fraud distributions.

## Data Preprocessing

The preprocessing pipeline prepares the transaction data for neural network training and visualization.

### Missing Value Handling

Rows containing missing values are removed with:

```python
data.dropna(inplace=True)
```

The dataset shape remains `50,492 x 31` after this operation, so no rows are removed by the missing-value cleanup step in the executed notebook.

### Time Column Removal

The `Time` column is dropped:

```python
data = data.drop(axis=1, columns="Time")
```

This reduces the dataset from 31 columns to 30 columns:

- 29 model input features
- 1 label column (`Class`)

The GAN is therefore built around a 29-dimensional feature vector.

### Amount Scaling

The `Amount` feature is standardized using `StandardScaler`:

```python
scaler = StandardScaler()
data["Amount"] = scaler.fit_transform(data[["Amount"]])
```

This step places `Amount` on a standardized scale so that transaction magnitude does not dominate the neural network's optimization dynamics relative to the other numerical features.

### Class-Based Splitting

The preprocessed data is split into separate DataFrames:

```python
data_fraud = data[data.Class == 1]
data_genuine = data[data.Class == 0]
```

This split is central to the GAN workflow because the generator is trained to synthesize fraudulent samples, and real fraud records are sampled from `data_fraud` during discriminator training.

### Feature and Label Separation

The model feature matrix and target label vector are separated as:

```python
X = data.drop("Class", axis=1)
y = data.Class
```

`X` contains the 29-dimensional transaction representation used for PCA visualization and GAN modeling. `y` contains the binary transaction labels used to visualize the genuine and fraudulent groups.

## Exploratory Analysis

The initial exploratory analysis confirms that the dataset is severely imbalanced:

```text
Class
0    50000
1      492
```

This means the fraudulent class accounts for only about 0.97% of all observations. The notebook then applies Principal Component Analysis (PCA) to reduce the 29-dimensional feature matrix into two components:

```python
pca = PCA(2)
transformed_data = pca.fit_transform(X)
df = pd.DataFrame(transformed_data)
df["label"] = y
```

The PCA output is visualized using a Plotly scatter plot:

```python
px.scatter(df, x=0, y=1, color=df.label.astype(str))
```

This visualization provides a two-dimensional view of the feature space and helps illustrate how fraud and genuine records are distributed after dimensionality reduction. PCA is also used later as a monitoring tool during GAN training to compare real fraud samples against synthetic fraud samples.

## GAN Architecture

The GAN consists of two neural networks:

- A generator that creates synthetic 29-feature fraud-like transaction records from random noise.
- A discriminator that learns to classify 29-feature records as real fraud samples or generated samples.

The combined GAN connects the generator to the discriminator and freezes the discriminator during generator updates. This allows the generator to improve based on discriminator feedback while preserving the discriminator's learned classification behavior during that specific update step.

### Generator

The generator maps a 29-dimensional random noise vector into a 29-dimensional synthetic transaction feature vector.

Architecture:

| Layer | Output size | Activation | Notes |
| --- | ---: | --- | --- |
| Input | 29 | None | Random normal noise vector |
| Dense | 32 | ReLU | He uniform initialization |
| Batch normalization | 32 | None | Stabilizes hidden activations |
| Dense | 64 | ReLU | Hidden representation expansion |
| Batch normalization | 64 | None | Stabilizes hidden activations |
| Dense | 128 | ReLU | Higher-capacity representation |
| Batch normalization | 128 | None | Stabilizes hidden activations |
| Dense | 29 | Linear | Synthetic transaction feature output |

Generator parameter count:

| Parameter type | Count |
| --- | ---: |
| Total parameters | 16,029 |
| Trainable parameters | 15,581 |
| Non-trainable parameters | 448 |

The linear output layer is appropriate because the generated data is continuous numerical tabular data rather than a bounded probability or class label.

### Discriminator

The discriminator receives a 29-dimensional transaction vector and returns a probability-like output indicating whether the input is real or generated.

Architecture:

| Layer | Output size | Activation |
| --- | ---: | --- |
| Input | 29 | None |
| Dense | 128 | ReLU |
| Dense | 64 | ReLU |
| Dense | 32 | ReLU |
| Dense | 32 | ReLU |
| Dense | 16 | ReLU |
| Dense | 1 | Sigmoid |

Discriminator parameter count:

| Parameter type | Count |
| --- | ---: |
| Total parameters | 15,777 |
| Trainable parameters | 15,777 |
| Non-trainable parameters | 0 |

The discriminator is compiled with:

```python
model.compile(optimizer="adam", loss="binary_crossentropy")
```

Binary cross-entropy is used because the discriminator performs a binary decision: real fraud transaction versus synthetic generated transaction.

### Combined GAN

The combined GAN is built by passing generator output into the discriminator:

```python
def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = Input(shape=(generator.input_shape[1],))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(gan_input, gan_output)
    return gan
```

During generator training, the GAN receives random noise as input and is trained with labels of `1`. This pushes the generator to produce outputs that the discriminator classifies as real.

## Project Workflow

1. Load transaction records from `Creditcard_dataset.csv`.
2. Inspect the dataset preview and confirm the initial shape of `50,492 x 31`.
3. Measure class imbalance with `data.Class.value_counts()`, showing `50,000` genuine transactions and `492` fraudulent transactions.
4. Remove rows with missing values using `dropna`; the row count remains unchanged.
5. Remove the `Time` column to focus the model on the numerical transaction feature space.
6. Standardize the `Amount` column with `StandardScaler`.
7. Split the dataset into `data_fraud` and `data_genuine`.
8. Separate the feature matrix `X` from the binary target label `y`.
9. Apply PCA to reduce the 29-dimensional feature matrix into two dimensions for exploratory visualization.
10. Build the generator network with a 29-dimensional input and a 29-dimensional output.
11. Build the discriminator network to distinguish real fraud vectors from generated vectors.
12. Combine generator and discriminator into a GAN model with the discriminator frozen for generator updates.
13. Define a synthetic data generation function that samples random normal noise and calls `generator.predict`.
14. Train the GAN for `1000` epochs with a batch size of `64`.
15. In each epoch, train the discriminator on:
    - `32` generated samples labeled `0`
    - `32` real fraud samples labeled `1`
16. Train the generator through the combined GAN using `64` random noise vectors labeled as real.
17. Every `10` epochs, monitor generator behavior by plotting PCA projections of real fraud samples and generated fraud samples.
18. Generate `1000` synthetic fraudulent data points using the trained generator.
19. Combine generated fraud samples with the original real fraud samples.
20. Compare real and synthetic fraud feature distributions using overlay histograms for each feature.

## Results

The notebook confirms the core data imbalance that motivates the use of generative balancing:

| Class | Meaning | Count | Share |
| ---: | --- | ---: | ---: |
| 0 | Genuine transaction | 50,000 | Approximately 99.03% |
| 1 | Fraudulent transaction | 492 | Approximately 0.97% |

Key preprocessing results:

- The dataset starts with `50,492` rows and `31` columns.
- Missing-value cleanup does not remove any records; the shape remains `50,492 x 31`.
- Dropping `Time` leaves 30 columns: 29 input features plus the `Class` label.
- The `Amount` column is standardized before model training.
- The GAN operates in the same 29-dimensional feature space used by the fraud records after preprocessing.

Key modeling results:

- The generator produces synthetic fraud-like records with 29 numerical features.
- The discriminator is trained to distinguish real fraud records from generated records.
- The combined GAN trains the generator through discriminator feedback while the discriminator is temporarily frozen.
- Training runs for `1000` epochs with a batch size of `64`.
- Generator monitoring is performed every `10` epochs using PCA projections of real and generated fraud samples.

Synthetic data generation result:

- The trained generator is used to create `1000` synthetic fraudulent transaction samples.
- These generated samples are combined with the original `492` real fraudulent samples.
- The final real-versus-synthetic comparison DataFrame contains `1,492` rows.
- Overlay histograms are created for each feature to visually compare the generated fraud distribution against the real fraud distribution.

The evaluation in this notebook is visual and distribution-focused. It checks whether synthetic samples occupy a plausible feature space relative to real fraud samples using PCA plots and per-feature histogram overlays. The notebook does not train a downstream fraud classifier on the augmented data and does not compute accuracy, precision, recall, F1-score, ROC-AUC, or other classification metrics.

## Key Takeaways

- The dataset has a severe minority-class imbalance, with only `492` fraudulent records out of `50,492` total transactions.
- Preprocessing converts the dataset into a 29-feature numerical modeling space by removing `Time`, scaling `Amount`, and excluding the `Class` label from the input matrix.
- PCA provides an interpretable two-dimensional visualization of both the original transaction space and the real-versus-synthetic fraud sample comparison.
- The generator learns to map 29-dimensional random noise into synthetic transaction-like vectors.
- The discriminator provides adversarial feedback by learning to separate real fraud samples from generated samples.
- The trained generator successfully produces `1000` synthetic fraud samples in the same feature dimensionality as the original preprocessed fraud records.
- The project demonstrates a generative data balancing workflow for fraud detection datasets where minority-class examples are scarce.
