# AI-Driven Early Diagnosis of Rare Genetic Disorders Using Genomic Data

## Project Title

AI-Driven Early Diagnosis of Rare Genetic Disorders Using Genomic Data

## Institution

Department of Computer Engineering  
Amrutvahini College of Engineering, Sangamner

## Authors and Contact Information

```
Ms. Parineeta Pareek         parineetapareek215@gmail.com
Mr. Mitanshu Shinde          mitanshushinde9@gmail.com.com
Ms. Gauri Mahale             gaurimahale23@gmail.com
Mr. Vedant Naikwadi          nvedant382@gmail.co

Department of Computer Engineering
Amrutvahini College of Engineering, Sangamner
```

## Abstract

Rare genetic disorders present significant diagnostic challenges due to low prevalence, heterogeneous clinical manifestations, and complex genomic architectures. Traditional approaches, including linkage analysis and positional cloning, are often time-consuming, limited in scalability, and insufficient for comprehensive variant interpretation.

Advances in artificial intelligence (AI) and machine learning (ML) enable the integration of genomic and phenotypic data for efficient variant prioritization. This project proposes an interpretable Random Forest–based pipeline for early diagnosis of rare genetic disorders. The system accepts patient Variant Call Format (VCF) files and clinical symptoms encoded using Human Phenotype Ontology (HPO) terms, annotates variants with multi-database resources (ClinVar, gnomAD, dbVar), applies computational pathogenicity predictors (SIFT, PolyPhen-2, CADD), and integrates phenotype similarity scores to generate a ranked list of candidate variants.

The framework emphasizes interpretability, reproducibility, scalability, and clinical decision support, providing a structured roadmap for AI-enabled rare disease genomics.

## Operating System and Python Requirements

```
Windows 11
Python 3.11.x
```

## Virtual Environment Setup

```
python -m venv venv311
venv311\Scripts\activate
```

Upgrade pip:

```
python -m pip install --upgrade pip
```

## Dependency Installation

```
pip install flask flask-cors
pip install torch torchvision torchaudio
pip install pandas numpy scikit-learn matplotlib seaborn
pip install tqdm joblib requests
pip install biopython cyvcf2 pysam openpyxl pyyaml
```

## Repository Folder Structure

```
Genetic_Disorder_Detection/
├── data/
│   ├── raw/
│   │   ├── clinvar.vcf         # ClinVar VCF (full dataset)
│   │   └── phenotype.hpoa      # HPO ↔ OMIM disease file
│   │
│   ├── processed/
│   │   ├── clinvar_parsed.csv
│   │   ├── disease_hpo_map.csv
│   │   └── training_vectors.pt
├── src/
│   ├── app.py.py
│   ├── train.py
│   ├── utils.py
├── preprocessing/
│   │   ├── Build_csv.py
│   │   ├── clean_csv.py
│   │   ├── compress_vcf.py
│   │   └── intro.py
│
│
├── models/
│
├── uploads/
│
├── Statistics/
│   └── diagrams/
│
├── requirements.txt
├── README.md
└── .gitignore
```

## System Architecture Overview

1. User uploads a VCF file through the REST API  
2. The server parses genomic variants  
3. Gene symbols are extracted from annotations  
4. Phenotype features are encoded using HPO  
5. Feature vectors are constructed  
6. Predictive models are executed  
7. Probabilities are aggregated across variants  
8. Diseases are ranked  
9. Associated genes are mapped  
10. Confidence thresholds are applied  
11. Results are returned as JSON  

## Running the Inference Server

```
python src/train.py
```
## Running the Inference Server

```
python src/app.py
```

Service URL:

```
http://127.0.0.1:5000
```

## API Endpoint Description

```
POST /predict_vcf
```

Multipart form-data:

```
file = patient_sample.vcf
```

## Response Format

Healthy case:

```
status: HEALTHY
variants_processed: count
affected_genes: []
```

Disease detected:

```
status: DISEASE_DETECTED
top_predictions: ranked disease list
variants_processed: count
```

## Data Preparation Workflow

1. Acquire ClinVar and gnomAD datasets  
2. Collect HPO disease associations  
3. Merge variant and phenotype tables  
4. Remove ultra-rare labels  
5. Encode chromosomes and nucleotide bases  
6. Encode genotype states  
7. Normalize sequencing depth and genotype quality  
8. Construct machine-learning matrices  
9. Save datasets into data/processed  

## Random Forest Training Methodology

The Random Forest model is used as an interpretable and clinically explainable baseline classifier.

Key strengths:

• Ensemble learning  
• Resistance to overfitting  
• Class balancing  
• Feature-importance extraction  
• Noise tolerance  

Training stages:

• Dataset loading  
• Feature validation  
• Encoding  
• Stratified splitting  
• Hyperparameter tuning  
• Model fitting  
• Evaluation  
• Confusion matrix analysis  
• Feature-importance ranking  
• Model serialization  
• Metrics archiving  

## Evaluation Metrics

Accuracy  
Macro Precision  
Macro Recall  
Macro F1-score  
Confusion matrices  
Calibration analysis  

## Reproducibility and Experiment Tracking

Fix random seeds  
Log dataset versions  
Store encoders  
Persist trained models  
Archive metrics  
Track configuration files  
Version experiments  

## Ethical and Clinical Disclaimer

This system is intended strictly for academic research use. It must not be deployed clinically without regulatory approval. Predictions must never replace professional genetic counseling.


## License

Academic and educational use only.

End of README



