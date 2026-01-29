# AI-Driven Early Diagnosis of Rare Genetic Disorders Using Genomic Data

## Project Title

AI-Driven Early Diagnosis of Rare Genetic Disorders Using Genomic Data

## Institution

Department of Computer Engineering  
Amrutvahini College of Engineering, Sangamner

## Authors and Contact Information

Ms. Parineeta Pareek        | parineetapareek215@gmail.com  
Mr. Mitanshu Shinde        | mitanshushinde9@gmail.com.com  
Ms. Gauri Mahale           | gaurimahale23@gmail.com  
Mr. Vedant Naikwadi        | nvedant382@gmail.co  

Department of Computer Engineering  
Amrutvahini College of Engineering, Sangamner

## Abstract

Rare genetic disorders present significant diagnostic challenges due to low prevalence, heterogeneous clinical manifestations, and complex genomic architectures. Traditional approaches, including linkage analysis and positional cloning, are often time-consuming, limited in scalability, and insufficient for comprehensive variant interpretation.

Advances in artificial intelligence (AI) and machine learning (ML) enable the integration of genomic and phenotypic data for efficient variant prioritization. This project proposes an interpretable Random Forest–based pipeline for early diagnosis of rare genetic disorders. The system accepts patient Variant Call Format (VCF) files and clinical symptoms encoded using Human Phenotype Ontology (HPO) terms, annotates variants with multi-database resources (ClinVar, gnomAD, dbVar), applies computational pathogenicity predictors (SIFT, PolyPhen-2, CADD), and integrates phenotype similarity scores to generate a ranked list of candidate variants.

The framework emphasizes interpretability, reproducibility, scalability, and clinical decision support, providing a structured roadmap for AI-enabled rare disease genomics.

## Operating System and Python Requirements

This project is officially supported on:

Windows 11  
Python 3.11.x  

Use of other operating systems or Python versions is not guaranteed to work without modification.

## Virtual Environment Setup

From Windows PowerShell or Command Prompt:

python -m venv venv311  
venv311\Scripts\activate  

Upgrade pip:

python -m pip install --upgrade pip  

## Dependency Installation

Install all required libraries using:

pip install flask flask-cors  
pip install torch torchvision torchaudio  
pip install pandas numpy scikit-learn matplotlib seaborn  
pip install tqdm joblib requests  
pip install biopython cyvcf2 pysam openpyxl pyyaml  

## Repository Folder Structure

Genetic_Disorder_Detection/

├── src/
│   ├── server.py
│   ├── train_neural_model.py
│   ├── train_random_forest.py
│   ├── preprocessing/
│   │   ├── parse_vcf.py
│   │   ├── annotate_variants.py
│   │   ├── hpo_encoding.py
│   │   └── feature_builder.py
│
├── data/
│   ├── raw/
│   └── processed/
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

## System Architecture Overview

1. User uploads a VCF file through the REST API  
2. The server parses genomic variants  
3. Gene symbols are extracted  
4. Phenotype features are encoded using HPO  
5. Feature vectors are constructed  
6. Predictive models are executed  
7. Probabilities are aggregated  
8. Diseases are ranked  
9. Genes are mapped  
10. Confidence thresholds applied  
11. Results returned as JSON  

## Running the Inference Server

python src/server.py  

Service URL:

http://127.0.0.1:5000  

## API Endpoint Description

POST /predict_vcf  

Multipart form-data:

file = patient_sample.vcf  

## Response Format

Healthy case:

status: HEALTHY  
variants_processed: count  
affected_genes: empty list  

Disease detected:

status: DISEASE_DETECTED  
top_predictions: ranked disease list  
variants_processed: count  

## Data Preparation Workflow

1. Acquire ClinVar and gnomAD datasets  
2. Collect HPO disease associations  
3. Merge datasets  
4. Remove ultra-rare labels  
5. Encode chromosomes and bases  
6. Encode genotype states  
7. Normalize sequencing depth and quality  
8. Construct ML-ready matrices  
9. Save to data/processed  

## Random Forest Training Methodology

The Random Forest model is used as an interpretable and clinically explainable baseline classifier.

Key strengths:

Ensemble learning  
Resistance to overfitting  
Class balancing  
Feature-importance extraction  
Noise tolerance  

Training stages:

Dataset loading  
Feature validation  
Encoding  
Stratified splitting  
Hyperparameter tuning  
Model fitting  
Evaluation  
Confusion matrix analysis  
Importance ranking  
Model serialization  
Metrics archiving  

## Evaluation Metrics

Accuracy  
Macro Precision  
Macro Recall  
Macro F1-score  
Confusion matrices  
Calibration analysis  

## Reproducibility and Experiment Tracking

Fix seeds  
Log dataset versions  
Store encoders  
Persist models  
Archive metrics  
Track configuration files  
Version experiments  

## Ethical and Clinical Disclaimer

This system is for research use only. It must not be deployed clinically without regulatory approval. Predictions should never replace professional genetic consultation.

## Future Extensions

Integration of large exome datasets  
HPO similarity engines  
Gradient boosting models  
Deep transformers  
Model ensembles  
Visualization dashboards  
Clinical reporting systems  

## License

Academic and educational use only.

End of README
