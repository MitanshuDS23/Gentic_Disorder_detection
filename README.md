# AI-Driven Early Diagnosis of Rare Genetic Disorders Using Genomic Data

---

## Project Title

AI-Driven Early Diagnosis of Rare Genetic Disorders Using Genomic Data

---

## Institution

Department of Computer Engineering  
Amrutvahini College of Engineering, Sangamner

---

## Authors and Contact Information

Ms. Parineeta Pareek 
Mr. Mitanshu Shinde  
Ms. Gauri Mahale  
Mr. Vedant Naikwadi  

Emails:

parineetapareek215@gmail.com
mitanshushinde9@gmail.com.com 
gaurimahale23@gmail.com   
nvedant382@gmail.co  

---

## Abstract

Rare genetic disorders present significant diagnostic challenges due to low prevalence, heterogeneous clinical manifestations, and complex genomic architectures. Traditional approaches, including linkage analysis and positional cloning, are often time-consuming, limited in scalability, and insufficient for comprehensive variant interpretation.

Advances in artificial intelligence (AI) and machine learning (ML) enable the integration of genomic and phenotypic data for efficient variant prioritization. This project proposes an interpretable Random Forest–based pipeline for early diagnosis of rare genetic disorders. The system accepts patient Variant Call Format (VCF) files and clinical symptoms encoded using Human Phenotype Ontology (HPO) terms, annotates variants with multi-database resources (ClinVar, gnomAD, dbVar), applies computational pathogenicity predictors (SIFT, PolyPhen-2, CADD), and integrates phenotype similarity scores to generate a ranked list of candidate variants.

The framework emphasizes interpretability, reproducibility, scalability, and clinical decision support, providing a structured roadmap for AI-enabled rare disease genomics.

---

## Operating System and Python Requirements

This project is officially supported on:

Windows 11  
Python 3.11.x  

Use of other operating systems or Python versions is not guaranteed to work without modification.

---

## Virtual Environment Setup

From Windows PowerShell or Command Prompt:

python -m venv venv311  
venv311\Scripts\activate  

Upgrade pip:

python -m pip install --upgrade pip  

---

## Dependency Installation

Install all required libraries using:

pip install flask flask-cors  
pip install torch torchvision torchaudio  
pip install pandas numpy scikit-learn matplotlib seaborn  
pip install tqdm joblib requests  
pip install biopython cyvcf2 pysam openpyxl pyyaml  

---

## Repository Folder Structure

The project follows a structured and modular layout:

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
├── src/
│   ├── train.py                # Training pipeline (CUDA)
│   ├── app.py                  # Flask API
│   ├── parser.py               # VCF + HPOA parsing logic
│   └── utils.py                # Shared utilities
│ 
├── requirements.txt  
├── README.md  
└── .gitignore  

---

## System Architecture Overview

1. User uploads a VCF file through the REST API.  
2. The server parses all genomic variants.  
3. Gene symbols are extracted from annotations.  
4. Clinical features are encoded using HPO terms.  
5. Numerical feature vectors are generated.  
6. Prediction models are executed.  
7. Probabilities are aggregated across variants.  
8. Top-ranked diseases are selected.  
9. Associated genes are retrieved.  
10. Confidence thresholds are applied.  
11. Results are returned in JSON format.

---

## Running the Inference Server

To launch the Flask-based backend:

python src/server.py  

The service will be available at:

http://127.0.0.1:5000  

---

## API Endpoint Description

POST /predict_vcf  

The endpoint accepts a multipart-form request containing:

Key: file  
Value: patient_sample.vcf  

---

## Response Format

Healthy case:

status: HEALTHY  
variants_processed: count  
affected_genes: empty list  

Disease detected:

status: DISEASE_DETECTED  
top_predictions: ranked disease list  
variants_processed: count  

---

## Data Preparation Workflow

1. Download curated variant datasets such as ClinVar.  
2. Obtain phenotype-disease associations from HPOA.  
3. Merge variant and phenotype tables using disease identifiers.  
4. Remove ultra-rare classes to stabilize training.  
5. Encode categorical genomic attributes numerically.  
6. Normalize sequencing depth and genotype quality.  
7. Construct final machine-learning feature matrices.  
8. Store processed datasets inside data/processed.

---

## Random Forest Training Methodology

The Random Forest model is used as a transparent and interpretable baseline classifier. It is trained using thousands of labeled genomic variants and disease associations.

Key characteristics:

• Ensemble of decision trees  
• Bagging-based variance reduction  
• Class-weight balancing for rare diseases  
• Feature importance extraction  
• Robust to noisy genomic signals  

Training pipeline stages include:

• Dataset loading and validation  
• Feature extraction  
• Label encoding  
• Stratified train-validation split  
• Hyperparameter optimization  
• Model fitting  
• Performance evaluation  
• Confusion matrix analysis  
• Feature-importance ranking  
• Serialization of trained models  
• Archival of evaluation statistics  

---

## Evaluation Metrics

Models are evaluated using:

Accuracy  
Macro Precision  
Macro Recall  
Macro F1-score  
Per-class confusion matrices  
Probability calibration curves  

---

## Reproducibility and Experiment Tracking

To ensure reproducible research:

• Fix random seeds during training  
• Record dataset versions  
• Archive preprocessing scripts  
• Save encoders and label mappings  
• Persist trained models  
• Track metrics across experiments  
• Version-control configurations  

---


---

## Future Extensions

Planned future work includes:

• Integration of Symptom sequencing datasets  
• HPO-based phenotype similarity models  
• XGBoost and LightGBM classifiers  
• Ensemble-based decision fusion  
• Web dashboard visualization  
• Clinical reporting modules  

---

## License



---

End of README.
