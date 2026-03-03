---
layout: post
title: "Clinical AI Basics for ML People"
date:   2026-03-13 00:00:00 -0700
description: "An introduction to medical imaging and electronic health records for AI/ML practitioners entering clinical AI."
tags: [Medical Imaging, Clinical Data, EHR, DICOM, Radiology]
categories: [AI, Biomedical Science]
---
* TOC
{:toc}

This is the companion post to [Biomedical Basics for AI/ML People](/2026/02/12/biomedical-basics/), which covers molecular biology, genomics, proteins, and drug discovery. Here we focus on the **clinical practice** side — the data that comes from hospitals and patient care.

We'll cover two major areas where AI is making an impact: **medical imaging** and **electronic health records (EHR)**.


# Medical Imaging

## Imaging Modalities

| Modality | What It Captures | Typical Data Format |
|----------|-----------------|-------------------|
| **X-ray** | 2D projection of dense structures (bones, lungs). | 2D grayscale image |
| **CT** (Computed Tomography) | 3D volume from multiple X-ray angles. Good for anatomy. | 3D volume (stack of 2D slices) |
| **MRI** (Magnetic Resonance Imaging) | 3D soft tissue detail using magnetic fields. Multiple contrast types (T1, T2, FLAIR). | 3D volume, often multi-channel |
| **Ultrasound** | Real-time imaging using sound waves. Common for obstetrics, cardiac. | 2D or 3D, often video |
| **PET** (Positron Emission Tomography) | Metabolic activity using radioactive tracers. Often combined with CT (PET/CT). | 3D volume, low resolution |
| **Histopathology** | Microscopic images of tissue (biopsies). Stained and digitized as whole slide images. | Very large 2D images (gigapixel-scale) |
| **Fundoscopy / OCT** | Retinal imaging. Fundus photos (2D) or OCT scans (3D cross-sections). | 2D or 3D |

## Key Terms

| Term | What It Is |
|------|-----------|
| **DICOM** | The standard file format for medical images. Contains pixel data + rich metadata (patient info, acquisition parameters). |
| **Segmentation** | Delineating structures (organs, tumors) in an image. |
| **CAD** (Computer-Aided Detection/Diagnosis) | Systems that flag suspicious regions for a radiologist to review. |
| **ROI** (Region of Interest) | A marked area in an image for analysis. |
| **Whole Slide Image (WSI)** | A digitized pathology slide, often 100,000 × 100,000 pixels. |
| **Radiomics** | Extracting quantitative features from medical images. |

## Where AI Fits In

- **Classification**: Disease detection from images (e.g., diabetic retinopathy from fundus photos, pneumonia from chest X-rays).
- **Segmentation**: Organ and tumor delineation in CT/MRI (U-Net and variants dominate).
- **Detection**: Localizing lesions, nodules, or fractures (object detection on medical images).
- **Registration**: Aligning images from different time points or modalities.
- **Report generation**: Generating radiology reports from images (vision-language models).
- **Digital pathology**: Classifying cancer subtypes and grading from whole slide images (multiple instance learning).


# Clinical Data & Electronic Health Records

## Key Terms

| Term | What It Is |
|------|-----------|
| **EHR** (Electronic Health Record) | A patient's digital medical record: diagnoses, medications, labs, notes, imaging. |
| **ICD codes** | Standardized codes for diagnoses (e.g., ICD-10: E11.9 = Type 2 diabetes). ~70,000 codes. |
| **CPT codes** | Codes for medical procedures. |
| **Lab values** | Numeric test results (blood glucose, hemoglobin, etc.) with reference ranges. |
| **Clinical notes** | Free-text notes written by clinicians. Rich but unstructured. |
| **FHIR** | A modern standard for exchanging healthcare data via APIs. |
| **Cohort** | A group of patients defined by shared characteristics for a study. |
| **Phenotyping** | Identifying patients with a specific condition from EHR data (not always straightforward). |
| **De-identification** | Removing personally identifiable information from medical data. Required for research use. |

## Key Challenges for ML

- **Data access**: Medical data is heavily regulated (HIPAA in the US, GDPR in the EU). Getting access is hard and slow.
- **Missing data**: Clinical data is collected for care, not research. Missingness is the norm, not the exception — and it's often informative (a missing test may mean the doctor didn't think it was needed).
- **Irregular time series**: Vital signs and labs are recorded at irregular intervals, unlike the uniform grids ML models typically expect.
- **Label noise**: Diagnoses in EHRs are billing codes, not ground-truth labels. A code may be present for "rule-out" purposes.
- **Class imbalance**: Rare diseases are, by definition, rare. Most clinical prediction tasks are highly imbalanced.
- **Distribution shift**: Patient populations, clinical practices, and coding conventions vary across hospitals.

## Where AI Fits In

- **Clinical prediction**: Readmission, mortality, sepsis, and deterioration risk scoring from EHR time series.
- **NLP on clinical notes**: Named entity recognition, relation extraction, and summarization (Med-BERT, Clinical-T5).
- **Medical coding**: Auto-assigning ICD/CPT codes to clinical encounters.
- **Federated learning**: Training models across hospitals without sharing patient data.
- **Foundation models**: Large models pre-trained on broad clinical data, fine-tuned for specific tasks (Med-PaLM, BioGPT).


# Clinical Databases & Benchmarks

| Resource | What It Contains |
|----------|-----------------|
| **MIMIC** | De-identified ICU patient records (EHR benchmark) |
| **CheXpert / NIH ChestX-ray14** | Chest X-ray datasets with labels |
| **TCGA** (The Cancer Genome Atlas) | Multi-omics cancer data (bridges research and clinical) |
| **UK Biobank** | Large-scale health and genomics data (~500K participants) |
