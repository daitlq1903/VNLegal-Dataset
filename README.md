# VNLegal-Dataset for Vietnamese Legal

> **⚠️ IMPORTANT NOTICE:**
> This repository is currently under preparation. The complete dataset, scripts, ... are being actively updated and will be made FULLY PUBLIC immediately upon the official publication of the related paper.

## 🌳 Planned Directory Structure (To be released)
```text
VNLegal-Dataset/
├── data/
│   ├── ContinueTraining           
│   ├── VNCivil-Pref           
│   ├── VNCivil-SFT    
│   ├── cite_evidence
│   └── ...                    
├── script/
│   ├── drl_model.py  
│   ├── generate_evol_data.py        
│   ├── evaluate_citer.py      
│   └── ...                        
└── README.md


## 🚀 How to Run the Code (Execution Instructions)

> **⚠️ NOTE:** 
> *Detailed command-line instructions, arguments, and hyperparameter configurations for each step are currently being populated.

### 1. Environment Setup and Dependencies
*(To be updated: Instructions for setting up the Python environment, installing requirements, and configuring FAISS for retrieval).*

### 2. Phase 1: Synthetic Data Generation
*(To be updated: Commands to run `generate_evol_data.py` over the seed legal provisions, apply the IRAC constraints, and execute the LLM-as-a-Judge filtering).*

### 3. Phase 2: Supervised Fine-Tuning
*(To be updated: Instructions for executing `train_sft.py` using LoRA adapters on Llama-3.1-8B and Qwen-2.5-7B).*

### 4. Reward Shaping
*(To be updated: Commands to calculate Semantic Relevance and NLI-based Faithfulness (Mean-Max penalty) using the customized XLM-RoBERTa cross-encoder).*

### 5. Phase 3: Context-DPO
*(To be updated: Instructions for running `train_dpo.py` using the autonomously generated contrastive preference pairs).*

### 6. Evaluation Pipeline
*(To be updated: Scripts to reproduce the automated metrics including CitER, BERTScore, and RAGAS framework).*
