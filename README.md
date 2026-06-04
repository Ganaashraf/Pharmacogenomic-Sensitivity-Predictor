# Pharmacogenomic-Sensitivity-Predictor
The Pharmacogenomic Sensitivity Predictor is a machine learning application that predicts whether a cancer cell line will be Sensitive or Resistant to a given drug compound, addressing one of precision oncology's core challenges ,the fact that tumors with similar clinical profiles can respond drastically differently to the same treatment due to underlying genetic variation. Built on the GDSC (Genomics of Drug Sensitivity in Cancer) dataset, the pipeline processes pharmacokinetic features such as IC50 values and AUC scores alongside drug target pathway annotations through a preprocessing stage of standard scaling, feature selection, and label encoding before passing them to a trained classifier. The results are served through an interactive Streamlit web application that requires no coding knowledge, making the tool accessible to researchers, clinicians, and students alike — enabling faster pre-clinical drug screening, biomarker discovery, and data-driven hypothesis generation in cancer research.
# datset'source
https://www.kaggle.com/datasets/samiraalipour/genomics-of-drug-sensitivity-in-cancer-gdsc?resource=download
# APP’s link:
https://pharmacogenomic-sensitivity-predictor-nxv9spuuhecfyzvqdyxhug.streamlit.app/
# video's link
https://drive.google.com/file/d/1DStIdecY6obOjjWQwX5yP-ZVjawS1deD/view?usp=sharing
# Required Libraries :
streamlit — builds and runs the interactive web application interface
pandas — loads and manipulates the input data into structured dataframes
numpy — handles numerical computations and array operations
scikit-learn==1.6.1 — provides the ML classifier, StandardScaler for feature normalization, SelectKBest for feature selection, and LabelEncoder for pathway encoding
pickle (built-in) — loads the pre-trained model and fitted preprocessing artifacts (.pkl files) at inference time
