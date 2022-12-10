# Airbnb Reviews per Month Regression
# Authors: Lauren Zung
# Date: December 10, 2022

# Run the analysis
all : results/final

# Preprocess raw data
data/processed/ : src/preprocess_data.py data/raw/AB_NYC_2019.csv
	python src/preprocess_data.py --input_path=data/raw/ --output_path=data/processed/

# Create eda charts and save to directory
eda/ : src/reviews_eda.py data/processed/
	python src/reviews_eda.py --input_path=data/processed/ --output_path=eda/

# Perform model selection and save results
results/model_selection/ : src/model_selection.py data/processed/
	python src/model_selection.py --data_path=data/processed/ --output_path=results/model_selection/

# Perform model tuning and save results
results/final/ : src/tune_model.py data/processed/
	python src/tune_model.py --data_path=data/processed/ --output_path=results/model_tuning/

# Remove intermediate and results files
clean:
	rm -f data/raw/*
	rm -f data/processed/*
	rm -f eda/figures/*
	rm -f eda/tables/*
	rm -f results/*