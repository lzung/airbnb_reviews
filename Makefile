# Airbnb Reviews per Month Regression
# Authors: Lauren Zung
# Date: December 10, 2022

# Run the analysis
all : eda/ results/

# Preprocess raw data
data/processed/ : src/preprocess_data.py data/raw/AB_NYC_2019.csv
	python src/preprocess_data.py --input_path=data/raw/ --output_path=data/processed/

# Create eda charts and save to directory
eda/ : src/reviews_eda.py data/processed/
	python src/reviews_eda.py --input_path=data/processed/ --output_path=eda/

# Perform model selection/tuning and save results
results/ : src/reviews_model_analysis.py data/processed/
	python src/reviews_model_analysis.py --input_path=data/processed/ --output_path=results/

# Remove intermediate and results files
clean:
	rm -f data/processed/*
	rm -f eda/*
	rm -f results/*
