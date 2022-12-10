# Airbnb Reviews per Month Regression

Contributors: Lauren Zung

A mini project completed during the Fall 2022 session of DSCI 573 (Feature and Model Selection).

## Usage

To replicate this analysis, you will first need to clone the repo.  You can do so with the following command line (terminal) commands.

```bash
# clone the repo
git clone https://github.com/lzung/airbnb_reviews.git

# change working directory to the root of the repository
cd airbnb_reviews
```

First, create and activate the required virtual environment with conda at the command line as follows:

```bash
conda env create -f environment.yaml
conda activate airbnb_reviews
```

Then, run the following command at the command line (terminal) to reset the repository to a clean state, with no intermediate or results files:


```bash
make clean
```

Finally, run the following command to replicate the analysis:

```bash
make all
```

## Makefile Dependency Diagram
![Makefile](Makefile.png)