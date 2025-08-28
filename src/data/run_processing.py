import numpy as np
import pandas as pd
import logging
from pathlib import Path
import os
import argparse


# set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("data-processor")


def load_data(file_path):
    """
    Load data from a CSV file.
    """
    logger.info(f"Loading data from {file_path}")
    return pd.read_csv(file_path)

def clean_data(df):
    """
    Clean the dataset by handling missing values and outliers
    """
    logger.info("Cleaning dataset")

    # Make a copy to avoid modifying the original dataset
    df_cleaned = df.copy()

    # Handle missing values
    for column in df_cleaned.columns:
        missing_count = df_cleaned[column].isnull().sum()
        if missing_count > 0 :
            logger.info(f"Found {missing_count} missing values in {column}")

            # For numeric columns, fill with median
            if pd.api.types.is_numeric_dtype(df_cleaned[column]):
                median_value = df_cleaned[column].median()
                df_cleaned[column] = df_cleaned[column].fillna(median_value)
                logger.info(f"Filled missing values in {column} with median {median_value}")
            
            # For categorical columns, fill with mode
            else:
                mode_value = df_cleaned[column].mode()[0]
                df_cleaned[column] = df_cleaned[column].fillna(mode_value)
                logger.info(f"Filled missing values in {column} with mode {mode_value}")

    # Handle outliers using IQR method
    Q1 = df_cleaned.quantile(0.25)
    Q3 = df_cleaned.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter out data
    outliers = (df_cleaned < lower_bound) | (df_cleaned > upper_bound)
    outlier_rows = df_cleaned[outliers.any(axis=1)]

    if not outlier_rows.empty:
        logger.info(f"Found {len(outlier_rows)} outliers in data")
        mask = ~((df_cleaned < lower_bound) | (df_cleaned > upper_bound)).any(axis=1)
        # Filter the dataframe
        df_cleaned = df_cleaned[mask]
        logger.info(f"Removed outliers. New dataset shape: {df_cleaned.shape}")
            
    return df_cleaned


def process_data(input_file, output_file):
    """
    Full data processing pipeline
    """
    
    # Load data
    df = load_data(input_file)
    logger.info(f"Loaded data with shape: {df.shape}")

    # Clean data
    df_cleaned = clean_data(df)

    # save processed data
    df_cleaned.to_csv(output_file, index=False)
    logger.info(f"Saved processed data to {output_file}")

    return df_cleaned


if __name__ == "__main__":

    Parser = argparse.ArgumentParser(description="input data and output data")
    Parser.add_argument("-i", "--input_file", default="data/processed/data_scientists_featurs_snp.csv")
    Parser.add_argument("-o", "--output_file", default="data/processed/cleaned_snp_data.csv")

    args = Parser.parse_args()
    # Example usage
    
    process_data(input_file=args.input_file,
                 output_file=args.output_file)


        



