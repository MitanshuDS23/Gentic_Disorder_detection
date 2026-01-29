import pandas as pd

INPUT_CSV = "data/processed/clinvar_hpoa_merged.csv"
OUTPUT_CSV = "data/processed/clinvar_hpoa_merged_cleaned.csv"

def main():
    print("Loading CSV...")
    df = pd.read_csv(INPUT_CSV)

    original_count = len(df)

    # Remove rows where disease_id equals disease_name
    df_cleaned = df[df["disease_id"] != df["disease_name"]]

    removed = original_count - len(df_cleaned)

    print(f"Original rows : {original_count}")
    print(f"Removed rows  : {removed}")
    print(f"Final rows    : {len(df_cleaned)}")

    # Save cleaned CSV
    df_cleaned.to_csv(OUTPUT_CSV, index=False)

    print(f"\nCleaned file saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
