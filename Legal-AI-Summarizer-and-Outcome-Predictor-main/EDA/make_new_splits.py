import pandas as pd
from sklearn.model_selection import train_test_split

# Load FULL cleaned dataset
df = pd.read_csv("scraped_it_cases/it_act_trainable.csv")

# Keep only binary PETITIONER vs RESPONDENT
df = df[df["who_won"].isin(["PETITIONER", "RESPONDENT"])].copy()
df["target"] = (df["who_won"] == "PETITIONER").astype(int)

# NEW SPLIT: 70% train / 15% valid / 15% test
train_df, temp_df = train_test_split(
    df, test_size=0.30, random_state=42, stratify=df["target"]
)
valid_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=42, stratify=temp_df["target"]
)

# Save splits
train_df.to_csv("scraped_it_cases/train.csv", index=False)
valid_df.to_csv("scraped_it_cases/valid.csv", index=False)
test_df.to_csv("scraped_it_cases/test.csv", index=False)

print("✅ New balanced splits created!\n")
print("Train class counts:\n", train_df["target"].value_counts())
print("\nValid class counts:\n", valid_df["target"].value_counts())
print("\nTest class counts:\n", test_df["target"].value_counts())
