#!/bin/sh
# Downloads every dataset from Hugging Face mirrors into data/raw/ (no account needed).
# File names match what the code expects. Parquet files are only needed for optional experiments.
set -eu

OUT="${1:-data/raw}"
HF="https://huggingface.co/datasets"
mkdir -p "$OUT"

get() {
  [ -s "$OUT/$2" ] || curl -fL --retry 3 -o "$OUT/$2" "$HF/$1"
}

get theman10/paysim/resolve/main/paysim.csv paysim.csv
get eexzzm/IBM-Transactions-for-Anti-Money-Laundering-HI-Small-Trans/resolve/main/HI-Small_Trans.csv.zip HI-Small_Trans.csv.zip
[ -s "$OUT/HI-Small_Trans.csv" ] || unzip -o -d "$OUT" "$OUT/HI-Small_Trans.csv.zip"

get Shade63/scam-classification-multiclass/resolve/main/sentinel_dataset_multiclass_clean.csv scam_multiclass.csv
get ysangam/Indian_Cyber_Scam_PhoneCall_Hinglish_Dataset/resolve/main/India_Cyber_Scam_Hinglish_Dataset.csv scam_calls_hinglish.csv

get CloveAI/india-spam-sms/resolve/main/data/train-00000-of-00001.parquet india_spam_sms.parquet
get ucirvine/sms_spam/resolve/main/plain_text/train-00000-of-00001.parquet sms_spam.parquet
get james-burton/fake_job_postings2/resolve/main/data/train-00000-of-00001-7b14dd51a92a5dd5.parquet emscad_train.parquet
get james-burton/fake_job_postings2/resolve/main/data/validation-00000-of-00001-a8bf33cea73dd9d4.parquet emscad_validation.parquet
get james-burton/fake_job_postings2/resolve/main/data/test-00000-of-00001-0cefc8bdf8f687b8.parquet emscad_test.parquet
