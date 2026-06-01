#!/bin/bash


cd "$(dirname "$0")/.."

echo "===== M=3 One-to-Many ====="
python gen_personalized_headline.py 3

echo "===== Done ====="
