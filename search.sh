#!/bin/bash

python -u ./search_incremental.py | tee logs/search_incremental.log
python -u ./search_random.py | tee logs/search_random.log
python -u ./search_random.py | tee logs/search_common.log
python -u ./search_model.py | tee logs/search_model.log
