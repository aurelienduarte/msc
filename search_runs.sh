#!/bin/bash

for index in {0..9};do
    echo $index
    # python -u ./search_incremental.py "runs/${index}_" | tee runs/${index}_search_incremental.log
    # python -u ./search_random.py "runs/${index}_" | tee runs/${index}_search_random.log
    # python -u ./search_model.py "runs/${index}_" | tee runs/${index}_search_model.log
    python -u ./search_common.py "runs/${index}_" | tee runs/${index}_search_common.log
done


# python -u ./search_incremental.py | tee logs/search_incremental.log
# python -u ./search_random.py | tee logs/search_random.log

# python -u ./search_common.py | tee logs/search_common.log

# python -u ./search_model.py | tee logs/search_model.log

for index in {7..9};do
    echo $index
    # python -u ./search_incremental.py "runs/${index}_" | tee runs/${index}_search_incremental.log
    # python -u ./search_random.py "runs/${index}_" | tee runs/${index}_search_random.log
    python -u ./search_model.py "runs/${index}_" | tee runs/${index}_search_model.log
    # python -u ./search_common.py "runs/${index}_" | tee runs/${index}_search_common.log
done