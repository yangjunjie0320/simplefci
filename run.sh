export PYTHONPATH=$(pwd)/src/:$PYTHONPATH

# pytest
python ./test/test_slater_condon.py
python ./test/test_direct.py
