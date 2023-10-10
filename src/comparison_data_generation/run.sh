pip install transformers==4.31.0
pip install tokenizers==0.13.3
pip install deepspeed==0.10.0
pip install accelerate -U


python main.py --model_type ${1} --id ${2}

