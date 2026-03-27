echo "Running all training scripts for GSM8K curriculum learning with QLoRA..."

python sft.py -c configs/olmoe/train/gsm8k_no_curriculum_qlora.yaml
python sft.py -c configs/olmoe/train/gsm8k_model_curriculum_qlora.yaml
python sft.py -c configs/olmoe/train/gsm8k_data_curriculum_qlora.yaml
python sft.py -c configs/olmoe/train/gsm8k_full_curriculum_qlora.yaml

echo "Training completed. Starting evaluation..."

python eval.py -c configs/olmoe/eval/gsm8k_no_curriculum_qlora.yaml
python eval.py -c configs/olmoe/eval/gsm8k_model_curriculum_qlora.yaml
python eval.py -c configs/olmoe/eval/gsm8k_data_curriculum_qlora.yaml
python eval.py -c configs/olmoe/eval/gsm8k_full_curriculum_qlora.yaml


echo "Running all training scripts for ARC curriculum learning with QLoRA..."

python sft.py -c configs/olmoe/train/arc_no_curriculum_qlora.yaml
python sft.py -c configs/olmoe/train/arc_model_curriculum_qlora.yaml
python sft.py -c configs/olmoe/train/arc_data_curriculum_qlora.yaml
python sft.py -c configs/olmoe/train/arc_full_curriculum_qlora.yaml

echo "Training completed. Starting evaluation..."

python eval.py -c configs/olmoe/eval/arc_no_curriculum_qlora.yaml
python eval.py -c configs/olmoe/eval/arc_model_curriculum_qlora.yaml
python eval.py -c configs/olmoe/eval/arc_data_curriculum_qlora.yaml
python eval.py -c configs/olmoe/eval/arc_full_curriculum_qlora.yaml


echo "Running all training scripts for sciq curriculum learning with QLoRA..."

python sft.py -c configs/olmoe/train/sciq_no_curriculum_qlora.yaml
python sft.py -c configs/olmoe/train/sciq_model_curriculum_qlora.yaml
python sft.py -c configs/olmoe/train/sciq_data_curriculum_qlora.yaml
python sft.py -c configs/olmoe/train/sciq_full_curriculum_qlora.yaml

echo "Training completed. Starting evaluation..."

python eval.py -c configs/olmoe/eval/sciq_no_curriculum_qlora.yaml
python eval.py -c configs/olmoe/eval/sciq_model_curriculum_qlora.yaml
python eval.py -c configs/olmoe/eval/sciq_data_curriculum_qlora.yaml
python eval.py -c configs/olmoe/eval/sciq_full_curriculum_qlora.yaml

echo "All training and evaluation scripts for GSM8K, ARC, and SciQ curriculum learning with QLoRA have been executed."