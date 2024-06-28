python main.py \
  --backbone resnet18 \
  --enc_layers 2 \
  --dec_layers 2 \
  --hidden_dim 16 \
  --nheads 2 \
  --num_queries 5 \
  --device cpu \
  --epochs 300 \
  --coco_path ./datasets/data_web/ \
  --output_dir ./output_web/
