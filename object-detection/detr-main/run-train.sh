python main.py \
  --backbone resnet18 \
  --enc_layers 2 \
  --dec_layers 2 \
  --hidden_dim 16 \
  --nheads 2 \
  --num_queries 5 \
  --device cpu \
  --coco_path ./datasets/data/ \
  --output_dir ./output/
