python main.py \
  --backbone resnet18 \
  --enc_layers 2 \
  --dec_layers 2 \
  --hidden_dim 16 \
  --nheads 2 \
  --num_queries 5 \
  --device cuda \
  --dataset_file web \
  --inference_image_path ./datasets/data_web/train2017/000000000001.jpg \
  --weights_path ./output_web/checkpoint0299.pth \
  --inference_annotations_file ./datasets/data_web/annotations/classes.json