OUT_DIR="/home/pikachu/Project/Farscape/zrl/arichitecture/outputs"

python infer.py \
  --data_root /home/pikachu/Project/Farscape/zrl/arichitecture/datasets/scene_002 \
  --sam2_cfg "/home/pikachu/Project/Farscape/zrl/arichitecture/modules/sam2/configs/sam2_1/sam2_1_hiera_l.yaml" \
  --sam2_ckpt "/home/pikachu/Project/Farscape/zrl/arichitecture/modules/sam2/checkpoints/sam2.1_hiera_large.pt" \
  --device cuda \
  --output_dir "$OUT_DIR" \
  --vis_vggt 
 