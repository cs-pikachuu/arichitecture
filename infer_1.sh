python /home/pikachu/Project/Farscape/zrl/arichitecture/infer.py \
  --data_root /path/to/images \
  --sam2_cfg sam2_1/sam2_1_hiera_l \
  --sam2_ckpt /home/pikachu/Project/Farscape/zrl/arichitecture/modules/sam2/checkpoints/sam2.1_hiera_large.pt \
  --vggt_ckpt /home/pikachu/Project/Farscape/zrl/arichitecture/modules/vggt/checkpoints/model.pt \
  --device cuda
