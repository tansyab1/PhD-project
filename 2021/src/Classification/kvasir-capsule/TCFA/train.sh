python3 TCFA.py \
    --data_root ./dataForVisualization/UI_var/ \
    --ref_root ./dataForVisualization/Ref/ \
    --pkl_root ./dataForVisualization/UI_var/ui_dict.pkl \
    --out_dir ./output/ \
    --tensorboard_dir ./tensorboard/ \
    --num_workers 4 \
    --bs 32 \
    --num_epochs 50 \
    --action "train"