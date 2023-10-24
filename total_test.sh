python Our_test.py --net "dualstream_v1" --pred_dir "dualstream_v1/" --model_dir "dualstream_v1/train.pth"
python Our_test.py --net "dualstream_v2" --pred_dir "dualstream_v2/" --model_dir "dualstream_v2/train.pth"

python Our_test.py --net "MS_guide_attention" --pred_dir "MS_guide_attention/" --model_dir "MS_guide_attention/train.pth"
python Our_test.py --net "pranet" --pred_dir "pranet/" --model_dir "pranet/train.pth"
python Our_test.py --net "pyramid" --pred_dir "pyramid/" --model_dir "pyramid/train.pth"
python Our_test.py --net "dualstream_CBAM" --pred_dir "dualstream_CBAM/" --model_dir "dualstream_CBAM/train.pth"
python Our_test.py --net "dualstream_SE" --pred_dir "dualstream_SE/" --model_dir "dualstream_SE/train.pth"
python Our_test.py --net "dualstream_Trip" --pred_dir "dualstream_Trip/" --model_dir "dualstream_Trip/train.pth"

#python Our_test.py --net "dualstream_v1_transformer" --pred_dir "dualstream_v1_transformer/" --model_dir ""
#python Our_test.py --net "dualstream_v2_transformer" --pred_dir "dualstream_v2_transformer/" --model_dir ""
#python Our_test.py --net "dualstream_v2_resnet" --pred_dir "dualstream_v2_resnet/" --model_dir ""
