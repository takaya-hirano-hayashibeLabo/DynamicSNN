TIMESCALES=(
    "1,1"
    "1,5"
    "5,1"
    "5,10"
    "10,5"
)

for TIMESCALE in "${TIMESCALES[@]}"; do
    IFS=',' read -r -a TS <<< "$TIMESCALE"
    python exp_gesture.py --device 1 --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train/output/0815/gesture/v1.1/dynasnn/dynasnn-tau_middle --testnum 8 --test_droprate 0.2 --timescale1 ${TS[0]} --timescale2 ${TS[1]} --saveto test/0902/test/dynasnn-ts${TS[0]}_${TS[1]}
    python exp_gesture.py --device 1 --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train/output/0815/gesture/v1.1/dynasnn/dynasnn-tau_large  --testnum 8 --test_droprate 0.2 --timescale1 ${TS[0]} --timescale2 ${TS[1]} --saveto test/0902/test/dynasnn_large-ts${TS[0]}_${TS[1]}
    python exp_gesture.py --device 1 --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train/output/0815/gesture/v1.1/dynasnn/dynasnn-tau_small  --testnum 8 --test_droprate 0.2 --timescale1 ${TS[0]} --timescale2 ${TS[1]} --saveto test/0902/test/dynasnn_small-ts${TS[0]}_${TS[1]}
    
    python exp_gesture.py --device 1 --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train-multitime/output/0902/gesture/lstm                       --testnum 8 --test_droprate 0.2 --timescale1 ${TS[0]} --timescale2 ${TS[1]} --saveto test/0902/test/lstm-ts${TS[0]}_${TS[1]}
    python exp_gesture.py --device 1 --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train-multitime/output/0902/gesture/paramsnn/paramsnn-tau_middle        --testnum 8 --test_droprate 0.2 --timescale1 ${TS[0]} --timescale2 ${TS[1]} --saveto test/0902/test/paramsnn-ts${TS[0]}_${TS[1]}
    python exp_gesture.py --device 1 --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train-multitime/output/0902/gesture/paramsnn/paramsnn-tau_large         --testnum 8 --test_droprate 0.2 --timescale1 ${TS[0]} --timescale2 ${TS[1]} --saveto test/0902/test/paramsnn_large-ts${TS[0]}_${TS[1]}
    python exp_gesture.py --device 1 --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train-multitime/output/0902/gesture/paramsnn/paramsnn-tau_small         --testnum 8 --test_droprate 0.2 --timescale1 ${TS[0]} --timescale2 ${TS[1]} --saveto test/0902/test/paramsnn_small-ts${TS[0]}_${TS[1]}
    python exp_gesture.py --device 1 --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train-multitime/output/0902/gesture/snn/snn-tau_middle             --testnum 8 --test_droprate 0.2 --timescale1 ${TS[0]} --timescale2 ${TS[1]} --saveto test/0902/test/snn-ts${TS[0]}_${TS[1]}
    python exp_gesture.py --device 1 --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train-multitime/output/0902/gesture/snn/snn-tau_large              --testnum 8 --test_droprate 0.2 --timescale1 ${TS[0]} --timescale2 ${TS[1]} --saveto test/0902/test/snn_large-ts${TS[0]}_${TS[1]}
    python exp_gesture.py --device 1 --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train-multitime/output/0902/gesture/snn/snn-tau_small              --testnum 8 --test_droprate 0.2 --timescale1 ${TS[0]} --timescale2 ${TS[1]} --saveto test/0902/test/snn_small-ts${TS[0]}_${TS[1]}

done