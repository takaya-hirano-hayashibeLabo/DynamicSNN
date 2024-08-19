TIMESCALES=(
    "1,1"
    "1,5"
    "5,1"
    "5,10"
    "10,5"
)

for TIMESCALE in "${TIMESCALES[@]}"; do
    IFS=',' read -r -a TS <<< "$TIMESCALE"
    python exp_gesture.py --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train/output/0815/gesture/v1.1/dynasnn/dynasnn-tau_middle --testnum 10 --test_droprate 0.2 --timescale1 ${TS[0]} --timescale2 ${TS[1]} --saveto test/0818/v1.0/dynasnn-ts${TS[0]}_${TS[1]} --is_video
    python exp_gesture.py --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train/output/0816/gesture/v1.1/paramsnn-tau_middle        --testnum 10 --test_droprate 0.2 --timescale1 ${TS[0]} --timescale2 ${TS[1]} --saveto test/0818/v1.0/paramsnn-ts${TS[0]}_${TS[1]}
    python exp_gesture.py --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train/output/0816/gesture/v1.1/snn-tau_middle             --testnum 10 --test_droprate 0.2 --timescale1 ${TS[0]} --timescale2 ${TS[1]} --saveto test/0818/v1.0/snn-ts${TS[0]}_${TS[1]}
    python exp_gesture.py --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train/output/0816/gesture/v1.0/lstm                       --testnum 10 --test_droprate 0.2 --timescale1 ${TS[0]} --timescale2 ${TS[1]} --saveto test/0818/v1.0/lstm-ts${TS[0]}_${TS[1]}
    python exp_gesture.py --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train/output/0815/gesture/v1.1/dynasnn/dynasnn-tau_large  --testnum 10 --test_droprate 0.2 --timescale1 ${TS[0]} --timescale2 ${TS[1]} --saveto test/0818/v1.0/dynasnn_large-ts${TS[0]}_${TS[1]}
    python exp_gesture.py --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train/output/0815/gesture/v1.1/dynasnn/dynasnn-tau_small  --testnum 10 --test_droprate 0.2 --timescale1 ${TS[0]} --timescale2 ${TS[1]} --saveto test/0818/v1.0/dynasnn_small-ts${TS[0]}_${TS[1]}
    python exp_gesture.py --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train/output/0816/gesture/v1.1/paramsnn-tau_large         --testnum 10 --test_droprate 0.2 --timescale1 ${TS[0]} --timescale2 ${TS[1]} --saveto test/0818/v1.0/paramsnn_large-ts${TS[0]}_${TS[1]}
    python exp_gesture.py --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train/output/0816/gesture/v1.1/paramsnn-tau_small         --testnum 10 --test_droprate 0.2 --timescale1 ${TS[0]} --timescale2 ${TS[1]} --saveto test/0818/v1.0/paramsnn_small-ts${TS[0]}_${TS[1]}
    python exp_gesture.py --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train/output/0816/gesture/v1.1/snn-tau_large              --testnum 10 --test_droprate 0.2 --timescale1 ${TS[0]} --timescale2 ${TS[1]} --saveto test/0818/v1.0/snn_large-ts${TS[0]}_${TS[1]}
    python exp_gesture.py --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train/output/0816/gesture/v1.1/snn-tau_small              --testnum 10 --test_droprate 0.2 --timescale1 ${TS[0]} --timescale2 ${TS[1]} --saveto test/0818/v1.0/snn_small-ts${TS[0]}_${TS[1]}

done