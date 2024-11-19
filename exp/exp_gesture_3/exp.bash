TIMESCALES=(
    "1.0,5.0"
    "1.0,0.3"
    "0.3,5"
    "5,0.3"
)

for TIMESCALE in "${TIMESCALES[@]}"; do
    IFS=',' read -r -a TS <<< "$TIMESCALE"
    cpulimit -f -l 150 -- python exp_gesture.py --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train/output/0815/gesture/v1.1/dynasnn/dynasnn-tau_middle --device 1 --testnum 10 --test_droprate 0.2 --timescale1 ${TS[0]} --timescale2 ${TS[1]} --saveto test/20241116/dynasnn-ts${TS[0]}_${TS[1]} --is_video
    cpulimit -f -l 150 -- python exp_gesture.py --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train/output/0816/gesture/v1.1/paramsnn-tau_middle        --device 1 --testnum 10 --test_droprate 0.2 --timescale1 ${TS[0]} --timescale2 ${TS[1]} --saveto test/20241116/paramsnn-ts${TS[0]}_${TS[1]}
    cpulimit -f -l 150 -- python exp_gesture.py --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train/output/0816/gesture/v1.1/snn-tau_middle             --device 1 --testnum 10 --test_droprate 0.2 --timescale1 ${TS[0]} --timescale2 ${TS[1]} --saveto test/20241116/snn-ts${TS[0]}_${TS[1]}
    cpulimit -f -l 150 -- python exp_gesture.py --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train/output/0816/gesture/v1.0/lstm                       --device 1 --testnum 10 --test_droprate 0.2 --timescale1 ${TS[0]} --timescale2 ${TS[1]} --saveto test/20241116/lstm-ts${TS[0]}_${TS[1]}
    # python exp_gesture.py --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train/output/0815/gesture/v1.1/dynasnn/dynasnn-tau_large  --testnum 10 --test_droprate 0.2 --timescale1 ${TS[0]} --timescale2 ${TS[1]} --saveto test/0818/v1.0/dynasnn_large-ts${TS[0]}_${TS[1]}
    # python exp_gesture.py --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train/output/0815/gesture/v1.1/dynasnn/dynasnn-tau_small  --testnum 10 --test_droprate 0.2 --timescale1 ${TS[0]} --timescale2 ${TS[1]} --saveto test/0818/v1.0/dynasnn_small-ts${TS[0]}_${TS[1]}
    # python exp_gesture.py --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train/output/0816/gesture/v1.1/paramsnn-tau_large         --testnum 10 --test_droprate 0.2 --timescale1 ${TS[0]} --timescale2 ${TS[1]} --saveto test/0818/v1.0/paramsnn_large-ts${TS[0]}_${TS[1]}
    # python exp_gesture.py --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train/output/0816/gesture/v1.1/paramsnn-tau_small         --testnum 10 --test_droprate 0.2 --timescale1 ${TS[0]} --timescale2 ${TS[1]} --saveto test/0818/v1.0/paramsnn_small-ts${TS[0]}_${TS[1]}
    # python exp_gesture.py --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train/output/0816/gesture/v1.1/snn-tau_large              --testnum 10 --test_droprate 0.2 --timescale1 ${TS[0]} --timescale2 ${TS[1]} --saveto test/0818/v1.0/snn_large-ts${TS[0]}_${TS[1]}
    # python exp_gesture.py --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train/output/0816/gesture/v1.1/snn-tau_small              --testnum 10 --test_droprate 0.2 --timescale1 ${TS[0]} --timescale2 ${TS[1]} --saveto test/0818/v1.0/snn_small-ts${TS[0]}_${TS[1]}

done