for TIMESCALE in {1..20}
do
    python exp_gesture.py --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train/output/0815/gesture/v1.1/dynasnn/dynasnn-tau_middle --testnum 10 --test_droprate 0.2 --timescale ${TIMESCALE} --saveto test/0818/v1.0/dynasnn-ts${TIMESCALE} --is_video
    python exp_gesture.py --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train/output/0816/gesture/v1.1/paramsnn-tau_middle        --testnum 10 --test_droprate 0.2 --timescale ${TIMESCALE} --saveto test/0818/v1.0/paramsnn-ts${TIMESCALE}
    python exp_gesture.py --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train/output/0816/gesture/v1.1/snn-tau_middle             --testnum 10 --test_droprate 0.2 --timescale ${TIMESCALE} --saveto test/0818/v1.0/snn-ts${TIMESCALE}
    python exp_gesture.py --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train/output/0815/gesture/v1.1/dynasnn/dynasnn-tau_large  --testnum 10 --test_droprate 0.2 --timescale ${TIMESCALE} --saveto test/0818/v1.0/dynasnn_large-ts${TIMESCALE}
    python exp_gesture.py --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train/output/0815/gesture/v1.1/dynasnn/dynasnn-tau_small  --testnum 10 --test_droprate 0.2 --timescale ${TIMESCALE} --saveto test/0818/v1.0/dynasnn_small-ts${TIMESCALE}
    python exp_gesture.py --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train/output/0816/gesture/v1.1/paramsnn-tau_large         --testnum 10 --test_droprate 0.2 --timescale ${TIMESCALE} --saveto test/0818/v1.0/paramsnn_large-ts${TIMESCALE}
    python exp_gesture.py --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train/output/0816/gesture/v1.1/paramsnn-tau_small         --testnum 10 --test_droprate 0.2 --timescale ${TIMESCALE} --saveto test/0818/v1.0/paramsnn_small-ts${TIMESCALE}
    python exp_gesture.py --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train/output/0816/gesture/v1.1/snn-tau_large              --testnum 10 --test_droprate 0.2 --timescale ${TIMESCALE} --saveto test/0818/v1.0/snn_large-ts${TIMESCALE}
    python exp_gesture.py --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train/output/0816/gesture/v1.1/snn-tau_small              --testnum 10 --test_droprate 0.2 --timescale ${TIMESCALE} --saveto test/0818/v1.0/snn_small-ts${TIMESCALE}
    python exp_gesture.py --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train/output/0816/gesture/v1.0/lstm                       --testnum 10 --test_droprate 0.2 --timescale ${TIMESCALE} --saveto test/0818/v1.0/lstm-ts${TIMESCALE}


done