for TIMESCALE in {1..20..1}
do
    python exp_gesture.py --device 1 --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train/output/0815/gesture/v1.1/dynasnn/dynasnn-tau_middle        --testnum 8 --test_droprate 0.2 --timescale ${TIMESCALE} --saveto test/0902/actual/dynasnn-ts${TIMESCALE} --is_video
    python exp_gesture.py --device 1 --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train/output/0815/gesture/v1.1/dynasnn/dynasnn-tau_large         --testnum 8 --test_droprate 0.2 --timescale ${TIMESCALE} --saveto test/0902/actual/dynasnn_large-ts${TIMESCALE}
    python exp_gesture.py --device 1 --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train/output/0815/gesture/v1.1/dynasnn/dynasnn-tau_small         --testnum 8 --test_droprate 0.2 --timescale ${TIMESCALE} --saveto test/0902/actual/dynasnn_small-ts${TIMESCALE}
    python exp_gesture.py --device 1 --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train-multitime/output/0902/gesture/paramsnn/paramsnn-tau_middle --testnum 8 --test_droprate 0.2 --timescale ${TIMESCALE} --saveto test/0902/actual/paramsnn-ts${TIMESCALE}
    python exp_gesture.py --device 1 --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train-multitime/output/0902/gesture/paramsnn/paramsnn-tau_large  --testnum 8 --test_droprate 0.2 --timescale ${TIMESCALE} --saveto test/0902/actual/paramsnn_large-ts${TIMESCALE}
    python exp_gesture.py --device 1 --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train-multitime/output/0902/gesture/paramsnn/paramsnn-tau_small  --testnum 8 --test_droprate 0.2 --timescale ${TIMESCALE} --saveto test/0902/actual/paramsnn_small-ts${TIMESCALE}
    python exp_gesture.py --device 1 --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train-multitime/output/0902/gesture/snn/snn-tau_large            --testnum 8 --test_droprate 0.2 --timescale ${TIMESCALE} --saveto test/0902/actual/snn_large-ts${TIMESCALE}
    python exp_gesture.py --device 1 --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train-multitime/output/0902/gesture/snn/snn-tau_middle           --testnum 8 --test_droprate 0.2 --timescale ${TIMESCALE} --saveto test/0902/actual/snn-ts${TIMESCALE}
    python exp_gesture.py --device 1 --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train-multitime/output/0902/gesture/snn/snn-tau_small            --testnum 8 --test_droprate 0.2 --timescale ${TIMESCALE} --saveto test/0902/actual/snn_small-ts${TIMESCALE}
    python exp_gesture.py --device 1 --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train-multitime/output/0902/gesture/lstm                         --testnum 8 --test_droprate 0.2 --timescale ${TIMESCALE} --saveto test/0902/actual/lstm-ts${TIMESCALE}
done