MODEL_PATH_LIST=(
    /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train-multitime/output/20250118/paramsnn/paramsnn_tau0.012
    /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train-multitime/output/20250118/paramsnn/paramsnn_tau0.012_res1lay
    /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train-multitime/output/20250118/snn/snn_tau0.012
    /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train-multitime/output/20250118/snn/snn_tau0.012_res1lay
)

TIMESCALE_LIST=(
    0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
    1 2 3 4 5 6 7 8 9 10
)

SAVETO=20250118/result
TESTNUM=3
DEVICE=5

for TIMESCALE in ${TIMESCALE_LIST[@]};do
    for MODEL_PATH in ${MODEL_PATH_LIST[@]};do
        cpulimit -f -l 300 -- python gesture_acc_eval.py --target ${MODEL_PATH} --saveto ${SAVETO} --testnum ${TESTNUM} --test_droprate 0.3 --timescale ${TIMESCALE} --device ${DEVICE}
    done
done