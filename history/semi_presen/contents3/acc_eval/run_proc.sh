MODEL_PATH_LIST=(
    /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train/output/0815/gesture/v1.1/dynasnn/dynasnn-tau_middle
    /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train-multitime/output/0902/gesture/paramsnn/paramsnn-tau_middle       
    /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train-multitime/output/0902/gesture/snn/snn-tau_middle            
    /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train-multitime/output/0902/gesture/lstm
)

TIMESCALE_LIST=(
    0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
    1 2 3 4 5 6 7 8 9 10
)

for TIMESCALE in ${TIMESCALE_LIST[@]};do
    for MODEL_PATH in ${MODEL_PATH_LIST[@]};do
        cpulimit -f -l 100 -- python gesture_acc_eval.py --target ${MODEL_PATH} --saveto 20241030/result --testnum 10 --test_droprate 0.3 --timescale ${TIMESCALE}
    done
done