for DELAY in {0..500..50}
do
    python exp_gesture.py --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train/output/0815/gesture/v1.1/dynasnn/dynasnn-tau_middle --delay ${DELAY} --saveto test/0816/v2.1/dynasnn-d${DELAY} --is_video
    python exp_gesture.py --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train/output/0816/gesture/v1.1/paramsnn-tau_middle        --delay ${DELAY} --saveto test/0816/v2.1/paramsnn-d${DELAY}
    python exp_gesture.py --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train/output/0816/gesture/v1.1/snn-tau_middle             --delay ${DELAY} --saveto test/0816/v2.1/snn-d${DELAY}
    python exp_gesture.py --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train/output/0816/gesture/v1.0/lstm                       --delay ${DELAY} --saveto test/0816/v2.1/lstm-d${DELAY}
    python exp_gesture.py --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train/output/0815/gesture/v1.1/dynasnn/dynasnn-tau_large  --delay ${DELAY} --saveto  test/0816/v2.1/dynasnn_large-d${DELAY}
    python exp_gesture.py --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train/output/0815/gesture/v1.1/dynasnn/dynasnn-tau_small  --delay ${DELAY} --saveto  test/0816/v2.1/dynasnn_small-d${DELAY}
    python exp_gesture.py --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train/output/0816/gesture/v1.1/paramsnn-tau_large         --delay ${DELAY} --saveto test/0816/v2.1/paramsnn_large-d${DELAY}
    python exp_gesture.py --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train/output/0816/gesture/v1.1/paramsnn-tau_small         --delay ${DELAY} --saveto test/0816/v2.1/paramsnn_small-d${DELAY}
    python exp_gesture.py --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train/output/0816/gesture/v1.1/snn-tau_large              --delay ${DELAY} --saveto      test/0816/v2.1/snn_large-d${DELAY}
    python exp_gesture.py --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train/output/0816/gesture/v1.1/snn-tau_small              --delay ${DELAY} --saveto      test/0816/v2.1/snn_small-d${DELAY}
done