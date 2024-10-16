for TIMESCALE in {2..20..5}
do
    python disc1.1.py --device 3 --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202409_2MR/laplace-version/train-withEncoder/output/20241011/dynasnn/th35_tanh  --testnum 1 --droprate 0.3 --timescale ${TIMESCALE} --saveto output/20241011/result/dynasnn-tanh-ts${TIMESCALE}
done