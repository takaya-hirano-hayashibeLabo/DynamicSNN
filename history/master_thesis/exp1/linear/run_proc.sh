TIMESCALE_LIST=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0)
TAU_LIST=(0.1 0.05 0.005)

for tau in "${TAU_LIST[@]}"; do
    for timescale in "${TIMESCALE_LIST[@]}"; do
        paramlist+=("$tau,$timescale")  # カンマ区切りで追加
    done
done

for param in "${paramlist[@]}"; do
    IFS=',' read -r tau timescale <<< "$param"  # カンマで分割してtauとtimescaleに代入
    # echo "timescale: $timescale, tau: $tau"
    cpulimit -f -l 200 -- python gen_result_data.py --timescale $timescale --testnums 50 --device 2 --tau $tau --track_batchsize 5 --batchsize 100
done
