MODELS=(
    "csnn"
    "rescsnn"
    "csnn_dropout"
    "csnn_batchnrm"
)

for TIMESCALE in {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0}
do
    for MODEL in ${MODELS[@]}
    do
        cpulimit -f -l 100 -- python gen_result_data.py --timescale $TIMESCALE --testnums 50 --device 3 --confpath configs/${MODEL}.yml --saveto 20241031_results/${MODEL}
    done
done


for MODEL in ${MODELS[@]}
do
    python json2csv.py --input 20241031_results/${MODEL}/json --output 20241031_results/${MODEL}
done