for TIMESCALE in {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0}
do
    cpulimit -f -l 100 -- python gen_result_data.py --timescale $TIMESCALE --testnums 50 --device 2 #-fをつけるとフォアグラウンドで実行され, ループが終わったら次のループに行ける
done

python json2csv.py
