# src
スクリプトディレクトリ

## dynamic_snn
動的に時定数と膜抵抗を変動させるSNN.  
importすれば普通のpytorchのモデルと同じように使える.  

## generate dataset
学習用のデータを生成する  
データセットごとにスクリプトを分けている  

### gen_xor_v1
一番基本的なxorを生成するスクリプト  
xorは時間軸上で行い, 高密度を1, 低密度を0と捉える

### gen_xor_v2
こっちもXORを生成するスクリプト  
空間的にXORを生成する