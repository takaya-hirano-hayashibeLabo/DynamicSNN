# 20250117
paramsnnとsnnのresidual blockの活性化関数がreluだったのでlifにしてやり直し

## 結果
resnetの活性感化数をrelu -> lifにした  
しかし, ACCが90いかない...？  
biasをfalseにしたdynaなら余裕で行くのに...  
もしかすると, バイアスをつけるとうまくいくのかもしれない.  
そこで, biasをfalseにしてやり直してみる (-> 20251018へ)