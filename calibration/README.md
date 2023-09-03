# stereo-calibration

## Setup
CUDAバージョンのOpenCVが必要


Jetsonの場合、VideoStabilizationのリポジトリにあるinstall_opencv.shのスクリプトでインストール可能

### ビルド
Makefileを使用
```
make 
```

生成物削除
```
make clean
```


## Usage
main.cuのL27,28で読み込むを画像(L,R)指定
https://github.com/YZ775/stereo-calibration/blob/1c384279fb11811d69f935d32a31af32036f1f64/main.cu#L27

起動コマンド
```
./main
```
CPU+GPUバージョンを実行した後、続いてCPUのみバージョンを実行する


## ソースファイルに関して

- affine.cu アフィン変換行列適用GPUカーネル
- cpu.cu　不使用　ゴミファイル
- fast-cpu.cu　FASTのCPUカーネルコード
- fast.cu　FASTのGPUカーネルコード
- main-cpu.cu　 CPUのみで実行する際のメイン関数
- main.cu　CPU+GPUで実行する際のメイン関数
- match-cpu.cu　マッチングのCPUカーネルコード
- match.cu　マッチングのGPUカーネルコード
- postprocess-cpu.cu 3次元変換行列を推定・適用するCPUコード
- postprocess.cu　3次元変換行列を推定・適用するGPUコード
- sad.cu SADのGPUカーネルコード
- utils.cu　画像読み込みとSADの結果出力関数をまとめたコード

## それぞれのコードの説明
### affine.cu
### cpu.cu
### fast-cpu.cu
### fast.cu
### main-cpu.cu
### main.cu
以下の2カ所のコメントアウトを外すと、３次元変換の結果がそれぞれ出力される
https://github.com/YZ775/stereo-calibration/blob/1c384279fb11811d69f935d32a31af32036f1f64/main.cu#L267-L271
https://github.com/YZ775/stereo-calibration/blob/1c384279fb11811d69f935d32a31af32036f1f64/main.cu#L274-L278

### match-cpu.cu
### match.cu
### postprocess-cpu.cu
### postprocess.cu
### sad.cu
### utils.cu


