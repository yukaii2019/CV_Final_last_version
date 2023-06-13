# 主要架構：

我們的project分成兩個part來train，可以看到有classifier，跟segmentation兩個資料夾，請分別進入兩個資料夾進行training。Inference的部分還有一個inference的資料夾來跑結果。

# Environment:
我們推薦使用conda來建置環境，安裝環境的方法如下。
```shell script=
conda env create -f environment.yml
```

# Dataset:
請準備一個資料夾，裡面包含，S1~S8，八個資料夾，格式如下：
```shell script=
dataset/
--S1/
----01/
------0.jpg
------0.png
----02/
------0.jpg
------0.png
--S2/
...
--S8/
```
另外我們有自己製作一個檔案叫做conf.json，裡面記的是training的圖片對應的開眼閉眼結果，這個檔案會被用在classifier及segmentation model的training。

# How to train the classifier:
In the directory "Classifier/python/"
```shell script=
bash run_train.sh ${dataset path}
```
一開始會生成一個log-{time}的資料夾在Classifier/ 目錄底下。最好的model會被留在Classifier/log-{time}/checkpoints底下。

# How to train the Segmentation model:
In the directory "Segmentation/python/"
```shell script=
bash run_train.sh ${dataset path}
```
一開始會生成一個log-{time}的資料夾在Segmentation/ 目錄底下。最好的model會被留在Segmentation/log-{time}/checkpoints底下。

# How to run inference:

## Download the checkpoint:
In the root directory:
```shell script=
bash download.sh
```
這會create一個資料夾叫CV_Final_Checkpoints，裡面會有兩個我們已經train好的checkpoint。

## Run inference:
我們已將使用的model的checkpoint的path hardcode在inference/run_inference.sh裡面，如果有需要，助教可以將inference/run_inference.sh裡面的classifier_checkpoint及segmatation_checkpoint兩個變數改成對應的路徑。

In the directory "inference/", run 
```shell script=
bash run_inference.sh ${dataset path} ${output path}
```
output path可以不用是已存在的資料夾。我們的inference預設是只跑S5~S8，此外最多只會跑26個sequence，如果不是這樣的話請助教自己修改inference/inference.py裡的第31行及35行。最後生成出來的格式會與上傳colab的相同。


