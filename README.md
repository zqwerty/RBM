# RBM
Pattern Classification Project



## preprocess

* 10 label (1-10)
* 3 kind (3 pattern each number)
* 6 version (6 kind of noise)




* TEST
  * `(index)\[(label)\].png`



* digits:
  * `(label)_(index1).png[_n(index2).png]`
  * e.g. `0_1.png_n1.png`
  * index1: 该label第几个数字
  * index2: 第几个噪声版本
* hjk_picture:
  * `(index1).(label).png[_n(index2).png]`
  * `1.0.png_n1.png`
  * `5_2.png_n1.png`
  * `2.9png.png_n1.png` ???
* Li Wanjin
  * `(label).jpg[_n(index2).png]`
  * `0.jpg_n1.png`
  * `(index1)-(label).jpg[_n(index2).png]`
  * `1-0.jpg_n1.png`
* number
  * `(index1).(label).png[_n(index2).png]`



#### `preprocess.py:load_data(data_dir)`

```python
read pic from dir
:param data_dir
:return: (X,y) X: np.array(dataset_size, 32*32, dtype = float64);
                y: np.array(dataset_size, 32*32, dtype = int64);
```



## EXP

* sklearn:LR
  * 100% ???
* sklearn:RBM+LR
  * 89% ...
  * 我能怎么办，我也很无奈啊



/Users/zhuqi/.virtualenvs/nlp/bin/python /Users/zhuqi/Desktop/大三下/模式识别/homework/proj/RBM/src/RBM.py
SEARCHING LOGISTIC REGRESSION
Fitting 3 folds for each of 3 candidates, totalling 9 fits
[Parallel(n_jobs=-1)]: Done   4 out of   9 | elapsed:    1.5s remaining:    1.9s
[Parallel(n_jobs=-1)]: Done   9 out of   9 | elapsed:    2.2s finished
done in 3.005s
best score: 0.878
LOGISTIC REGRESSION PARAMETERS

	 C: 1.000000
SEARCHING RBM + LOGISTIC REGRESSION
Fitting 3 folds for each of 81 candidates, totalling 243 fits
[Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:   47.3s
[Parallel(n_jobs=-1)]: Done 184 tasks      | elapsed:  4.8min
[Parallel(n_jobs=-1)]: Done 243 out of 243 | elapsed:  6.5min finished

done in 407.554s
best score: 0.477
RBM + LOGISTIC REGRESSION PARAMETERS
	 logistic__C: 100.000000
	 rbm__learning_rate: 0.001000
	 rbm__n_components: 200.000000
	 rbm__n_iter: 80.000000

IMPORTANT
Now that your parameters have been searched, manually set
them and re-run this script with --search 0

Process finished with exit code 0



RBM+LR: 100%…………...

```
 logistic__C: 100.000000
 rbm__learning_rate: 0.001000
 rbm__n_components: 1000.000000
 rbm__n_iter: 80.000000
```

```
 logistic__C: 10000.000000
 rbm__learning_rate: 0.001000
 rbm__n_components: 200.000000
 rbm__n_iter: 20.000000
```

RBM+SVC: 100%……..

参数的影响很大啊