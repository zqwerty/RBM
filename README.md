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