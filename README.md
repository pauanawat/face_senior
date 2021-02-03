### Classroom Smart Box with Face ID

## Available Scripts

In the project directory, you can run:

### `python run.py`

You can change process scale in line 8

### `python saveEmbeded.py` 

first, delete data in /storeEmbedding/embedding.npy and name.npy <br />
second, add your picture in /picture/store and set file name  `name`.jpg <br />
finally, run this command

### `python saveGT.py`

Before you test your picture, you must save ground truth <br />
first, delete data in gt.npy <br />
second, add your test picture in picture/forTest and set file name `name`.jpg <br />
finally, run this command

### `python test.py`

Test detection and recognition all picture in /forTest with scale 1, 0.5 and 0.25 
