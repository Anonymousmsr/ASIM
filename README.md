# ASIM

The code is modified from [RE2](https://github.com/alibaba-edu/simple-effective-text-matching-pytorch)

## Dependencies

- python >= 3.6
- Pytorch (1.0 <= version < 1.2). 
- `pip install -r requirements.txt`

## Usage

The format of the dataset is as follows, one sample per row: 

```
{Knowledge X}\t{Knowledge Y}\t{label}\n
```

To train a new model, run the following command: 

```bash
python train.py stackoverflow.json5
```

To evaluate the model, use `python evaluate.py $model_path $data_file`, here's an example:

```bash
python evaluate.py ASIM/data/stackoverflow/benchmark/best.pt data/stackoverflow/dev.txt 
```

