## Laboratory 1

For the task involving encrypting secrets.yaml file, I had some problems with pgp key-pair generator 
so instead I used [age](https://github.com/FiloSottile/age) as mentioned [here](https://github.com/getsops/sops?tab=readme-ov-file#23encrypting-using-age).


In order tu run the webserver, you first need to run the training script.
```
uv run src/model/training.py 
```

The model is trained and saved in saved_models/ directory, then the webserver can be run.
