# THUMT

This is the experimental TensorFlow branch of THUMT, which is currently under
active development. 

To use this version, you need to add the path of THUMT to PYTHONPATH 
environment variable. You can add the following line to .bash_profile or 
.bashrc.
```
export PYTHONPATH="$PYTHONPATH:PATH/TO/THUMT"
```

## Basic Usage
### Training
```
python THUMT/thumt/launcher/trainer.py --input source.txt target.txt          
                                       --output train                         
                                       --vocabulary vocab.src vocab.tgt       
                                       --model rnnsearch                      
                                       --paramters=ADDITIONAL_PARAMTERS
```
There are several additional parameters that you can specify. 

### Decoding
```
python THUMT/thumt/launcher/translator.py --model rnnsearch                   
                                          --input input.txt                   
                                          --output output.txt                 
                                          --vocabulary vocab.src vocab.tgt    
                                          --path train/                       
                                          --parameters=ADDITIONAL_PARAMTERS
```
