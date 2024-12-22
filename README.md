# LLM Knowledge Analysis

Installation (close to same dependencies as Memit 2): 
```bash
pip install -r requirements.txt
```

I am currently using Python 3.9.7. 



### Main script: `analyze_knowledge_layers.py`

Settings: 
- Selecting a model and hparam file:
    ```python
    # EXAMPLE 
    model_name = /data/akshat/models/gpt2-xl
    hparams_filename = hparams/gpt2-xl.json
    ```
- Select a dataest: 
    ```python
    # EXAMPLE
    for pos_tag in ['REASONING', 'BASELINE', 'MULTIQNA']: 
    ```
- Enable -1 (previous token analysis): 
    ```python
    # EXAMPLE
    PREVIOUS_TOKEN = True
    ```
- Enable Tuned Lens: 
    ```python
    # EXAMPLE
    USING_TUNED = True
    ```
- Multiple Token analysis: Look at code for first/second/third token answers

After running code, dataset will appear in outer directory. You can then move it to `out/data` folder for graphing.

### Graphing script: `graph.ipynb`

Settings
- Select a model: 
    ```python
    # EXAMPLE
    model_name = 'gpt2-xl'
    ```
- Select a naming convention: 
    ```python
    # EXAMPLE
    dset_type = 'reasoning_tuned_-1'
    ```
- Selecting what you want plotted: 
    ```python
    # EXAMPLE
    pos_tags = ['MULTIQNA_tuned', 'MULTIQNA_second_tuned', 'MULTIQNA_third_tuned']
    ```
- If you want to analyze first/second/third token completions: 
    ```python
    # EXAMPLE
    ANALYZE_MULTIPLE_TOKENS = True
    ```

### Other scripts: 
Under `dsets` folder, you can find scripts for generating datasets for different tasks.
In `localization.py` was my attempt at localizing knowledge. 
