# Install 
```bash
git clone https://github.com/semihgunel/Df3dBehav
cd Df3dBehav
pip install -e .
```

# Run
```bash
df3dbehav --path '/folder/with_pose_result_file/'
```
This will save a file called df3d_behav.csv under the same folder.

# Analyze the Results
You can read the csv file using pandas:
```python
import pandas as pd
pd.read_csv('behav_clsf.csv')
```
![image](https://user-images.githubusercontent.com/20509861/123464401-0901a400-d5ed-11eb-844a-7a88eb44eadd.png)
