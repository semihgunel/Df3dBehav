# Install 
```bash
pip install df3dbehav
```
or, you can install locally by:
```bash
git clone https://github.com/semihgunel/Df3dBehav/
cd df3dbehav
pip install -e .
```
# Run
```bash
df3dbehav --path '/folder/with_pose_result_file/'
```
It will automatically find the pose_result file and save a file called df3d_behav.csv under the same folder.

# Analyze the Results
You can read the csv file using pandas:
```python
import pandas as pd
pd.read_csv('behav_clsf.csv')
```
![image](https://user-images.githubusercontent.com/20509861/123464401-0901a400-d5ed-11eb-844a-7a88eb44eadd.png)


# Limitations
- Currently handles only canonical camera ordering.
- You might get strange REST classification in case you are using unexpected FPS values (outside of 30-100 range).
